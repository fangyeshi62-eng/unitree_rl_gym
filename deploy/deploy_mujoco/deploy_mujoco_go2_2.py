import time
import mujoco.viewer
import mujoco
import numpy as np
import torch
from pynput import keyboard

cmd_state = {
    "vx": 0.0,
    "vy": 0.0,
    "yaw": 0.0
}
def on_press(key):
    try:
        step_size = 0.1  # 每次按键增加的速度分量
        yaw_step = 0.2
        
        if key == keyboard.Key.up:
            cmd_state["vx"] += step_size
        elif key == keyboard.Key.down:
            cmd_state["vx"] -= step_size
        elif key == keyboard.Key.left:
            cmd_state["yaw"] += yaw_step
        elif key == keyboard.Key.right:
            cmd_state["yaw"] -= yaw_step
        elif key == keyboard.Key.space:
            cmd_state["vx"] = 0.0
            cmd_state["vy"] = 0.0
            cmd_state["yaw"] = 0.0
            print("\n[STOP] 指令已重置为 0")
        
        # 限制最大速度，防止机器人摔倒 (根据 Go2 的策略能力调整)
        cmd_state["vx"] = np.clip(cmd_state["vx"], -0.8, 1.2)
        cmd_state["yaw"] = np.clip(cmd_state["yaw"], -1.2, 1.2)
        
        print(f"\r当前指令 -> vx: {cmd_state['vx']:.2f}, yaw: {cmd_state['yaw']:.2f}    ", end="")
    except Exception as e:
        pass

# 启动后台监听线程
listener = keyboard.Listener(on_press=on_press)
listener.start()

def get_gravity_orientation(quaternion):
    # 保持你原始的四元数投影逻辑
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

if __name__ == "__main__":
    # --- 1. 参数配置 ---
    num_actions, num_obs = 12, 48
    lin_vel_scale, ang_vel_scale = 2.0, 0.25
    dof_pos_scale, dof_vel_scale = 1.0, 0.05
    action_scale = 0.25 
    
    default_angles = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5], dtype=np.float32)
    
    # 初始参数优化
    target_kps = np.full(12, 25.0) 
    kds = np.full(12, 0.5)          # 稍微调高阻尼，有助于吸收冲击
    control_decimation = 4 
    simulation_dt = 0.005 

    # --- 2. 环境初始化 ---
    model_path = "/root/gpufree-data/unitree_rl_gym-main/resources/robots/go2/scene.xml"
    policy_path = "/root/gpufree-data/unitree_rl_gym-main/deploy/pre_train/g2/policy_new.pt"
    
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    
    # 【补丁 1：精准初始化，防止穿模排斥】
    d.qpos[7:] = default_angles
    d.qpos[2] = 0.445              # Go2 身体中心离地约 0.3-0.34m
    d.qvel[:] = 0                 # 初始速度彻底清零
    mujoco.mj_forward(m, d)       # 计算运动学，消除初始应力

    policy = torch.jit.load(policy_path)
    obs = np.zeros(num_obs, dtype=np.float32)
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    
    # 指令设定
    target_cmd = np.array([0 , 0.5, 0]) 
    
    counter = 0
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # 设置初始相机角度（可选）
        viewer.cam.distance = 3.0  # 相机距离机器人的距离
        viewer.cam.azimuth = 132   # 水平角度
        viewer.cam.elevation = -20 # 俯仰角
        print("Control Ready: Use Arrow Keys to move, 'Space' to stop.")
        while viewer.is_running():
            step_start = time.time()
            viewer.cam.lookat[:] = d.body('base').xpos
            # 【补丁 2：KP 爬坡逻辑，防止开场“炸飞”】
            # 在前 1.0 秒内，KP 从 0 逐渐增加到 20
            # 这让机器人像慢慢醒过来，而不是突然被电击
            current_kp_scale = min(1.0, d.time / 1.0)
            current_kps = target_kps * current_kp_scale
            
            # --- 3. 物理仿真 step ---
            tau = pd_control(target_dof_pos, d.qpos[7:], current_kps, 0, d.qvel[6:], kds)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            
            # --- 4. 策略推断 ---
            if counter % control_decimation == 0:
                res_mat = np.zeros(9)
                mujoco.mju_quat2Mat(res_mat, d.qpos[3:7])
                rot_mat = res_mat.reshape(3, 3)
                rot_mat_T = rot_mat.T # 机体坐标系基向量

                qj, dqj = d.qpos[7:], d.qvel[6:]
                quat, omega = d.qpos[3:7], d.qvel[3:6]
                lin_vel = d.qvel[:3]
                # 获取传感器线速度
                #base_lin_vel = d.sensor('base_lin_vel').data.copy()
                
                # 【补丁 3：初始观测过滤】
                # 前 0.5 秒即使身体有晃动，我们也告诉网络速度为 0，防止它产生过大的纠偏动作
                if d.time < 0.5:
                    input_lin_vel_world = np.zeros(3)
                    current_cmd = np.zeros(3)
                else:
                    input_lin_vel_world = d.qvel[:3]
                    current_cmd = np.array([cmd_state["vx"], cmd_state["vy"], cmd_state["yaw"]])
                # 构造 Observation
                obs[0:3] = (rot_mat_T @ input_lin_vel_world)* lin_vel_scale 
                obs[3:6] = omega * ang_vel_scale
                obs[6:9] = get_gravity_orientation(quat)
                obs[9:12] = current_cmd * np.array([lin_vel_scale, lin_vel_scale, ang_vel_scale])
                obs[12:24] = (qj - default_angles) * dof_pos_scale
                obs[24:36] = dqj * dof_vel_scale
                obs[36:48] = action
                
                # 推理
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
                with torch.no_grad():
                    new_action = policy(obs_tensor).detach().numpy().squeeze()
                
                # 只有在 KP 稳定后才允许动作大幅度更新
                if d.time > 0.2:
                    target_dof_pos = action * action_scale + default_angles
                action = new_action.copy()
                # 打印调试信息
                if counter % 100 == 0:
                     print(f"Time: {d.time:.2f} | KP_Scale: {current_kp_scale:.2f} | Z-Vel: {lin_vel[2]:.2f}")
                     print(obs[0:3])
            counter += 1
            viewer.sync()
            
            # 频率控制
            time_to_sleep = simulation_dt - (time.time() - step_start)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)