import time
import mujoco.viewer
import mujoco
import numpy as np
import torch

# --- 全局状态 ---
current_cmd = np.array([0.0, 0.0, 0.0]) # [vx, vy, yaw_rate]
STEP_SIZE_LIN = 0.1 
STEP_SIZE_ANG = 0.2 

def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

# --- 键盘回调函数 (使用方向键避开 MuJoCo 快捷键) ---
def key_callback(keycode):
    global current_cmd
    # 打印 keycode 帮助调试，如果你按了键没反应，看这里是否有输出
    print(f"Key Pressed: {keycode}") 
    
    if keycode == 265:     # Up Arrow
        current_cmd[0] += STEP_SIZE_LIN
    elif keycode == 264:   # Down Arrow
        current_cmd[0] -= STEP_SIZE_LIN
    elif keycode == 263:   # Left Arrow
        current_cmd[1] += STEP_SIZE_LIN
    elif keycode == 262:   # Right Arrow
        current_cmd[1] -= STEP_SIZE_LIN
    elif keycode == 81:    # Q Key (Yaw Left)
        current_cmd[2] += STEP_SIZE_ANG
    elif keycode == 69:    # E Key (Yaw Right)
        current_cmd[2] -= STEP_SIZE_ANG
    elif keycode == 32:    # Space (Reset)
        current_cmd[:] = 0.0
        print("\n[RESET] 指令已清零")

if __name__ == "__main__":
    # --- 1. 参数与路径 ---
    num_actions, num_obs = 12, 48
    lin_vel_scale, ang_vel_scale = 2.0, 0.25
    dof_pos_scale, dof_vel_scale = 1.0, 0.05
    action_scale = 0.25 
    default_angles = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5], dtype=np.float32)
    target_kps = np.full(12, 20.0) 
    kds = np.full(12, 0.8)
    control_decimation = 4 
    simulation_dt = 0.005 

    model_path = "/root/gpufree-data/unitree_rl_gym-main/resources/robots/go2/scene.xml"
    policy_path = "/root/gpufree-data/unitree_rl_gym-main/deploy/pre_train/g2/policy_new.pt"
    
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    
    # 初始化姿态
    d.qpos[7:] = default_angles
    d.qpos[2] = 0.34
    mujoco.mj_forward(m, d)

    policy = torch.jit.load(policy_path)
    obs = np.zeros(num_obs, dtype=np.float32)
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    
    counter = 0
    # --- 2. 启动仿真 ---
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # 绑定自定义键盘逻辑
        viewer.on_key_callback = key_callback
        
        print("\n" + "="*40)
        print("GO2 键盘控制模式 (避冲突版)")
        print("↑/↓: 前进后退 | ←/→: 左右平移")
        print("Q/E: 原地旋转 | Space: 停止")
        print("="*40 + "\n")

        while viewer.is_running():
            step_start = time.time()

            # KP 启动平滑逻辑
            current_kp_scale = min(1.0, d.time / 1.0)
            current_kps = target_kps * current_kp_scale
            
            # PD 控制器
            tau = pd_control(target_dof_pos, d.qpos[7:], current_kps, 0, d.qvel[6:], kds)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            
            # 策略网络更新
            if counter % control_decimation == 0:
                qj, dqj = d.qpos[7:], d.qvel[6:]
                quat, omega = d.qpos[3:7], d.qvel[3:6]
                base_lin_vel = d.sensor('base_lin_vel').data.copy()
                
                # 构造输入张量
                obs[0:3] = base_lin_vel * lin_vel_scale 
                obs[3:6] = omega * ang_vel_scale
                obs[6:9] = get_gravity_orientation(quat)
                obs[9:12] = current_cmd * np.array([lin_vel_scale, lin_vel_scale, ang_vel_scale])
                obs[12:24] = (qj - default_angles) * dof_pos_scale
                obs[24:36] = dqj * dof_vel_scale
                obs[36:48] = action
                
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
                with torch.no_grad():
                    action = policy(obs_tensor).detach().numpy().squeeze()
                
                if d.time > 0.4:
                    target_dof_pos = action * action_scale + default_angles
                
                # 实时状态显示
                if counter % 100 == 0:
                     print(f"指令 -> Vx: {current_cmd[0]:.1f}, Vy: {current_cmd[1]:.1f}, Yaw: {current_cmd[2]:.1f}", end='\r')

            counter += 1
            viewer.sync()

            # 仿真步长控制
            time_to_sleep = simulation_dt - (time.time() - step_start)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)