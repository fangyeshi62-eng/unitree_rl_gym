import time
import os
import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import sys

# 假设 LEGGED_GYM_ROOT_DIR 已在环境中或手动定义
# from legged_gym import LEGGED_GYM_ROOT_DIR 

# --- 完全复刻第一个文件的辅助函数 ---

def update_command(data, cmd, heading_stiffness, heading_target, heading_command=True):
    """ 处理航向角逻辑，完全对齐第一个文件 """
    if heading_command:
        # 注意：这里需要你环境中 utils 包含这两个函数，或者手动实现 wrap_to_pi
        import utils
        current_heading = utils.quat_to_heading_w(data.qpos[3:7])
        heading_err = utils.wrap_to_pi(heading_target - current_heading)
        cmd[2] = np.clip(heading_err * heading_stiffness, -1, 1)
    return cmd

def get_gravity_orientation(quaternion):
    """ 完全对齐第一个文件的重力计算 (MuJoCo quat: [qw, qx, qy, qz]) """
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    # 第一个文件中的特定公式
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """ 计算 PD 力矩 """
    return (target_q - q) * kp + (target_dq - dq) * kd

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name")
    args = parser.parse_args()

    # 1. 加载配置
    # 路径处理建议保持你原来的 LEGGED_GYM_ROOT_DIR 逻辑
    from legged_gym import LEGGED_GYM_ROOT_DIR
    config_path = os.path.join(LEGGED_GYM_ROOT_DIR, "deploy/deploy_mujoco/configs", args.config_file)
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

    # 2. 初始化 MuJoCo
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = config["simulation_dt"]

    # 3. 加载策略
    policy = torch.jit.load(policy_path)
    print(f"Loaded policy from {policy_path}")

    # 4. 从 Config 读取参数 (完全对齐第一个文件的变量名)
    simulation_duration = config["simulation_duration"]
    control_decimation = config["control_decimation"]
    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    
    lin_vel_scale = config["lin_vel_scale"]
    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale = config["action_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

    num_actions = config["num_actions"]
    num_obs = 48 # 强制 48 维
    policy2model = np.array(config["mapping_joints"], dtype=np.int32)
    
    cmd = np.array(config["cmd_init"], dtype=np.float32)
    heading_stiffness = config["heading_stiffness"]
    heading_target = config["heading_target"]
    heading_command = config["heading_command"]

    # 运行状态变量
    action_policy_prev = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0

    # 5. 仿真渲染
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # 初始相机设置
        viewer.cam.azimuth = 0
        viewer.cam.elevation = -20
        viewer.cam.distance = 1.5
        
        start_time = time.time()
        while viewer.is_running() and time.time() - start_time < simulation_duration:
            step_start = time.time()

            if config.get("lock_camera", True):
                viewer.cam.lookat[:] = d.qpos[:3]

            counter += 1
            # --- 策略推理逻辑 (按 control_decimation 频率) ---
            if counter % control_decimation == 0:
                # 1. 更新指令
                cmd = update_command(d, cmd, heading_stiffness, heading_target, heading_command)

                # 2. 获取原始数据 (对齐第一个文件：直接取 qvel 和 qpos)
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]     # [qw, qx, qy, qz]
                lin_vel = d.qvel[:3]   # 全局线速度
                ang_vel = d.qvel[3:6]  # 角速度

                # 3. 缩放与映射
                qj_scaled = (qj - default_angles) * dof_pos_scale
                dqj_scaled = dqj * dof_vel_scale
                
                # 映射到 Policy 顺序
                qj_policy = qj_scaled[policy2model]
                dqj_policy = dqj_scaled[policy2model]
                
                gravity_orientation = get_gravity_orientation(quat)
                
                # 4. 填充 48 维 Obs (严格按第一个文件的 if num_obs == 48 顺序)
                obs[:3] = lin_vel * lin_vel_scale
                obs[3:6] = ang_vel * ang_vel_scale
                obs[6:9] = gravity_orientation
                obs[9:12] = cmd * cmd_scale
                obs[12:24] = qj_policy
                obs[24:36] = dqj_policy
                obs[36:48] = action_policy_prev

                # 5. Policy 推理
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                with torch.no_grad():
                    action_policy = policy(obs_tensor).numpy().flatten()

                # 6. 动作后处理
                # 映射回 Model 顺序
                target_dof_pos = action_policy * action_scale + default_angles
                action_policy_prev[:] = action_policy
# --- 每 100 次推理打印一次关键数据 ---
                if counter % 100 == 0:
                    print(f"\n{'='*20} Step {counter} {'='*20}")
                    print(f"Lin Vel (scaled): {obs[:3]}")
                    print(f"Ang Vel (scaled): {obs[3:6]}")
                    print(f"Gravity Project: {obs[6:9]}  (Should be near [0,0,-1] if flat)")
                    print(f"Commands (scaled): {obs[9:12]}")
                    print(f"Action (raw, first 3): {action_policy[:3]}")
                    print(f"Base Height: {d.qpos[2]:.3f}")
                    print(f"{'='*50}")
                # 7. PD 控制步进 (对齐第一个文件的 for _ in range(4) 逻辑或类似频率)
                # 注：第一个文件是在 if 块内跑了 4 次步进，我们这里为了平滑可以根据你的频率调整
                for _ in range(4): 
                    tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
                    d.ctrl[:] = tau
                    mujoco.mj_step(m, d)

            viewer.sync()

            # 时间补偿
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)