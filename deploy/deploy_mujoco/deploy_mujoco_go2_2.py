# import time
# import mujoco.viewer
# import mujoco
# import numpy as np
# import torch
# import yaml
# from legged_gym import LEGGED_GYM_ROOT_DIR

# def get_gravity_orientation(quaternion):
#     # 此处 quaternion 假设为 [w, x, y, z]
#     qw, qx, qy, qz = quaternion
#     gravity_orientation = np.zeros(3)
#     gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
#     gravity_orientation[1] = -2 * (qz * qy + qw * qx)
#     gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
#     return gravity_orientation

# def pd_control(target_q, q, kp, target_dq, dq, kd):
#     return (target_q - q) * kp + (target_dq - dq) * kd

# if __name__ == "__main__":
#     # --- 1. 配置加载 (建议通过 YAML 或直接从 config 类中提取) ---
#     # 依据 go2_config.py
#     num_actions = 12
#     num_obs = 48 # 依据 legged_robot_config.py 的 env.num_observations
    
#     # 缩放系数 (依据 legged_robot_config.py)
#     lin_vel_scale = 2.0
#     ang_vel_scale = 0.25
#     dof_pos_scale = 1.0
#     dof_vel_scale = 0.05
#     action_scale = 0.25 # 依据 go2_config.py
    
#     # 默认关节角度 (依据 go2_config.py, 顺序需与 Mujoco xml 一致)
#     # 注意：这里的顺序必须严格匹配 Mujoco m.dof_names 或 d.qpos[7:] 的排列
#     default_angles = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5], dtype=np.float32)
    
#     # 控制参数
#     kps = np.full(12, 20.0) # 依据 go2_config.py stiffness
#     kds = np.full(12, 0.5)  # 依据 go2_config.py damping
#     control_decimation = 4  # 依据 go2_config.py
#     simulation_dt = 0.005   # 依据 legged_robot_config.py sim.dt

#     # --- 2. 环境初始化 ---
#     model_path = "/root/gpufree-data/unitree_rl_gym-main/resources/robots/go2/scene.xml"
#     policy_path = "/root/gpufree-data/unitree_rl_gym-main/deploy/pre_train/g2/policy_new.pt"
    
#     m = mujoco.MjModel.from_xml_path(model_path)
#     d = mujoco.MjData(m)
#     m.opt.timestep = simulation_dt
#     policy = torch.jit.load(policy_path)
    
#     obs = np.zeros(num_obs, dtype=np.float32)
#     action = np.zeros(num_actions, dtype=np.float32)
#     target_dof_pos = default_angles.copy()
#     cmd = np.array([0.5, 0, 0]) # 设定一个前进 0.5m/s 的指令
    
#     counter = 0
#     with mujoco.viewer.launch_passive(m, d) as viewer:
#         start_time = time.time()
#         while viewer.is_running():
#             step_start = time.time()
            
#             # --- 3. 物理仿真 step ---
#             # 底层 PD 控制
#             tau = pd_control(target_dof_pos, d.qpos[7:], kps, 0, d.qvel[6:], kds)
#             d.ctrl[:] = tau
#             mujoco.mj_step(m, d)
            
#             # --- 4. 策略网络推断 (按 Decimation 频率执行) ---
#             if counter % control_decimation == 0:
#                 # 获取状态
#                 qj = d.qpos[7:]
#                 dqj = d.qvel[6:]
#                 quat = d.qpos[3:7] # Mujoco 默认为 [w, x, y, z]
#                 omega = d.qvel[3:6]
                
#                 # 构造 Observation (必须严格遵守 legged_robot.py 的顺序)
#                 # 1. Base Lin Vel (Mujoco 中通常无法直接从传感器获得完美线速度，实际部署需状态估计器)
#                 # 此处 Sim2Sim 暂用 0 或从 d.qvel[0:3] 模拟获取
#                 base_lin_vel = d.sensor('base_lin_vel').data    # 自动就是机体系速度！
#                 obs[0:3] = base_lin_vel * lin_vel_scale 
#                 # 打印局部线速度的前向分量
#                 print(f"Body X Vel: {base_lin_vel[0]:.2f}")
#                 print(f"Body Y Vel: {base_lin_vel[1]:.2f}")      
#                 print(f"Body Z Vel: {base_lin_vel[2]:.2f}")      
#                 # 2. Base Ang Vel
#                 obs[3:6] = omega * ang_vel_scale
                
#                 # 3. Projected Gravity
#                 obs[6:9] = get_gravity_orientation(quat)
                
#                 # 4. Commands
#                 obs[9:12] = cmd * np.array([lin_vel_scale, lin_vel_scale, ang_vel_scale])
                
#                 # 5. DOF Pos
#                 obs[12:24] = (qj - default_angles) * dof_pos_scale
                
#                 # 6. DOF Vel
#                 obs[24:36] = dqj * dof_vel_scale
                
#                 # 7. Previous Actions
#                 obs[36:48] = action
#                 print(obs)
#                 # 推理
#                 obs_tensor = torch.from_numpy(obs).unsqueeze(0)
#                 action = policy(obs_tensor).detach().numpy().squeeze()
                
#                 # 映射到目标关节角度
#                 target_dof_pos = action * action_scale + default_angles
                
#             counter += 1
#             viewer.sync()
            
#             # 控制仿真频率
#             time_to_sleep = simulation_dt - (time.time() - step_start)
#             if time_to_sleep > 0:
#                 time.sleep(time_to_sleep)
import time
import mujoco.viewer
import mujoco
import numpy as np
import torch

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
    target_kps = np.full(12, 20.0) 
    kds = np.full(12, 0.8)          # 稍微调高阻尼，有助于吸收冲击
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
    d.qpos[2] = 0.32              # Go2 身体中心离地约 0.3-0.34m
    d.qvel[:] = 0                 # 初始速度彻底清零
    mujoco.mj_forward(m, d)       # 计算运动学，消除初始应力

    policy = torch.jit.load(policy_path)
    obs = np.zeros(num_obs, dtype=np.float32)
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    
    # 指令设定
    target_cmd = np.array([0.5, 0, 0]) 
    
    counter = 0
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
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
                qj, dqj = d.qpos[7:], d.qvel[6:]
                quat, omega = d.qpos[3:7], d.qvel[3:6]
                
                # 获取传感器线速度
                base_lin_vel = d.sensor('base_lin_vel').data.copy()
                
                # 【补丁 3：初始观测过滤】
                # 前 0.5 秒即使身体有晃动，我们也告诉网络速度为 0，防止它产生过大的纠偏动作
                if d.time < 0.5:
                    input_lin_vel = np.zeros(3)
                    cmd = np.zeros(3) # 延迟执行移动指令
                else:
                    input_lin_vel = base_lin_vel
                    cmd = target_cmd

                # 构造 Observation
                obs[0:3] = input_lin_vel * lin_vel_scale 
                obs[3:6] = omega * ang_vel_scale
                obs[6:9] = get_gravity_orientation(quat)
                obs[9:12] = cmd * np.array([lin_vel_scale, lin_vel_scale, ang_vel_scale])
                obs[12:24] = (qj - default_angles) * dof_pos_scale
                obs[24:36] = dqj * dof_vel_scale
                obs[36:48] = action
                
                # 推理
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
                with torch.no_grad():
                    action = policy(obs_tensor).detach().numpy().squeeze()
                
                # 只有在 KP 稳定后才允许动作大幅度更新
                if d.time > 0.2:
                    target_dof_pos = action * action_scale + default_angles
                
                # 打印调试信息
                if counter % 100 == 0:
                     print(f"Time: {d.time:.2f} | KP_Scale: {current_kp_scale:.2f} | Z-Vel: {base_lin_vel[2]:.2f}")

            counter += 1
            viewer.sync()
            
            # 频率控制
            time_to_sleep = simulation_dt - (time.time() - step_start)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)