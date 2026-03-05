import time
import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import os
from legged_gym import LEGGED_GYM_ROOT_DIR

def get_base_data(d):
    """ 获取局部坐标系下的线速度、角速度和重力投影 """
    quat = d.qpos[3:7] # MuJoCo format: [qw, qx, qy, qz]
    
    # 计算旋转矩阵
    res = np.zeros(9)
    mujoco.mju_quat2Mat(res, quat)
    R = res.reshape(3, 3)
    
    # 局部线速度: R^T * 世界线速度
    local_lin_vel = R.T @ d.qvel[0:3]
    # 局部角速度: 机器人的角速度通常直接在 body frame
    local_ang_vel = d.qvel[3:6]
    # 投影重力: R^T * [0, 0, -1]
    projected_gravity = R.T @ np.array([0, 0, -1])
    
    return local_lin_vel, local_ang_vel, projected_gravity

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name")
    args = parser.parse_args()

    # 1. 加载 YAML 配置
    config_path = os.path.join(LEGGED_GYM_ROOT_DIR, "deploy/deploy_mujoco/configs", args.config_file)
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

    # 2. 初始化 MuJoCo 模型 (解决 'm' is not defined)
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found at: {xml_path}")
    
    m = mujoco.MjModel.from_xml_path(xml_path) # 定义 m
    d = mujoco.MjData(m)                       # 定义 d
    m.opt.timestep = config["simulation_dt"]

    for i in range(12):
        # mjOBJ_JOINT 的索引从 1 开始（0是 root joint）
        joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i + 1)
        print(f"MuJoCo Joint {i}: {joint_name}")

    # 3. 加载策略
    policy = torch.jit.load(policy_path)

    # 4. 准备参数与变量
    num_obs = 48
    num_actions = 12
    control_decimation = config["control_decimation"]
    action_scale = config["action_scale"]
    
    # 缩放因子 (根据你的 compute_observations 逻辑)
    lin_vel_scale = 2.0
    ang_vel_scale = 0.25
    dof_pos_scale = 1.0
    dof_vel_scale = 0.05
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

    default_angles = np.array(config["default_angles"], dtype=np.float32)
    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    
    # 初始状态
    target_dof_pos = default_angles.copy()
    action = np.zeros(num_actions, dtype=np.float32)
    cmd = np.array(config["cmd_init"], dtype=np.float32)
    
    d.qpos[7:] = default_angles # 初始关节位置
    d.qpos[2] = 0.45            # 初始高度，防止摔地
    
    counter = 0

    # 5. 运行仿真渲染
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start_time = time.time()
        while viewer.is_running():
            step_start = time.time()

            # --- A. 底层 PD 控制 (高频) ---
            # 目标扭矩 = (目标角度 - 当前角度) * Kp + (0 - 当前速度) * Kd
            tau = (target_dof_pos - d.qpos[7:]) * kps + (0 - d.qvel[6:]) * kds
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            axis_fix = np.array([1, 1, -1,  1, 1, -1,  1, 1, -1,  1, 1, -1], dtype=np.float32)
            # --- B. 策略推理 (低频) ---
            # if counter % control_decimation == 0:
            #     base_lin, base_ang, proj_grav = get_base_data(d)
                
            #     obs = np.zeros(num_obs, dtype=np.float32)
            #     # 按照 compute_observations 源码顺序填充:
            #     obs[0:3]   = base_lin * lin_vel_scale
            #     obs[3:6]   = base_ang * ang_vel_scale
            #     obs[6:9]   = proj_grav
            #     obs[9:12]  = cmd * cmd_scale
            #     obs[12:24] = (d.qpos[7:] - default_angles) * dof_pos_scale
            #     obs[24:36] = d.qvel[6:] * dof_vel_scale
            #     obs[36:48] = action # 上一次的动作

            #     obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                
            #     action = policy(obs_tensor).detach().numpy().squeeze()
                
            #     # 更新目标位置
            #     target_dof_pos = action * action_scale + default_angles


            # ... 在推理循环 B 中 ...
# --- B. 策略推理 (低频) ---
            if counter % control_decimation == 0:
                base_lin, base_ang, proj_grav = get_base_data(d)
                
                # 构造 obs (确保这里的顺序和训练时完全一致)
                obs = np.zeros(num_obs, dtype=np.float32)
                obs[0:3]   = base_lin * lin_vel_scale
                obs[3:6]   = base_ang * ang_vel_scale
                obs[6:9]   = proj_grav
                obs[9:12]  = cmd * cmd_scale
                obs[12:24] = (d.qpos[7:] - default_angles) * dof_pos_scale
                obs[24:36] = d.qvel[6:] * dof_vel_scale
                obs[36:48] = action # 必须是上一次【未缩放】的原始网络输出

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                
                # 获取原始输出，不做任何加工
                with torch.no_grad():
                    action_tensor = policy(obs_tensor)
                
                # 关键：直接转为 numpy，不要在中间加任何逻辑
                action = action_tensor.cpu().numpy().flatten()
                
                # 计算目标位置
                target_dof_pos = action * action_scale + default_angles

                # raw_action = policy(obs_tensor).detach().numpy().squeeze()
                # raw_action[2] *= -1.0
                # raw_action[3] *= -1.0
                # action_for_control = raw_action[action_reorder_idx]
                # 6. 计算目标姿态
                # target_dof_pos = action_for_control * action_scale + default_angles

                if (counter // control_decimation) % 100 == 0:
                    print("-" * 50)
                    print(f"Step: {counter // control_decimation}")
                    print(f"Projected Gravity: {proj_grav} (应接近 [0, 0, -1])")
                    print(f"Local Lin Vel: {base_lin} (应随移动变化)")
                    print(f"Command Obs: {obs[9:12]} (检查指令是否正确输入网络)")
                    print(f"Policy Action (raw): {action[:4]}... (网络输出是否全为0?)")
                    print(f"Target DOF Pos (offset): {(target_dof_pos - default_angles)[:4]}...")
                    print(f"Base Height: {d.qpos[2]:.3f}")
                    print(policy(obs_tensor))
            counter += 1
            viewer.sync()

            # 保持实时仿真速度
            elapsed = time.time() - step_start
            if elapsed < m.opt.timestep:
                time.sleep(m.opt.timestep - elapsed)

            if time.time() - start_time > config["simulation_duration"]:
                break