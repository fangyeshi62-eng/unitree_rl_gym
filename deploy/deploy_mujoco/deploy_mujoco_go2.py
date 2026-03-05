import time
import mujoco.viewer
import mujoco
import numpy as np
import torch

# --- 工具函数 ---
def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qx * qx + qy * qy)
    return gravity_orientation

def quat_rotate_inverse(q, v):
    q_w = q[0]
    q_vec = q[1:4]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

# 线性插值函数：生成平滑路径
def linear_interpolation(now_pos, default_angles, num_steps):
    interpolation_points = np.zeros((num_steps + 1, len(now_pos)))
    interpolation_points[0] = now_pos
    step_size = (default_angles - now_pos) / num_steps
    for i in range(1, num_steps + 1):
        interpolation_points[i] = now_pos + i * step_size
    return interpolation_points

if __name__ == "__main__":
    # 1. 基础配置
    lin_vel_scale, ang_vel_scale = 2.0, 0.25
    dof_pos_scale, dof_vel_scale = 1.0, 0.05
    action_scale = 0.25
    simulation_dt = 0.005
    control_decimation = 4
    num_obs = 48

    # 2. 加载模型与策略
    model_path = "/root/gpufree-data/unitree_rl_gym-main/resources/robots/go2/scene.xml"
    policy_path = "/root/gpufree-data/unitree_rl_gym-main/deploy/pre_train/g2/policy_new.pt"
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    policy = torch.jit.load(policy_path)

    # 3. 关节映射 (防止顺序错位)
    actual_dof_names = [m.joint(i).name for i in range(m.njnt) if m.joint(i).type == mujoco.mjtJoint.mjJNT_HINGE]
    default_joint_angles_dict = {
        'FL_hip_joint': 0.1,   'FL_thigh_joint': 0.8,   'FL_calf_joint': -1.5,
        'FR_hip_joint': -0.1,  'FR_thigh_joint': 0.8,   'FR_calf_joint': -1.5,
        'RL_hip_joint': 0.1,   'RL_thigh_joint': 1.0,   'RL_calf_joint': -1.5,
        'RR_hip_joint': -0.1,  'RR_thigh_joint': 1.0,   'RR_calf_joint': -1.5,
    }
    default_angles = np.array([default_joint_angles_dict[name] for name in actual_dof_names], dtype=np.float32)
    kps = np.full(12, 20.0)
    kds = np.full(12, 0.5)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # --- [软启动阶段] ---
        print("正在进行软启动，请稍候...")
        current_q = d.qpos[7:].copy()
        # 400步，每步0.005s，共计2秒的平滑过渡
        soft_start_steps = 400 
        smooth_path = linear_interpolation(current_q, default_angles, soft_start_steps)

        for target_q in smooth_path:
            step_start = time.time()
            tau = pd_control(target_q, d.qpos[7:], kps, 0, d.qvel[6:], kds)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(max(0, simulation_dt - (time.time() - step_start)))
        print("软启动完成，进入策略控制模式。")

        # --- [策略控制阶段] ---
        obs = np.zeros(num_obs, dtype=np.float32)
        action = np.zeros(12, dtype=np.float32)
        target_dof_pos = default_angles.copy()
        cmd = np.array([0.5, 0.0, 0.0]) # 前进命令
        counter = 0

        while viewer.is_running():
            step_start = time.time()
            
            # 底层控制
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, 0, d.qvel[6:], kds)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)

            if counter % control_decimation == 0:
                # 状态获取与坐标转换
                v_world = d.qvel[0:3]
                quat = d.qpos[3:7]  # WXYZ
                omega = d.qvel[3:6]
                q, dq = d.qpos[7:], d.qvel[6:]

                base_lin_vel = quat_rotate_inverse(quat, v_world)

                # 填充 Observation (对齐 legged_robot.py)
                obs[0:3] = base_lin_vel * lin_vel_scale
                obs[3:6] = omega * ang_vel_scale
                obs[6:9] = get_gravity_orientation(quat)
                obs[9:12] = cmd * np.array([lin_vel_scale, lin_vel_scale, ang_vel_scale])
                obs[12:24] = (q - default_angles) * dof_pos_scale
                obs[24:36] = dq * dof_vel_scale
                obs[36:48] = action

                # 策略推理
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                with torch.no_grad():
                    action = policy(obs_tensor).detach().numpy().squeeze()
                
                target_dof_pos = action * action_scale + default_angles

            counter += 1
            viewer.sync()
            
            time_to_sleep = simulation_dt - (time.time() - step_start)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)