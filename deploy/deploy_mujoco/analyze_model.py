import torch
import collections

def analyze_pt_model(model_path):
    print(f"正在分析模型文件: {model_path}")
    try:
        # 1. 尝试作为 TorchScript (JIT) 模型加载
        model = torch.jit.load(model_path, map_location='cpu')
        print("检测到模型类型: TorchScript (JIT)")
        
        # 2. 分析模型参数获取维度 (针对 GRU/MLP 层)
        print("\n--- 网络层维度详情 ---")
        state_dict = model.state_dict()
        
        # 按层名字排序显示
        for name, param in state_dict.items():
            print(f"层名: {name:40} | 形状: {list(param.shape)}")

        # 3. 专门识别输入维度 (通常是第一层)
        # 寻找 GRU 或第一个 Linear 层的输入
        input_dim = None
        if hasattr(model, 'memory'): # 针对你代码中的 GRU 结构
            input_dim = model.memory.input_size
            print(f"\n[确认] GRU 输入维度 (Input Size): {input_dim}")
        elif 'actor.0.weight' in state_dict: # 针对普通 MLP
            input_dim = state_dict['actor.0.weight'].shape[1]
            print(f"\n[确认] MLP 第一层输入维度: {input_dim}")
        
        # 4. 识别输出维度
        output_dim = None
        # 寻找最后一层线性层的输出
        last_layer_key = list(state_dict.keys())[-2] # 通常倒数第二个是最后一层的 weight
        if 'weight' in last_layer_key:
            output_dim = state_dict[last_layer_key].shape[0]
            print(f"[确认] 策略输出维度 (Action Size): {output_dim}")

        # 5. 打印 GRU 隐藏状态信息
        if hasattr(model, 'hidden_state'):
            print(f"[确认] GRU 隐藏状态维度: {list(model.hidden_state.shape)}")

    except Exception as e:
        print(f"加载模型失败: {e}")
        print("尝试作为普通 PyTorch 模型加载...")
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict):
                for k, v in checkpoint.items():
                    if hasattr(v, 'shape'):
                        print(f"键: {k:20} | 形状: {list(v.shape)}")
            else:
                print("该文件可能仅包含权重数据。")
        except Exception as e2:
            print(f"无法解析该文件: {e2}")

if __name__ == "__main__":
    # 将此路径替换为你实际的 pt 文件路径
    # MODEL_PATH = "/root/parkour/legged_gym/deploy/pre_train/g2/policy_lstm_1.pt" 
    MODEL_PATH = "/root/gpufree-data/unitree_rl_gym-main/deploy/pre_train/g2/policy_new.pt" 
    analyze_pt_model(MODEL_PATH)