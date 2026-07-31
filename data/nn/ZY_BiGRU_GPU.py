# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 13:31:30 2026

@author: Xuzen
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# 自定义数据处理函数
def ANN_data_transfer(x, taps):
    x = x.reshape(-1, 1)
    z = int(len(x) - taps + 1)
    y = np.zeros((z, taps))
    for i in range(z):
        y[i] = x[i:i+taps, 0].T
    return y

def split_sequence(sequence, time_step):
    X = []
    for i in range(len(sequence) - time_step + 1):
        X.append(sequence[i:i+time_step])
    return np.array(X)

# 双向GRU模型
class EnhancedBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, time_step, output_dim):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size*2*time_step, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out.reshape(out.size(0), -1)
        return self.fc(out)

def load_and_process_data():
    """加载并处理数据，返回处理后的数据和原始数据"""
    # 加载原始数据
    data_Tx = np.loadtxt('./Txdata_NN.txt')
    data_Rx = np.loadtxt('./Rxdata_NN1.txt')
    
    # 归一化
    data_Tx = data_Tx / np.max(np.abs(data_Tx))
    data_Rx = data_Rx / np.max(np.abs(data_Rx))
    
    return data_Tx, data_Rx
def load_and_process_data2():
    """加载并处理数据，返回处理后的数据和原始数据"""
    # 加载原始数据
    data_Tx = np.loadtxt('./Txdata_NN.txt')
    data_Rx = np.loadtxt('./Rxdata_NN2.txt')
    
    # 归一化
    data_Tx = data_Tx / np.max(np.abs(data_Tx))
    data_Rx = data_Rx / np.max(np.abs(data_Rx))
    
    return data_Tx, data_Rx
def main():
    # 配置参数
    config = {
        'epochs': 8,
        'batch_size': 128,
        'ratio': 0.3,
        'output_dim': 1,
        'tap_num': 91,
        'time_step': 15,
        'feature_num': 91 + 1 - 15,
        'unit_num': 91 + 1 - 15,
        'seed': 15,
        'load_pretrained': True,  # 新增：是否加载预训练模型
        'pretrained_path': './pretrained_model.pth',  # 新增：预训练模型路径
        'save_model': True,  # 新增：是否保存训练好的模型
    }

    # 兼容：若 pretrained_model.pth 不存在而 trained_model_temp.pth 存在，则回退
    if not os.path.exists(config['pretrained_path']) and os.path.exists('./trained_model_temp.pth'):
        config['pretrained_path'] = './trained_model_temp.pth'

    # 设置随机种子
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

    # 设备检测
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载和处理数据
    data_Tx, data_Rx = load_and_process_data()
    
    # 数据划分
    x_train_temp = data_Rx[0:int(len(data_Rx)*config['ratio'])]
    y_train_temp = data_Tx[0:int(len(data_Tx)*config['ratio'])]
    x_test_temp = data_Rx[int(len(data_Rx)*config['ratio'])+1:]
    y_test_temp = data_Tx[int(len(data_Tx)*config['ratio'])+1:]

    # 数据处理
    x_train_temp2 = ANN_data_transfer(x_train_temp, config['feature_num'])
    x_train = split_sequence(x_train_temp2, config['time_step'])
    y_train_temp2 = y_train_temp[int((config['feature_num'] - config['output_dim'] + config['time_step'])/2): 
                               int(-(config['feature_num'] - config['output_dim'] + config['time_step'])/2 or None)]
    y_train = ANN_data_transfer(y_train_temp2, config['output_dim'])

    x_test_temp2 = ANN_data_transfer(x_test_temp, config['feature_num'])
    x_test = split_sequence(x_test_temp2, config['time_step'])
    y_test_temp2 = y_test_temp[int((config['feature_num'] - config['output_dim'] + config['time_step'])/2): 
                             int(-(config['feature_num'] - config['output_dim'] + config['time_step'])/2 or None)]
    y_test = ANN_data_transfer(y_test_temp2, config['output_dim'])

    # 转换为张量并发送到设备
    x_train_t = torch.FloatTensor(x_train).to(device)
    y_train_t = torch.FloatTensor(y_train[:, config['output_dim']//2]).unsqueeze(1).to(device)
    x_test_t = torch.FloatTensor(x_test).to(device)
    y_test_t = torch.FloatTensor(y_test[:, config['output_dim']//2]).unsqueeze(1).to(device)

    # 创建DataLoader
    train_dataset = TensorDataset(x_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_dataset = TensorDataset(x_test_t, y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])

    # 初始化模型
    model = EnhancedBiGRU(
        input_size=config['feature_num'],
        hidden_size=config['unit_num'],
        time_step=config['time_step'],
        output_dim=config['output_dim']
    ).to(device)

    # 新增：加载预训练模型
    if config['load_pretrained'] and os.path.exists(config['pretrained_path']):
        print(f"Loading pretrained model from {config['pretrained_path']}")
        model.load_state_dict(torch.load(config['pretrained_path'], map_location=device))
    else:
        print("Initializing new model from scratch")

    print(f"Model moved to: {next(model.parameters()).device}")

    # 训练配置
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    # 初始化可视化
    plt.figure(figsize=(12, 6))
    train_losses, val_losses = [], []

    # 训练循环
    for epoch in range(config['epochs']):
        model.train()
        epoch_train_loss = 0.0
        
        # 使用tqdm添加进度条（可通过环境变量 DISABLE_TQDM=1 关闭）
        disable_tqdm = os.environ.get("DISABLE_TQDM", "0") == "1"
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}',
                            leave=False, disable=disable_tqdm)
        for inputs, targets in progress_bar:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 验证
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                epoch_val_loss += criterion(outputs, targets).item()

        # 记录损失
        train_loss = epoch_train_loss / len(train_loader)
        val_loss = epoch_val_loss / len(test_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 更新学习率
        scheduler.step(val_loss)

        # 实时更新损失曲线
        plt.clf()
        plt.plot(train_losses, 'r-', label='Train Loss')
        plt.plot(val_losses, 'b-', label='Validation Loss')
        plt.title('Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.pause(0.1)  # 短暂暂停以更新图像

    # 保存最终训练曲线
    plt.savefig('training_curve.png')
    plt.close()
    
    # 新增：保存训练好的模型
    if config['save_model']:
        torch.save(model.state_dict(), 'trained_model_temp.pth')
        if config['load_pretrained']==False:
            torch.save(model.state_dict(), config['pretrained_path']+'trained_model.pth')
        print("Model saved to trained_model.pth")
    
    def calculate_output_length(input_len, feature_num, time_step):
        """计算经过两次滑动窗口处理后的输出长度"""
        first_step = input_len - feature_num + 1  # ANN_data_transfer后的长度
        final_length = first_step - time_step + 1  # split_sequence后的长度
        return final_length
    
    def calculate_required_padding(input_len, feature_num, time_step):
        """计算需要补充的padding长度"""
        output_len = calculate_output_length(input_len, feature_num, time_step)
        return input_len - output_len
    
    # 模型预测
    def predict(model, raw_data, feature_num, time_step, device, batch_size=128):
     """
     分批预测不爆显存 + 尾部自动补0到原始长度
     前面预测值完全不动，只在最后补0，不偏移、不扭曲数据
     """
     print("Prediction started (safe batch mode)")
     original_len = len(raw_data)  # 目标长度：214528
 
     # 你的原有逻辑 100% 保留
     total_pad = calculate_required_padding(original_len, feature_num, time_step)
     pad_left = total_pad // 2
     pad_right = total_pad - pad_left
     padded_data = np.pad(raw_data, (pad_left, pad_right), mode='constant')
 
     processed_data = ANN_data_transfer(padded_data, feature_num)
     input_data = split_sequence(processed_data, time_step)
 
     # 分批预测（和训练一样，不爆显存）
     input_tensor = torch.FloatTensor(input_data)
     infer_dataset = TensorDataset(input_tensor)
     infer_loader = DataLoader(infer_dataset, batch_size=batch_size, shuffle=False)
 
     model.eval()
     predictions = []
     with torch.no_grad():
         for batch in infer_loader:
             x = batch[0].to(device)
             pred = model(x).cpu().numpy()
             predictions.append(pred)
             del x, pred
             torch.cuda.empty_cache()
 
     predictions = np.concatenate(predictions).flatten()
     need_pad = original_len - len(predictions)
     if need_pad > 0:
         # 尾部补0，长度刚好等于 original_len
         predictions = np.concatenate([predictions, np.zeros(need_pad)])
     print("Prediction finished")
     print(f"Prediction length: {len(predictions)} (original: {original_len}, padding: {need_pad})")
     return predictions

    # 执行预测并可视化
    # print("Move data and model to CPU")
    # model = model.to('cpu') 
    
    # ch1
    # 训练结束后
    torch.cuda.empty_cache()  # 必须加
   # data_Tx, data_Rx = load_and_process_data()
    # 开始预测
    predictions = predict(
        model=model,
        raw_data=data_Rx, 
        feature_num=config['feature_num'],
        time_step=config['time_step'],
        device=device,        # 保持GPU/CPU都可以
        batch_size=128        # 和训练一样的批次
    )

    # 保存预测结果
    np.savetxt('Rxdata_afterNN1.txt', predictions, fmt='%.6f')
    plt.figure(figsize=(12, 6))
    plt.plot(predictions[:500], label='Predicted', alpha=0.8)
    plt.plot(data_Tx[:500], label='Ground Truth', alpha=0.6)
    plt.legend()
    plt.title('Prediction vs Ground Truth')
    plt.xlabel('Time Step')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig('prediction_result.png')
    plt.show()
    
    # # ch2
    # data_Tx, data_Rx = load_and_process_data2()
    # predictions = predict(
    #     model=model,
    #     raw_data=data_Rx,
    #     feature_num=config['feature_num'],
    #     time_step=config['time_step'],
    #     device=torch.device('cpu')
    # )
    
    # # 保存预测结果
    # np.savetxt('Rxdata_afterNN2.txt', predictions, fmt='%.6f')
    
    
    # plt.figure(figsize=(12, 6))
    # plt.plot(predictions[:500], label='Predicted', alpha=0.8)
    # plt.plot(data_Tx[:500], label='Ground Truth', alpha=0.6)
    # plt.legend()
    # plt.title('Prediction vs Ground Truth')
    # plt.xlabel('Time Step')
    # plt.ylabel('Amplitude')
    # plt.grid(True)
    # plt.savefig('prediction_result.png')
    # plt.show()

if __name__ == "__main__":
    print(torch.cuda.is_available())
    main()