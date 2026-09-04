# -*- coding: utf-8 -*-
"""多频带 CAP 的简单 BiGRU 后均衡器。

输入：接收到的多频带 CAP 波形采样（Rxdata_NN1.txt）
输出：各频带的发送符号，实部/虚部交错排列
      （Rxdata_afterNN1.txt，形状为 (N, 6)）

移植自 MATLAB main_CAP_3band_totalB.m 的 SCAP_DNNpy2 工作流。
"""
import os
import sys
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


def _nn_progress(epoch=None, total=None, train_loss=None, val_loss=None, phase="train"):
    if os.environ.get("DISABLE_TQDM", "0") != "1":
        return
    payload = {"phase": phase}
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if total is not None:
        payload["total"] = int(total)
    if train_loss is not None:
        payload["train_loss"] = float(train_loss)
    if val_loss is not None:
        payload["val_loss"] = float(val_loss)
    print(f"[NN_PROGRESS] {json.dumps(payload, ensure_ascii=False)}", flush=True)


def split_sequence(sequence, time_step):
    X = []
    for i in range(len(sequence) - time_step + 1):
        X.append(sequence[i:i + time_step])
    return np.array(X)


class CAPBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, time_step, output_dim):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size * 2 * time_step, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out.reshape(out.size(0), -1)
        return self.fc(out)


def main():
    config = {
        "epochs": 20,
        "batch_size": 256,
        "ratio": 0.3,
        "output_dim": 6,          # 3 个频带 ×（实部，虚部）
        "tap_num": 91,            # 输入特征窗口长度
        "time_step": 15,
        "hidden_size": 128,
        "seed": 15,
        "learning_rate": 1e-3,
        "save_model": True,
        "model_path": "./cap_multiband_model.pth",
    }

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    data_tx = np.loadtxt("./Txdata_NN.txt")  # (N, 6)
    data_rx = np.loadtxt("./Rxdata_NN1.txt")  # (N, 6) 匹配滤波输出

    # 逐列归一化
    tx_max = np.max(np.abs(data_tx), axis=0, keepdims=True)
    rx_max = np.max(np.abs(data_rx), axis=0, keepdims=True)
    tx_max[tx_max == 0] = 1.0
    rx_max[rx_max == 0] = 1.0
    data_tx = data_tx / tx_max
    data_rx = data_rx / rx_max

    n_train = int(len(data_rx) * config["ratio"])
    x_train_temp = data_rx[:n_train]
    y_train_temp = data_tx[:n_train]
    x_test_temp = data_rx[n_train:]
    y_test_temp = data_tx[n_train:]

    # 构建窗口：索引 k 处的每个输出样本使用 rx[k:k+tap_num]
    tap_num = config["tap_num"]
    time_step = config["time_step"]
    output_dim = config["output_dim"]

    def build_windows(x, y):
        x_win = []
        y_win = []
        for k in range(len(x) - tap_num + 1):
            x_win.append(x[k:k + tap_num])
            # 目标是窗口的中心样本
            centre = k + tap_num // 2
            if centre < len(y):
                y_win.append(y[centre])
            else:
                y_win.append(y[-1])
        x_seq = split_sequence(np.array(x_win), time_step)
        # 对齐 y：每个 x_seq[i] 对应窗口 i..i+time_step-1
        y_seq = np.array(y_win[time_step - 1:])
        return x_seq, y_seq

    x_train, y_train = build_windows(x_train_temp, y_train_temp)
    x_test, y_test = build_windows(x_test_temp, y_test_temp)

    x_train_t = torch.FloatTensor(x_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    x_test_t = torch.FloatTensor(x_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)

    train_dataset = TensorDataset(x_train_t, y_train_t)
    test_dataset = TensorDataset(x_test_t, y_test_t)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    model = CAPBiGRU(
        input_size=output_dim,
        hidden_size=config["hidden_size"],
        time_step=time_step,
        output_dim=output_dim,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    disable_tqdm = os.environ.get("DISABLE_TQDM", "0") == "1"

    for epoch in range(config["epochs"]):
        model.train()
        epoch_train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"第 {epoch + 1}/{config['epochs']} 轮",
                            leave=False, disable=disable_tqdm)
        for inputs, targets in progress_bar:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            progress_bar.set_postfix({"损失": f"{loss.item():.4f}"})

        avg_train_loss = epoch_train_loss / len(train_loader)

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item()
        avg_val_loss = epoch_val_loss / len(test_loader)
        scheduler.step(avg_val_loss)

        _nn_progress(epoch=epoch + 1, total=config["epochs"],
                     train_loss=avg_train_loss, val_loss=avg_val_loss)
        print(f"第 {epoch + 1}/{config['epochs']} 轮  训练损失={avg_train_loss:.6f}  验证损失={avg_val_loss:.6f}")

    # 在完整接收数据上进行预测
    model.eval()
    x_full, _ = build_windows(data_rx, data_tx)
    x_full_t = torch.FloatTensor(x_full).to(device)
    with torch.no_grad():
        predictions = model(x_full_t).cpu().numpy()

    # 反归一化
    predictions = predictions * tx_max[time_step - 1:time_step - 1 + len(predictions)]
    np.savetxt("./Rxdata_afterNN1.txt", predictions)
    print(f"预测结果已保存到 ./Rxdata_afterNN1.txt，形状为 {predictions.shape}")

    if config["save_model"]:
        torch.save(model.state_dict(), config["model_path"])
        print(f"模型已保存到 {config['model_path']}")


if __name__ == "__main__":
    main()
