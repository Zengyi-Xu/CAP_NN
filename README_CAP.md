# UW_APSK_CAP_PY —— 面向水下可见光通信的 Python CAP 收发机

本项目将 MATLAB 版 UW_APSK CAP（无载波幅度相位，Carrierless Amplitude Phase）收发机
移植到 Python，复用 `DMT_PY_NN` 的工程框架。

## 范围

- **单频带 CAP** —— 移植自 `oldcapAPSKTxRx20220406.m`
- **多频带 CAP** —— 移植自 `main_CAP_3band_totalB.m`
- **APSK/QAM 星座图** —— 移植自 `GS_CCSDSmodulation_cons.m` / `GS_CCSDSdemodulation_cons.m`
- **LMS / LMS+Volterra 均衡器** —— 移植自 `LMS_1DownS_Testnan.m` / `LMS_volterra_1DownS_Testnan.m`
- **VLC 信道模型** —— 移植自 `vlc_channel.m`
- **NN 后均衡器** —— 移植自 SCAP_DNNpy2 工作流（多频带 CAP DNN）

## 文件对应关系

| Python 文件 | MATLAB 源文件 | 用途 |
|---|---|---|
| `constellation.py` | `GS_CCSDSmodulation_cons.m`, `GS_CCSDSdemodulation_cons.m` | APSK/QAM 映射/解映射 |
| `cap_core.py` | `CAPmod.m`, `cap_gen.m`, `shaping_fildes.m`, `Pulse_shaping_ZY.m` | CAP 调制、多频带合成 |
| `cap_rx.py` | `CAPmatch_filter.m`, `mdb_match_filter.m` | CAP 匹配滤波 / 解调 |
| `equalizer.py` | `LMS_1DownS_Testnan.m`, `LMS_volterra_1DownS_Testnan.m` | LMS 与 LMS+Volterra 均衡器 |
| `channel.py` | `vlc_channel.m` | VLC 信道 + AWGN |
| `config_cap.py` | — | CAP 专用参数 |
| `main_cap.py` | — | 命令行入口 |
| `data/nn/CAP_multiband_NN.py` | `SCAP_DNNpy2` | PyTorch BiGRU 后均衡器 |
| `nn_cap_equalizer.py` | — | CAP NN 的子进程封装 |

## 依赖要求

```bash
pip install -r requirements.txt
```

`torch` 仅 NN 均衡器（`--use-nn`）需要。请根据本机环境选用 CPU/CUDA 版本的安装包。

## 快速开始

### 运行测试

```bash
python test_cap.py
```

### 单频带 CAP 离线仿真

```bash
python main_cap.py --mode singleband --order 64 --constellation QAM --snr 27 --seed 100
```

### 多频带 CAP 离线仿真

```bash
python main_cap.py --mode multiband --order 16 --constellation QAM --snr 25 --seed 1
```

### 多频带 + NN 后均衡器

```bash
python main_cap.py --mode multiband --order 16 --use-nn
```

### GUI

```bash
python cap_gui.py
```

提供参数配置、一键运行、星座图/波形/频谱显示和结果面板。

## 配置说明

编辑 `config_cap.py` 可配置：

- 符号速率、采样速率、滚降系数
- AWG / 示波器的 VISA 地址
- 虚拟信道信噪比、非线性、衰落
- LMS/Volterra 的抽头数与步长
- NN 训练参数（位于 `data/nn/CAP_multiband_NN.py` 内）

## 黄金参考验证

`test_cap.py` 将 Python 生成的单频带 CAP 波形与 MATLAB 生成的
`data32QAM.txt` 进行比对。归一化最大差异低于 `1e-6`，
确认了脉冲成形路径的逐比特等价性。

## 备注

- MATLAB 工程使用的是 **CAP**，而非 DMT/OFDM。因此本 Python 移植实现的是
  CAP 调制/解调，而非基于 IFFT/FFT 的多载波方案。
- 多频带 CAP 的各频带有意相互重叠，因此未经均衡的匹配滤波误码率较高。
  可使用 LMS（`--no-lms` 禁用）或 NN（`--use-nn`）进行分离/均衡。
