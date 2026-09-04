"""虚拟信道仿真。

用于在没有 M8190A/示波器的情况下进行代码调试和维护。
包括：
1. 一阶低通响应（模拟发送端高频滚降）
2. 接收端加性高斯白噪声
3. 接收端三阶非线性失真
4. 整数采样延迟
"""
import numpy as np
import config


class VirtualChannel:
    """虚拟信道。"""

    def __init__(self,
                 fs: float = config.AWG_SAMPLE_RATE,
                 fc: float = config.VIRTUAL_CHANNEL_FC,
                 snr_db: float = config.VIRTUAL_CHANNEL_SNR_DB,
                 nonlin_coeff: float = config.VIRTUAL_CHANNEL_NONLINEARITY,
                 delay: int = config.VIRTUAL_CHANNEL_DELAY,
                 attenuation: float = config.VIRTUAL_CHANNEL_ATTENUATION,
                 seed: int = None):
        """
        Args:
            fs: 采样率 (Hz)
            fc: 一阶低通截止频率 (Hz)
            snr_db: 接收端 SNR (dB)
            nonlin_coeff: 三阶非线性系数
            delay: 整数采样延迟
            attenuation: 线性幅度衰减
            seed: 随机种子，None 表示不固定
        """
        self.fs = fs
        self.fc = fc
        self.snr_db = snr_db
        self.nonlin_coeff = nonlin_coeff
        self.delay = delay
        self.attenuation = attenuation
        self.rng = np.random.default_rng(seed)

    def first_order_lpf(self, x: np.ndarray) -> np.ndarray:
        """频域一阶低通：H(f) = 1 / (1 + j f/fc)。"""
        x = np.asarray(x).ravel()
        N = len(x)
        freqs = np.fft.fftfreq(N, d=1.0 / self.fs)
        H = 1.0 / (1.0 + 1j * freqs / self.fc)
        return np.real(np.fft.ifft(np.fft.fft(x) * H))

    def add_noise(self, x: np.ndarray) -> np.ndarray:
        """按 SNR 添加 AWGN。"""
        x = np.asarray(x).ravel()
        sig_pow = np.mean(x ** 2)
        noise_pow = sig_pow / (10.0 ** (self.snr_db / 10.0))
        noise = np.sqrt(noise_pow) * self.rng.standard_normal(len(x))
        return x + noise

    def apply_nonlinearity(self, x: np.ndarray) -> np.ndarray:
        """三阶非线性：y = x + coeff * x^3。"""
        x = np.asarray(x).ravel()
        return x + self.nonlin_coeff * x ** 3

    def apply(self, x: np.ndarray) -> np.ndarray:
        """按顺序应用：衰减 -> 一阶低通 -> 非线性 -> 加噪 -> 延迟。

        在两端补零，使输出比输入更长，便于互相关同步。
        """
        x = np.asarray(x).ravel()
        y = x * self.attenuation
        y = self.first_order_lpf(y)
        y = self.apply_nonlinearity(y)
        y = self.add_noise(y)
        if self.delay != 0:
            y = np.roll(y, self.delay)
        # 为同步补零
        pad = max(100, 4 * abs(self.delay))
        return np.concatenate([np.zeros(pad), y, np.zeros(pad)])

    def channel_response(self, N: int = 8192) -> tuple:
        """返回用于绘图/分析的信道频率响应。"""
        freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1.0 / self.fs))
        H = 1.0 / (1.0 + 1j * freqs / self.fc)
        return freqs, H


def apply_virtual_channel(x: np.ndarray,
                          fs: float = config.AWG_SAMPLE_RATE,
                          fc: float = config.VIRTUAL_CHANNEL_FC,
                          snr_db: float = config.VIRTUAL_CHANNEL_SNR_DB,
                          nonlin_coeff: float = config.VIRTUAL_CHANNEL_NONLINEARITY,
                          delay: int = config.VIRTUAL_CHANNEL_DELAY,
                          attenuation: float = config.VIRTUAL_CHANNEL_ATTENUATION,
                          seed: int = None) -> np.ndarray:
    """便捷函数：对波形应用完整的虚拟信道。"""
    ch = VirtualChannel(fs=fs, fc=fc, snr_db=snr_db,
                        nonlin_coeff=nonlin_coeff, delay=delay,
                        attenuation=attenuation, seed=seed)
    return ch.apply(x)
