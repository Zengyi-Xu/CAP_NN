"""Virtual channel simulation.

Used for code debugging and maintenance without an M8190A/oscilloscope.
Includes:
1. First-order low-pass response (simulates TX high-frequency roll-off)
2. Receiver additive white Gaussian noise
3. Receiver third-order nonlinear distortion
4. Integer sample delay
"""
import numpy as np
import config


class VirtualChannel:
    """Virtual channel."""

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
            fs: sample rate (Hz)
            fc: first-order low-pass cutoff frequency (Hz)
            snr_db: receiver SNR (dB)
            nonlin_coeff: third-order nonlinearity coefficient
            delay: integer sample delay
            attenuation: linear amplitude attenuation
            seed: random seed, None means not fixed
        """
        self.fs = fs
        self.fc = fc
        self.snr_db = snr_db
        self.nonlin_coeff = nonlin_coeff
        self.delay = delay
        self.attenuation = attenuation
        self.rng = np.random.default_rng(seed)

    def first_order_lpf(self, x: np.ndarray) -> np.ndarray:
        """First-order low-pass in frequency domain: H(f) = 1 / (1 + j f/fc)."""
        x = np.asarray(x).ravel()
        N = len(x)
        freqs = np.fft.fftfreq(N, d=1.0 / self.fs)
        H = 1.0 / (1.0 + 1j * freqs / self.fc)
        return np.real(np.fft.ifft(np.fft.fft(x) * H))

    def add_noise(self, x: np.ndarray) -> np.ndarray:
        """Add AWGN according to SNR."""
        x = np.asarray(x).ravel()
        sig_pow = np.mean(x ** 2)
        noise_pow = sig_pow / (10.0 ** (self.snr_db / 10.0))
        noise = np.sqrt(noise_pow) * self.rng.standard_normal(len(x))
        return x + noise

    def apply_nonlinearity(self, x: np.ndarray) -> np.ndarray:
        """Third-order nonlinearity: y = x + coeff * x^3."""
        x = np.asarray(x).ravel()
        return x + self.nonlin_coeff * x ** 3

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply in order: attenuation -> first-order LPF -> nonlinearity -> noise -> delay.

        Zero-pad both ends so the output is longer than the input, facilitating cross-correlation synchronization.
        """
        x = np.asarray(x).ravel()
        y = x * self.attenuation
        y = self.first_order_lpf(y)
        y = self.apply_nonlinearity(y)
        y = self.add_noise(y)
        if self.delay != 0:
            y = np.roll(y, self.delay)
        # Zero-pad for synchronization
        pad = max(100, 4 * abs(self.delay))
        return np.concatenate([np.zeros(pad), y, np.zeros(pad)])

    def channel_response(self, N: int = 8192) -> tuple:
        """Return channel frequency response for plotting/analysis."""
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
    """Convenience function: apply the complete virtual channel to a waveform."""
    ch = VirtualChannel(fs=fs, fc=fc, snr_db=snr_db,
                        nonlin_coeff=nonlin_coeff, delay=delay,
                        attenuation=attenuation, seed=seed)
    return ch.apply(x)
