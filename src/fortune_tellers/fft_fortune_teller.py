import numpy as np


class FFTFortuneTeller:
    def __init__(self):
        self.__asset_expected_returns = {}

    def make_prediction(self, times, prices):
        predicted_prices = self.fourier_extrapolation(prices, n_predict=100)
        return predicted_prices

    def fourier_extrapolation(self, x, n_predict):
        n = x.size
        n_harm = 50
        t = np.arange(0, n)
        p = np.polyfit(t, x, 1)
        x_notrend = x - p[0] * t
        x_freqdom = np.fft.fft(x_notrend)
        f = np.fft.fftfreq(n)
        indexes = list(range(n))
        indexes.sort(key=lambda i: np.absolute(f[i]))

        t = np.arange(0, n + n_predict)
        restored_sig = np.zeros(t.size)
        for i in indexes[:1 + n_harm * 2]:
            ampli = np.absolute(x_freqdom[i]) / n
            phase = np.angle(x_freqdom[i])
            restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
        restored_sig = restored_sig + p[0] * t

        return restored_sig[-n_predict:]

