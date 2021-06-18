import numpy as np
import scipy
import scipy.signal
from matplotlib import pyplot as plt

from field import OneSpaceField

class FieldFourierTransform:
    def __init__(self, field: OneSpaceField, dx: float, dt: float):
        self.field = field
        self.dx = dx
        self.dt = dt

    def space_fft(self, xwindow: tuple, t: int) -> tuple:
        a, b = int(xwindow[0]), int(xwindow[1])
        field = self.field.get_val_time(t)
        signal = field[a:b]
        return FieldFourierTransform.fft(signal, self.dx)
    
    def time_fft(self, twindow: tuple, x: int) -> tuple:
        a, b = int(twindow[0]), int(twindow[1])
        field = self.field.get_val_pos(x)
        signal = field[a:b]
        return FieldFourierTransform.fft(signal, self.dt)
    
    def get_peaks(self, ft: list, h=0.0) -> list:
        """
            :return: list that each entry corresponds to a peak, the entry is a tuple containing indexes of: the left of the peak, the max of the peak, the right of the peak
            :rtype: list
        """
        spectrum = np.abs(ft)
        idx_peaks, _ = scipy.signal.find_peaks(spectrum, height=h)
        idx = np.array([i for i in range(0, ft.shape[-1])])
        peaks = []
        for peak in idx_peaks:
            fwhm = 0.5*spectrum[peak]
            under = spectrum <= fwhm
            bottom = idx[under]
            left = bottom < peak
            right = bottom > peak
            bottom_left = bottom[left]
            bottom_right = bottom[right]
            a, b = bottom_left[-1], bottom_right[0] # the interval of the peak
            peaks.append(np.array([a, peak, b]))
        return peaks

    @staticmethod
    def fft(signal: list, d: float) -> tuple:
        ft = scipy.fft.fft(signal)
        ftfreq = scipy.fft.fftfreq(signal.shape[-1], d=d)
        return (ftfreq, ft)

fe = 20
amp = 1.0
f = 1.0
puls = 2*np.pi*f
dx = 0.01
duration = 6.15
x = np.arange(0.0, duration, 1/fe)
idx = np.array([i for i in range(0, len(x))])
x = np.array([x])
n = x.shape[-1]
s = 1.0*np.sin(puls*x) # + 2.0*np.cos(3*puls*x)
field = OneSpaceField(s)
fft = FieldFourierTransform(field, dx, 1.0)
freq, tf = fft.space_fft((0, n), 0)
dk = np.abs(freq[0] - freq[1])
spec = np.abs(tf)
peaks, properties = scipy.signal.find_peaks(tf, height=0.1, width=0.0, prominence=0.0)
print(properties["width_heights"])
print(peaks)
plt.plot(freq, spec, "b")
plt.plot(freq[peaks], spec[peaks], "r.")
plt.vlines(x=freq[peaks], ymin=spec[peaks]-properties["prominences"], ymax=spec[peaks], color="r")

a = properties["left_ips"]
am = np.floor(a)
left_ips_freq = # np.interp(a, [am, am + 1], [freq[am], freq[am + 1]])

b = properties["right_ips"]
bm = np.floor(b)
right_ips_freq = # np.interp(b, [bm, bm + 1], [freq[bm], freq[bm + 1]])
plt.hlines(y=properties["width_heights"], xmin=( - middle_idx)/dx, xmax=(properties["right_ips"] - middle_idx)/dx, color="r")
plt.show()
