import numpy as np
import scipy
from matplotlib import pyplot as plt

def fft(signal: list, length: float) -> tuple:
    n = signal.shape[-1]
    ft = scipy.fft.fft(signal)
    ftfreq = scipy.fft.fftfreq(n, d=length/n)
    spectrum = np.abs(ft)
    positive = ftfreq >= 0
    return (ftfreq[positive], spectrum[positive])

fe = 100
amp = 1.0
f = 1.0
puls = 2*np.pi*f
dt = 0.01
duration = 10.3
t = np.arange(0.0, duration, 1/fe)
s = amp*np.sin(puls*t)
n = t.shape[-1]
fft = scipy.fft.fft(s)
spectre = np.abs(fft)*2/n
fttfreq = scipy.fft.fftfreq(t.shape[-1], d=dt)
m = np.max(fft)
fond = fttfreq[np.where(fft == np.max(fft))[0]]
print(fond, f)

fig, ax = plt.subplots(1, 2)
ax[0].plot(t, s)
ax[1].plot(np.abs(fttfreq), spectre)
plt.show()