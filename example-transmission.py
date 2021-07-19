from datetime import time
import os

from quantumstring.simulation import CenterFixed, FreeString, RingString, Simulation, Cavity
from quantumstring.edge import ExcitatorSinAbsorber, MirrorEdge, ExcitatorSin, AbsorberEdge
from quantumstring.particle import Particles, Particle
from quantumstring.process import PostProcess

import numpy as np
from matplotlib import pyplot as plt
from scipy import fftpack as fft

mypath = os.path.dirname(os.path.abspath(__file__))
mypath = "C:\\Users\\leog\\Desktop\\lg2021stage\\output"

duration = 1.0 # [s]
space_steps = 511
length = 6.0 # [m]
tension = 10.0 # [N]
density = 0.005 # [kg/m]

c = np.sqrt(tension/density)
dx = length/space_steps
dt = dx/c
time_steps = int(duration/dt)
duration = dt*time_steps

pmass = 0.0001 # [kg]
pstiff = 1000.0 # [N/m]
p = [
        Particle(int(space_steps*0.5), 0.0, pmass, pstiff, True, space_steps)
]
ps = Particles(*p, space_steps=space_steps)

examp = 0.05 # [m]
exfreq = 50.0 # [Hz]
expuls = 2*np.pi*exfreq # [rad/s]
left = ExcitatorSinAbsorber(dt, examp, expuls)
right = AbsorberEdge()

ic0 = [0.0]*space_steps
ic1 = [0.0]*space_steps

simu = Simulation(dt, time_steps, space_steps, length, density, tension, left, right, ic0, ic1, ps)

fpath, ppath, epath = simu.run(mypath)

process = PostProcess(
    open(fpath, "r"),
    open(ppath, "r"),
    open(epath, "r"),
    log=True
)

space_freq = 2*np.pi*exfreq/c
alpha = 2*space_freq*tension/(pstiff - pmass*expuls**2)
mod_t_sqr_theory = 1/(1 + 1/alpha**2)

fourier_paths = process.fourier((0.52, 0.98), frameskip=5, path=mypath, spectrograph=False)

fourier_path = fourier_paths[0]
mat = PostProcess.file2matrix(open(fourier_path, "r"), type=np.complex)
t_fft, f_fft = mat.shape
fft_freq = fft.fftfreq(f_fft, d=dx)
fft_time = np.linspace(0.0, t_fft*dt, t_fft)
amp = np.abs(mat)
phase = np.arctan(mat.imag/mat.real)
idxfreq = np.argmin(np.abs(fft_freq - space_freq))
amp_at_freq = amp[:, idxfreq]
amp_at_freq_perm = amp_at_freq[int(t_fft*0.8):t_fft]
mod_t_sim = max(amp_at_freq_perm)/examp
mod_t_sqr_sim = mod_t_sim**2

plt.plot(fft_time, amp_at_freq)

print("|t|²\n    ={} (theory)\n    ={} (simulation)".format(mod_t_sqr_theory, mod_t_sqr_sim))