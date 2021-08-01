from datetime import time
import os
import random

from quantumstring.simulation import CenterFixed, FreeString, RingString, Simulation, Cavity
from quantumstring.edge import ExcitatorSinAbsorber, MirrorEdge, ExcitatorSin, AbsorberEdge
from quantumstring.particle import Particles, Particle
from quantumstring.process import PostProcess

from scipy.signal import find_peaks
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

mypath = os.path.dirname(os.path.abspath(__file__))
mypath = "C:\\Users\\grdle\\Documents\\UGA\\M1\\Stage\\output"

duration = 5.0 # [s]
space_steps = 511
length = 1.0 # [m]
tension = 1.0 # [N]
density = 0.005 # [kg/m]

c = np.sqrt(tension/density)
dx = length/space_steps
dt = dx/c
time_steps = int(duration/dt)
duration = dt*time_steps

colors = list()
for c in mcolors.CSS4_COLORS.items():
    colors.append(c[1])

random.shuffle(colors)

stiffnesses = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0] # [kg/s²]
masses = [0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001, 0.0005] # [kg]
colors = colors[0:len(stiffnesses)]
data_simus = list()

pmass = 0.005 # [kg]
pstiff = 100.0
for k in stiffnesses:
    data_simus.append(
        dict(
            k=k,
            p=Particle(int(space_steps*0.5), 0.0, pmass, k, True, space_steps)
        )
    )

examp = 0.05 # [m]
expuls = 2*np.pi*25 # [rad/s]
left = ExcitatorSinAbsorber(dt, examp, expuls)
right = AbsorberEdge()

ic0 = [0.0]*space_steps
ic1 = [0.0]*space_steps

for data in data_simus:
    fig, (ax, ax_tau) = plt.subplots(1, 2)

    k = data["k"]
    ps = Particles(data["p"])
    simu = Simulation(dt, time_steps, space_steps, length, density, tension, left, right, ic0, ic1, ps)
    fpath, ppath, epath = simu.run(mypath)
    process = PostProcess(
        open(fpath, "r"),
        open(ppath, "r"),
        open(epath, "r")
    )
    z = process._particles_pos()
    z = z.reshape(z.size)
    peaks, _ = find_peaks(z)
    zpeaks = np.abs(z[peaks])
    zpeaks -= zpeaks[-1]
    tpeaks = process.tline[peaks]
    nb = 15
    polyfited = np.polyfit(tpeaks[0:nb], zpeaks[0:nb], 1)
    invtau = -polyfited[0]/polyfited[1]
    tau = 1/invtau
    ax.plot(process.tline, z)
    print("k={}N/m   τ={}s".format(k, tau))
    plt.show()


plt.show()