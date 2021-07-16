from datetime import time
import os

from quantumstring.simulation import CenterFixed, FreeString, RingString, Simulation, Cavity
from quantumstring.edge import ExcitatorSinAbsorber, MirrorEdge, ExcitatorSin, AbsorberEdge
from quantumstring.particle import Particles, Particle
from quantumstring.process import PostProcess

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

mypath = os.path.dirname(os.path.abspath(__file__))
mypath = "C:\\Users\\leog\\Desktop\\lg2021stage\\output"

duration = 2.0 # [s]
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
for c in mcolors.TABLEAU_COLORS.items():
    colors.append(c[1])

stiffnesses = [1000.0] # [kg/s²]
colors = colors[0:len(stiffnesses)]
data_simus = list()

pmass = 0.1 # [kg]
for k, c in zip(stiffnesses, colors):
    data_simus.append(
        dict(
            k=k,
            p=Particle(int(space_steps*0.5), 0.0, pmass, k, True, space_steps, color=c)
        )
    )

examp = 0.05 # [m]
expuls = 2*np.pi*50 # [rad/s]
left = ExcitatorSinAbsorber(dt, examp, expuls)
right = AbsorberEdge()

ic0 = [0.0]*space_steps
ic1 = [0.0]*space_steps

ax = plt.axes()

for data in data_simus:
    k = data["k"]
    ps = Particles(data["p"])
    simu = Simulation(dt, time_steps, space_steps, length, density, tension, left, right, ic0, ic1, ps)
    fpath, ppath, epath = simu.run(mypath)
    process = PostProcess(
        open(fpath, "r"),
        open(ppath, "r"),
        open(epath, "r")
    )
    process.plot_particles(ax=ax, show=False, label="{}kg/s²".format(k))

plt.show()