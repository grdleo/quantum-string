from datetime import time

from matplotlib.pyplot import yscale
from particle import Particle, Particles
from simulation import CenterFixed, FreeString, RingString, Simulation, Cavity
from edge import MirrorEdge, ExcitatorSin, AbsorberEdge
from process import PostProcess

import numpy as np

import os

mypath = os.path.dirname(os.path.abspath(__file__))
mypath = "C:\\Users\\leog\\Desktop\\lg2021stage\\output"

duration = 0.5 # [s]
space_steps = 511
length = 1.0 # [m]
tension = 1.0 # [N]
density = 0.005 # [kg/m]

c = np.sqrt(tension/density)
dx = length/space_steps
dt = dx/c
time_steps = int(duration/dt)
duration = dt*time_steps

xline = np.linspace(0.0, length, space_steps)

k = 3*np.pi/length
omega = c*k
mysin = lambda x, t: 0.05*np.sin(k*x)*np.cos(omega*t)

p = Particles(
    Particle(int(0.5*space_steps), 0.0, 0.001, omega, True, space_steps)
)

ic0 = mysin(xline, 0.0)
ic1 = mysin(xline, dt)

simu = Cavity(dt, time_steps, space_steps, length, density, tension, ic0, ic1, p)

"""
wavelen = 0.2
def mysin(x):
    ptn = x <= wavelen
    return 0.05*np.sin(2*np.pi/wavelen*x)*ptn.astype(float)

ic0 = mysin(xline)
ic1 = np.insert(ic0[0:-1], 0, 0)
simu = RingString(dt, time_steps, space_steps, length, density, tension, ic0, ic1, p)
"""


"""
left = ExcitatorSin(dt, 0.1, 60*np.pi, 0.0)
right = AbsorberEdge()

simu = CenterFixed(dt, time_steps, space_steps, length, density, tension, left, right, 0.002, 13*np.pi)
print(simu)
"""

fpath, ppath, epath = simu.run(mypath)

process = PostProcess(
    open(fpath, "r"),
    open(ppath, "r"),
    open(epath, "r")
)

process.anim(mypath)
# process.energy()
process.plot2d()
# process.plot3d()