from particle import Particle, Particles
from simulation import CenterFixed, FreeString, Simulation
from edge import MirrorEdge, ExcitatorSin, AbsorberEdge
from process import PostProcess

import numpy as np

import os

mypath = "C:\\Users\\leog\\Desktop\\lg2021stage\\output"

duration = 0.5 # [s]
space_steps = 512
length = 1.0 # [m]
tension = 1.0 # [N]
density = 0.005 # [kg/m]

c = np.sqrt(tension/density)
dx = length/space_steps
dt = dx/c
time_steps = int(duration/dt)
duration = dt*time_steps

p = Particles(
    space_steps=space_steps
)

left = ExcitatorSin(dt, 0.1, 20*np.pi, 0.0)
right = AbsorberEdge()

simu = CenterFixed(dt, time_steps, space_steps, length, density, tension, left, right, 0.002, 0.0)
print(simu)

fpath = os.path.join(mypath, "QuantumString-field_1625828881.txt")
ppath = os.path.join(mypath, "QuantumString-particles_1625828881.txt")

fpath, ppath = simu.run(mypath)

process = PostProcess(
    open(fpath, "r"),
    open(ppath, "r")
)

process.anim(mypath)
process.plot2d()
process.plot3d()