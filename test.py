from particle import Particle, Particles
from simulation import Simulation
from edge import MirrorEdge, ExcitatorSin, AbsorberEdge
from process import PostProcess

import numpy as np

mypath = "C:\\Users\\leog\\Desktop\\lg2021stage\\output"

duration = 2.5 # [s]
space_steps = 511
length = 1.0 # [m]
tension = 1.0 # [N]
density = 0.005 # [kg/m]

c = np.sqrt(tension/density)
dx = length/space_steps
dt = dx/c
time_steps = int(duration/dt)
duration = dt*time_steps

k = np.pi/length # [rad/m]
omega = 10*2*np.pi # [rad/s]
stationary_field = lambda x, t: 0.025*np.sin(k*x)*np.cos(omega*t)

space_field = np.linspace(0.0, length, space_steps)

ic0 = stationary_field(space_field, 0.0)
ic1 = stationary_field(space_field, dt)

particles = Particles(
    # space_steps=space_steps
    Particle(int(space_steps*0.5), 0.0, 0.01, omega, True, space_steps)
)

left = MirrorEdge()
right = MirrorEdge()

simu = Simulation(dt, time_steps, space_steps, length, density, tension, left, right, ic0, ic1, particles)
fieldp, particlesp = simu.run(mypath)

post = PostProcess(
    open(fieldp, "r"), open(particlesp, "r")
)

post.anim(mypath, frameskip=10, yscale=3.0)