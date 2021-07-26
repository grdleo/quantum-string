from datetime import time
import os
from quantumstring.field import OneSpaceField

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

examp = 0.05 # [m]
exfreq = 50.0 # [Hz]
expuls = 2*np.pi*exfreq # [rad/s]
exk = expuls/c
exwavelen = 2*np.pi/exk
excitator = lambda x, t: examp*np.sin(exk*x - expuls*t)
left = ExcitatorSinAbsorber(dt, examp, expuls)
right = AbsorberEdge()

pmass = 0.001 # [kg]
stiffnesses = np.array([1000.0, 900.0, 800.0, 700.0, 600.0, 500.0, 400.0, 300.0, 200.0, 100.0])
transmissions = []
alpha = 2*expuls*density*c/(stiffnesses - pmass*expuls**2)
transmissions_theory = 1/(1 + 1/alpha**2)

for stiff in stiffnesses:
    p = [
            Particle(int(space_steps*0.5), 0.0, pmass, stiff, True, space_steps)
    ]
    ps = Particles(*p, space_steps=space_steps)

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

    ### PROCESSING TRANSMISSION ###

    ### PHASE
    last_ts = simu.s.field.current_time_step()
    last_field = simu.s.field.get_val_time(last_ts)
    last_time = last_ts*dt
    xlin = np.linspace(0.0, length, space_steps)
    field_from_excitator = excitator(xlin, last_time)
    wavelen_steps_approx = int(1.25*exwavelen/dx)

    win_end_a, win_end_b = space_steps - 2*wavelen_steps_approx, space_steps - wavelen_steps_approx
    field_near_end_sim = last_field[win_end_a:win_end_b]
    idx_node_sim = np.argmax(field_near_end_sim) + win_end_a

    # -->
    ex_field_right = field_from_excitator[idx_node_sim:idx_node_sim + wavelen_steps_approx]
    idx_node_ex_right = np.argmax(ex_field_right)

    # <--
    ex_field_left = field_from_excitator[idx_node_sim - wavelen_steps_approx:idx_node_sim + 1]
    idx_node_ex_left = np.argmax(np.flip(ex_field_left))

    # get the space delay between the two sines
    space_delay = min(idx_node_ex_right, idx_node_ex_left)

    if idx_node_ex_right > idx_node_ex_left: # then the transmitted wave is LATE compared to excitator
        space_delay *= -1

    phase_t = 2*space_delay*np.pi*dx/exwavelen
    phase_t %= np.pi

    ### MODULE
    after = last_field[int(space_steps*0.55):int(space_steps*0.95)]
    max_after = max(after)
    mod_t = max_after/examp
    transmissions.append(mod_t**2)

    ### THEORY ###

    alpha = 2*exk*tension/(stiff - pmass*expuls**2)
    t_heory = 1/np.complex(1, -1/alpha)

    print("|t|²\n    ={} (theory)\n    ={} (simulation)".format(np.abs(t_heory)**2, mod_t**2))
    print("arg(t)\n    ={} (theory)\n    ={} (simulation)".format(np.angle(t_heory), phase_t))

"""
plt.plot(stiffnesses, transmissions_theory, "r.", label="theory")
plt.plot(stiffnesses, transmissions, "b+", label="simulation")
plt.xlabel("spring stiffness [N/m]")
plt.ylabel("|t|²")
plt.show()
"""