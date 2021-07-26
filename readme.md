# Quantum String

This Python module gives you the tools to create a simulation of a d'Alembertian string, in which you can put any bead of mass *m* attached to the system with a spring of stiffness *k*.

This module has been developped by Léo Giroud during the summer 2021 during his internship in Néel laboratory (Grenoble, France), for NOF team.

## Dependencies
* `numpy`
* `scipy`
* `matplotlib`
* `ffmpeg-python`
* `opencv-python`

## Quick start

```python
    import numpy as np

    import quantumstring as qs

    ### PATH
    mypath = "/choose/a/path"

    duration = 1.0 # duration of simulation [s]
    length = 1.0 # [m]
    tension = 10.0 # [N]
    density = 0.001 # [kg/m]
    space_steps = 1024 # string discretisation
    
    celerity = np.sqrt(tension/density)
    
    dx = length/space_steps
    dt = dx/celerity
    time_steps = int(duration/dt)

    ### INITIAL FIELD
    ic0 = [0.0]*space_steps # field at rest
    ic1 = [0.0]*space_steps

    ### PARTICLES
    particles_list = [
        qs.Particle(int(0.5*space_steps), 0.0, 0.01, 100.0, True, space_steps)
    ]
    particles = qs.Particles(*particles_list, space_steps=space_steps)

    ### EDGES
    left_edge = qs.ExcitatorSinAbsorber(dt, 0.05, 2*np.pi*50.0)
    right_edge = qs.AbsorberEdge()
    
    ### SIMULATION 
    sim = qs.Simulation(dt, time_steps, space_steps, length, density, tension, left_edge, right_edge, ic0, ic1, particles)

    field, parts, energy = sim.run(mypath)
    
    ### POST PROCESS
    process = qs.PostProcess(
        open(field, "r"),
        open(parts, "r"),
        open(energy, "r")
    )

    process.anim(mypath, frameskip=5)
    process.plot2d()
    process.plot_particles()
    process.phasegraph_particles()
```