from __future__ import annotations

import numpy as np

from quantumstring.field import OneSpaceField 
from quantumstring.edge import Edge, LoopEdge, AbsorberEdge
from quantumstring.particle import Particles

"""
    Class for dealing with the actual string
"""

class PhyString:
    """
        Class for the simulation of the string
    """
    def __init__(self, length: float, space_steps: int, dt: float, linear_density: float, tension: float, edge_left: Edge, edge_right: Edge, ic0: list[float], ic1: list[float], particles: Particles, memory_field=5):
        """
            Initialisation of the string

            :param length: length of the string [m]
            :param space_steps: number of cells in the string
            :param dt: value of the time step [s]
            :param linear_density: linear density of the string [kg/m]
            :param tension: tension in the string [kg.m/s²]
            :param edge_left: the condition at the left extremity of the string
            :param edge_right: the condition at the right extremity of the string
            :param ic_pos: initial condition of the position of the string
            :param ic_vel: initial condition of the velocity of the string
            :param particles: Particles object 
            :param memory_field: the maximum simultaneous elements the field class can hold. 'np.inf' for no limitation
        """
        self.dx = length/float(space_steps)
        self.invdx2 = 1/self.dx**2
        self.space_steps = space_steps
        self.dt = dt
        self.invdt2 = 1/self.dt**2
        self.length = length # [m]
        self.discret_vel = float(self.dx/self.dt) # [m/s]
        self.invv2 = 1/(self.discret_vel*self.discret_vel)
        self.linear_density = linear_density # [kg/m]
        self.tension = tension # [kg.m/s²]
        self.edge_left = edge_left
        self.edge_right = edge_right
        self.particles = particles

        self.celerity = np.sqrt(tension/linear_density)

        if len(ic0) != space_steps or len(ic1) != space_steps:
            raise ValueError("Initial conditions shapes for initial positions not matching! ")

        ### MASKS COMPUTATION (for absorbers) ###
        self.mask_utm = [1.0]*self.space_steps
        self.mask_uxp = [1.0]*self.space_steps
        self.mask_uxm = [1.0]*self.space_steps

        if self.edge_left.absorber:
            self.mask_utm[0] = 0.0
            self.mask_uxm[0] = 0.0
        if self.edge_right.absorber:
            self.mask_utm[-1] = 0.0
            self.mask_uxp[-1] = 0.0
        
        self.mask_utm = np.array(self.mask_utm)
        self.mask_uxp = np.array(self.mask_uxp)
        self.mask_uxm = np.array(self.mask_uxm)
        ###

        ic1 = self.apply_edge(ic1, 1)
        init_val = np.vstack((ic0, ic1))
        self.field = OneSpaceField(init_val, memory=memory_field)

        self.energy = OneSpaceField(init_val*0.0, memory=5)

    def __repr__(self):
        return "[STRING]    L={0:.1f}m, T={1:.1f}N, ρ={2:.1f}g/m, c={3:.1f}m/s ; {4}|~~~~|{5} ; {6} particles".format(
            self.length,
            self.tension,
            self.linear_density*1e3,
            self.celerity,
            self.edge_left,
            self.edge_right,
            self.particles.particles_quantity
        )
        
    def update(self):
        """
            Updates the string for the next time step
        """

        ### IF THE PARTICLES ARE ALL FIXED, NO NEED TO RECOMPUTE THIS EVERY FRAME !!!
        m = self.particles.mass_density()
        k = self.particles.spring_density()
        rho = self.linear_density + m
        beta = m/(self.linear_density*self.dx)
        gamma = k*self.dx/self.tension
        ###

        tstep = self.field.current_time_step()
        last_val = self.field.get_last() # field at t
        llast_val = self.field.get_prev() # field at t - 1
        last_val_m = PhyString.shift_list_right(last_val) # field at t right shifted. means that at x position, will return value at x - 1
        last_val_p = PhyString.shift_list_left(last_val) # field at t left shifted. means that at x position, will return value at x + 1

        newval = self.field_evo(last_val, llast_val, last_val_p, last_val_m, beta, gamma) # evolution of the string according to the equations
        newval = self.apply_edge(newval, tstep + 1) # apply the conditions at both of the edges 

        energy = self.linear_energy(last_val, llast_val, last_val_p, last_val_m, rho, k)

        self.field.update(newval) # update field
        self.particles.update() # update particles
        self.energy.update(energy)
            
    def field_evo(self, u: list[float], utm: list[float], uxp: list[float], uxm: list[float], beta: list[float], gamma: list[float]) -> list[float]:
        """
            Given the field, returns the evolution in time

            :param u: field at current time
            :param utm: field at previous time
            :param uxp: field at current time shifted -Δx
            :param uxm: field at current time shifted +Δx
            :param beta: (see equation)
            :param gamma: (see equation)
        """
        dbg = 2.0*beta - gamma
        return (uxp*self.mask_uxp + uxm*self.mask_uxm + u*dbg)/(1.0 + beta) - utm*self.mask_utm

    def linear_energy(self, u: list[float], utm: list[float], uxp: list[float], uxm: list[float], rho: list[float], kappa: list[float]) -> list[float]:
        cin = rho*self.invdt2*(u - utm)**2
        ten = 0.25*self.tension*self.invdx2*(uxp - uxm)**2
        spr = kappa*u*u

        ten[0] = 0.0
        ten[-1] = 0.0

        le = 0.5*(cin + ten + spr)
        return le
    
    def apply_edge(self, f: list[float], t: int) -> list[float]:
        """
            Apply the edge conditions to the string

            :param f: the field to be conditioned
            :param t: time step
        """
        u = np.copy(f)

        try:
            u[0] = self.edge_left.condition(t)
        except TypeError:
            pass
            
        try:
            u[-1] = self.edge_right.condition(t)
        except TypeError: 
            pass

        return u

    @staticmethod
    def shift_list_right(lst: list) -> list: # (corresponds to "-1" in equations)
        """
            Shifts a list to the right

            >>> Simulation.shift_list_right([1, 2, 3, 4])
            [4, 1, 2, 3]

            :param lst: the list to be shifted
        """
        l = lst.copy()
        l = np.insert(l, 0, l[-1])
        l = np.delete(l, -1)
        return l
    
    @staticmethod
    def shift_list_left(lst: list) -> list: # (corresponds to "+1" in equations)
        """
            Shifts a list to the left

            >>> Simulation.shift_list_left([1, 2, 3, 4])
            [2, 3, 4, 1]

            :param lst: the list to be shifted
        """
        l = lst.copy()
        l = np.insert(l, len(l), l[0])
        l = np.delete(l, 0)
        return l