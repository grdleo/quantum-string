from __future__ import annotations

import numpy as np

from field import OneSpaceField 
from edge import Edge, LoopEdge, AbsorberEdge
from particle import Particles

"""
    Class for dealing with the actual string
"""

class PhyString:
    """
        Class for the simulation of the string
    """
    def __init__(self, length: float, space_steps: int, dt: float, linear_density: float, tension: float, edge_left: Edge, edge_right: Edge, ic_pos: list[float], ic_vel: list[float], particles: Particles, memory_field=5):
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

        if len(ic_pos) != space_steps or len(ic_vel) != space_steps:
            raise ValueError("Initial conditions shapes for position and velocity not matching! ")
        
        rho = self.linear_density + self.particles.mass_density()/self.dx
        kappa = self.particles.spring_density()

        ic_pos = np.array(ic_pos)
        ic_vel = np.array(ic_vel)

        ic_pos = self.apply_edge(ic_pos, ic_pos, 0)
        utm = ic_pos - self.dt*ic_vel # for the initialisation, corresponds to the previous field
        uxp = PhyString.shift_list_left(ic_pos)
        uxm = PhyString.shift_list_right(ic_pos)
        ic_pos1 = self.field_evo(ic_pos, utm, uxp, uxm, rho, kappa)
        ic_pos1 = self.apply_edge(ic_pos1, utm, 1)

        init_val = np.vstack((ic_pos, ic_pos1))
        self.field = OneSpaceField(init_val, memory=memory_field)
    
    def __repr__(self):
        return "[STRING]    L={0:.3f}m, T={1:.3f}N, rho={2:.3f}kg/m, c={3:.3f}m/s ; >{4}   {5}< ; {6} particles".format(
            self.length,
            self.tension,
            self.linear_density,
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
        beta = m/(self.linear_density*self.dx)
        gamma = k*self.dx/self.tension
        ###

        tstep = self.field.current_time_step()
        last_val = self.field.get_last() # field at t
        llast_val = self.field.get_prev() # field at t - 1
        last_val_m = PhyString.shift_list_right(last_val) # field at t right shifted. means that at x position, will return value at x - 1
        last_val_p = PhyString.shift_list_left(last_val) # field at t left shifted. means that at x position, will return value at x + 1

        newval = self.field_evo(last_val, llast_val, last_val_p, last_val_m, beta, gamma) # evolution of the string according to the equations
        newval = self.apply_edge(newval, last_val, tstep + 1) # apply the conditions at both of the edges 

        self.field.update(newval) # update field
        self.particles.update() # update particles
    
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
        invb = 1/(1 + beta)
        dbg = beta - gamma
        return invb*(uxp + uxm + dbg*u) - utm
    
    def linear_energy(self, u: list, utm: list, uxp: list, uxm: list, rho: list, kappa: list) -> list:
        return 0.5*(rho*self.invdt2*(u - utm)**2 + 0.25*self.tension*self.invdx2*(uxp - uxm)**2 + kappa*u*u)
    
    def apply_edge(self, f: list[float], ftm: list[float], t: int) -> list[float]:
        """
            Apply the edge conditions to the string

            :param f: the field to be conditioned
            :param t: time step
        """
        u = np.copy(f)

        type_left = type(self.edge_left)
        type_right = type(self.edge_right)

        if type_left != LoopEdge:
            u[0] = ftm[1] if type_left == AbsorberEdge else self.edge_left.condition(t)
        
        if type_right != LoopEdge:
            u[-1] = ftm[-2] if type_right == AbsorberEdge else self.edge_right.condition(t)

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