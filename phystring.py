import numpy as np

from field import OneSpaceField 

class PhyString:
    """
        Class for the simulation of the string
    """
    def __init__(self, length: float, nb_linear_steps: float, dt: float, linear_density: float, tension: float, exitation_fx, ic_pos: list, ic_vel: list, particles):
        """
            Initialisation of the string

            :param length: length of the string [m]
            :param nb_linear_steps: number of cells in the string
            :param dt: value of the time step [s]
            :param linear_density: linear density of the string [kg/m]
            :param tension: tension in the string [kg.m/s²]
            :param excitation_fx: function corresponding to the condition at the left side of the string
            :param ic_pos: initial condition of the position of the string
            :param ic_vel: initial condition of the velocity of the string
            :param particles: Particles object 

            :type length: float
            :type nb_linear_steps: int
            :type dt: float
            :type celerity: float
            :type linear_density: float
            :type tension: float
            :type excitation_fx: function
            :type ic_pos: list
            :type ic_vel: list
            :type particles: Particles object 
        """
        self.dx = length/nb_linear_steps
        self.nb_linear_steps = nb_linear_steps
        self.dt = dt
        self.length = length # [m]
        self.discret_vel = self.dx/self.dt # [m/s]
        self.v2 = self.discret_vel**2
        self.linear_density = linear_density # [kg/m]
        self.tension = tension # [kg.m/s²]
        self.exitation = exitation_fx
        self.particles = particles

        self.celerity = np.sqrt(tension/linear_density)

        if len(ic_pos) != nb_linear_steps or len(ic_vel) != nb_linear_steps:
            raise ValueError()
        
        rho = self.linear_density + self.particles.mass_density()/self.dx
        kappa = self.particles.spring_density()

        ic_pos = np.array(ic_pos)
        ic_vel = np.array(ic_vel)

        ic_pos = self.apply_edge(ic_pos, 0)
        utm = ic_pos - self.dt*ic_vel # for the initialisation, corresponds to the previous field
        uxp = PhyString.shift_list_left(ic_pos)
        uxm = PhyString.shift_list_right(ic_pos)
        ic_pos1 = self.field_evo(ic_pos, utm, uxp, uxm, rho, kappa)
        ic_pos1 = self.apply_edge(ic_pos1, 1)

        init_val = np.vstack((ic_pos, ic_pos1))
        self.field = OneSpaceField(init_val, memory=5)
    
    def __repr__(self):
        return "[STRING]    L={}m, T={}N, ρ={}kg/m ; {} particles".format(
            self.length,
            self.tension,
            self.linear_density,
            self.particles.particles_quantity
        )
        
    def update(self):
        """
            Updates the string for the next time step
        """
        pt = self.particles.mass_presence() # boolean vector where particle are

        ### IF THE PARTICLES ARE ALL FIXED, NO NEED TO RECOMPUTE THIS EVERY FRAME !!!
        rho = self.linear_density + self.particles.mass_density()/self.dx
        kappa = self.particles.spring_density()
        ###

        tstep = self.field.current_time_step()
        last_val = self.field.get_last() # field at t
        llast_val = self.field.get_prev() # field at t - 1
        last_val_m = PhyString.shift_list_right(last_val) # field at t right shifted. means that at x position, will return value at x - 1
        last_val_p = PhyString.shift_list_left(last_val) # field at t left shifted. means that at x position, will return value at x + 1

        newval = self.field_evo(last_val, llast_val, last_val_p, last_val_m, rho, kappa) # evolution of the string according to the equations
        newval = self.apply_edge(newval, tstep + 1) # apply the conditions at both of the edges 

        self.field.update(newval) # update field
        self.particles.update() # update particles
    
    def field_evo(self, u: list, utm: list, uxp: list, uxm: list, rho: list, kappa: list) -> list:
        """
            Given the field, returns the evolution in time with the effective ρ and k

            :param u: field at current time
            :param utm: field at previous time
            :param uxp: field at current time shifted -Δx
            :param uxm: field at current time shifted +Δx
            :param rho: effective linear density 
            :param kappa: effective stiffness spring
        """
        inv_force = 1/(rho*self.v2)
        return 2*u*(1 - inv_force*(self.tension + 0.5*kappa*self.dx)) + inv_force*self.tension*(uxp + uxm) - utm
    
    def apply_edge(self, f: list, t: int) -> list:
        """
            Apply the edge conditions to the string
        """
        u = np.copy(f)
        u[0] = self.exitation(t)
        u[-1] = 0.0
        return u

    @staticmethod
    def shift_list_right(lst: list) -> list:
        # (corresponds to "-1" in equations)
        """
            Shifts a list to the right
            ex: [a, b, c, d] --> [d, a, b, c]

            :param lst: the list to be shifted
            :type lst: list

            :return: shifted list
            :rtype: list
        """
        l = lst.copy()
        l = np.insert(l, 0, l[-1])
        l = np.delete(l, -1)
        return l
    
    @staticmethod
    def shift_list_left(lst: list) -> list:
        # (corresponds to "+1" in equations)
        """
            Shifts a list to the left
            ex: [a, b, c, d] --> [b, c, d, a]

            :param lst: the list to be shifted
            :type lst: list

            :return: shifted list
            :rtype: list
        """
        l = lst.copy()
        l = np.insert(l, len(l), l[0])
        l = np.delete(l, 0)
        return l