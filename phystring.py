import numpy as np

from field import OneSpaceField 

class PhyString:
    """
        Class for the simulation of the string
    """
    def __init__(self, length: float, nb_linear_steps: float, dt: float, celerity: float, linear_density: float, exitation_fx, ic_pos: list, ic_vel: list, particles):
        """
            Initialisation of the string

            :param length: length of the string [m]
            :param nb_linear_steps: number of cells in the string
            :param dt: value of the time step [s]
            :param celerity: celerity of the string [m/s]
            :param linear_density: linear density of the string [kg/m]
            :param excitation_fx: function corresponding to the condition at the left side of the string
            :param ic_pos: initial condition of the position of the string
            :param ic_vel: initial condition of the velocity of the string
            :param particles: Particles object 

            :type length: float
            :type nb_linear_steps: int
            :type dt: float
            :type celerity: float
            :type linear_density: float
            :type excitation_fx: function
            :type ic_pos: list
            :type ic_vel: list
            :type particles: Particles object 
        """
        self.dx = length/nb_linear_steps
        self.nb_linear_steps = nb_linear_steps
        self.dt = dt
        self.length = length # [m]
        self.celerity = celerity # [m/s]
        self.linear_density = linear_density # [kg/m]
        self.exitation = exitation_fx
        self.particles = particles

        self.cst_npt = (celerity*dt/self.dx)**2
        self.cst_pt = 0.25*linear_density/self.dx*(celerity*dt)**2

        if len(ic_pos) != nb_linear_steps or len(ic_vel) != nb_linear_steps:
            raise ValueError()
        
        ### PREPARE FIELD ###
        beg_pos = exitation_fx(0)
        beg_poss = exitation_fx(1)
        beg_vel = (beg_poss - beg_pos)/self.dt
        ic_pos[0] = beg_pos
        ic_pos1 = np.copy(ic_pos)
        ic_pos1[0] = beg_poss
        ic_vel[0] = beg_vel
        ic_pos = np.array(ic_pos)
        ic_vel = np.array(ic_vel)
        self.null_string = [0.0]*nb_linear_steps
        self.false_string = [False]*nb_linear_steps

        pt = particles.mass_presence()
        npt = np == False
        mass_density = particles.mass_density()
        self.borders = np.array(self.false_string)
        self.borders[0] = True
        self.borders[-1] = True
        bdl = [1, -1] # indexes of borders

        # firs we update as free dalembertian where no particles are
        ic_pos1 = np.array(self.null_string)
        ic_pos1[npt] = ic_pos[npt] + ic_vel[npt]*self.dt + 0.5*self.cst_npt*(PhyString.shift_list_left(ic_pos)[npt] + PhyString.shift_list_right(ic_pos)[npt] - 2*ic_pos[npt])
        ic_pos1[bdl] = ic_pos[bdl] # not computing borders!

        # and correctly update the string where the particles are
        ic_pos1[pt] = ic_pos[pt] + ic_vel[pt]*self.dt - self.cst_pt/mass_density[pt]*(PhyString.shift_list_left(ic_pos)[pt] - PhyString.shift_list_right(ic_pos)[pt])
        ic_pos1[bdl] = ic_pos[bdl] # not computing borders again!

        self.field = OneSpaceField(ic_pos, ic_pos1) 
    
    def __repr__(self):
        return "[STRING]    L={}m, c={}m/s, ρ={}kg/m ; with {} particles".format(
            self.length,
            self.celerity,
            self.linear_density,
            self.particles.particles_quantity
        )
        
    def update(self):
        """
            Updates the string for the next time step
        """
        pt = self.particles.mass_presence() # boolean vector where particle are
        npt = pt == False
        mass_density = self.particles.mass_density()

        newval = np.array(self.null_string)

        tstep = self.field.current_time_step()
        beg_val = self.exitation(tstep + 1)
        last_val = self.field.get_val_time(tstep) # field at t
        llast_val = self.field.get_val_time(tstep - 1) # field at t - 1
        last_val_r = PhyString.shift_list_right(last_val) # field at t right shifted. means that at x position, will return value at x - 1
        last_val_l = PhyString.shift_list_left(last_val) # field at t left shifted. means that at x position, will return value at x + 1
        newval[npt] = 2*last_val[npt] - llast_val[npt] + self.cst_npt*(last_val_r[npt] + last_val_l[npt] - 2*last_val[npt])
        newval[0] = beg_val # do not compute the boundaries, but set the beggining of string
        newval[-1] = last_val[-1] # do not compute the boundaries
        newval[pt] = 2*last_val[pt] - llast_val[pt] - 2*self.cst_pt/mass_density[pt]*(last_val_l[pt] - last_val_r[pt])

        self.field.update(newval) # update field
        self.particles.update() # update particles

    
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