import numpy as np

from field import OneSpaceField

class Particle:
    """
        A class to create a particle to be on the string
    """
    def __init__(self, pos: int, vel: float, mass: float, pulsation: float, fixed: bool, nb_linear_steps: int):
        """
            Initalises a particle

            :param pos: index of the cell to be for the particle
            :param vel: initial (vertical) velocity of particle [m/s]
            :param mass: mass of the particle [kg]
            :param pulsation: pulsation of the particle (frequency) [rad/s]
            :param fixed: is the particle fixed horizontally on the bench?
            :param nb_linear_steps: number of cells in the string
            :type pos: float
            :type vel: float
            :type mass: float
            :type pulsation: float
            :type fixed: bool
            :type nb_linear_steps: int
        """
        self.mass = mass
        self.pulsation = pulsation
        self.nb_linear_steps = nb_linear_steps
        self.fixed = fixed
        self._firstpos = pos
        pos_next = pos

        if not 0 <= pos < nb_linear_steps:
            raise ValueError("Wrong initial condition for particle!")

        if not fixed:
            raise NotImplementedError("Moving particles not implemented yet! :(")

        self.pos = OneSpaceField(pos, pos_next)
    
    def update(self):
        """
            Updates the particle
        """
        if not self.fixed:
            raise NotImplementedError()
        else:
            self.pos.update(self._firstpos)

class Particles:
    """
        A wrapper of a list of particles to be used for the string simulation
    """
    def __init__(self, nb_linear_steps: int, particles: list):
        """
            Initialise the list of particles

            :param nb_linear_steps: number of cells in the string
            :param particles: all the particles considered in a list
            :type nb_linear_steps: int
            :type particles: list of objects Particle
        """
        self.empty = True
        self.nb_linear_steps = nb_linear_steps
        self.particles_quantity = len(particles)
        if self.particles_quantity != 0:
            self.empty = False
            self.particles = particles
            for p in particles:
                if p.nb_linear_steps != self.nb_linear_steps:
                    raise ValueError("Some particles are on different strings!")
    
    def __repr__(self):
        s = "[PARTICLES]    "
        for (p, i) in zip(self.particles, range(0, self.particles_quantity)):
            s += "{}: m={}kg, ω={}rad/s ; ".format(i, p.mass, p.pulsation)
        return s
    
    def update(self):
        """
            Updates all the particles
        """
        if not self.empty:
            for p in self.particles:
                p.update()
    
    def list_pos(self, tstep=-1):
        """
            Return a list where each entry corresponds to the index of the cell where a particle is, at a step

            :param tstep: the time step considered
            :type tstep: int

            :return: list containing the position (cells) of all the particles at given time step
            :rtype: list
        """
        s = []
        if not self.empty:
            for p in self.particles:
                s.append(int(p.pos.get_val_time(tstep)))
        return np.array(s)
    
    def list_free(self, tstep=-1):
        """
            Return a list where each entry corresponds to the index of the cell where a particle is NOT, at a step (complementary of list_pos)

            :param tstep: the time step considered
            :type tstep: int

            :return: list containing the position of all free cells
            :rtype: list
        """
        s = [i for i in range(0, self.nb_linear_steps)]
        lp = self.list_pos(tstep=tstep)
        if not self.empty:
            for i in lp:
                s = np.delete(s, i)
        return np.array(s)
    
    def mass_density(self, tstep=-1):
        """
            Return a list where each cell corresponds to a cell of the string. if cell == 0, no particle on this step. if cell >= 1, corresponds to the total mass at that cell
            
            :param tstep: the time step considered
            :type tstep: int

            :return: list corresponding to the mass of the particles of the string in each cells
            :rtype: list 
        """
        s = [0]*self.nb_linear_steps
        if not self.empty:
            for p in self.particles:
                pos = int(p.pos.get_val_time(tstep)) # get the position of each particle
                s[pos] += p.mass # increment the vector where the particle is
        return np.array(s)
    
    def mass_presence(self, tstep=-1):
        """
            Return a list where each cell corresponds to a cell of the string. if cell == False, no particle on this step. if cell == True, at least 1 particle is present on this cell
            
            :param tstep: the time step considered
            :type tstep: int

            :return: list corresponding to the presence of the particle
            :rtype: list 
        """
        s = [False]*self.nb_linear_steps
        if not self.empty:
            for p in self.particles:
                pos = int(p.pos.get_val_time(tstep)) # get the position of each particle
                s[pos] = True # increment the vector where the particle is
        return np.array(s)

