from __future__ import annotations

import random

import numpy as np

from field import OneSpaceField

"""
    Classes for dealing with particles on the string
"""

class Particle:
    """
        A class to create a particle to be on the string
    """
    STR_MASS = "m"
    STR_PULSATION = "omega"
    STR_COLOR = "color"

    mass: float
    """ Mass *m* of the particle [kg] """
    pulsation: float
    """ Pulsation *ω* of the particle [rad/s] (relation with the spring stifness *K=mω²*) """
    space_steps: int
    """ Number of cells (discretisation) that composes the string """
    fixed: bool
    """ if True, the particle is fixed on the string """

    def __init__(self, pos: int, vel: float, mass: float, pulsation: float, fixed: bool, space_steps: int, color=(0, 0, 255)):
        """
            Initalises a particle

            :param pos: index of the cell to be for the particle
            :param vel: initial (vertical) velocity of particle [m/s]
            :param mass: mass of the particle [kg]
            :param pulsation: pulsation of the particle [rad/s]
            :param fixed: is the particle fixed horizontally on the bench?
            :param space_steps: number of cells in the string
            :type pos: float
            :type vel: float
            :type mass: float
            :type pulsation: float
            :type fixed: bool
            :type space_steps: int
        """
        self.mass = mass
        self.pulsation = pulsation
        self.space_steps = space_steps
        self.fixed = fixed
        self._firstpos = pos
        pos_next = pos
        self.color = color

        if not 0 <= pos < space_steps:
            raise ValueError("Cell position of particle {} is not on the string [0, {}]".format(pos, space_steps - 1))

        if not fixed:
            raise NotImplementedError("Moving particles not implemented yet! :(")
        
        init_val =np.vstack((pos, pos_next))
        self.pos = OneSpaceField(init_val, memory=5)
    
    def __repr__(self):
        return "m={:.2f}kg, ω={:.2f}rad/s;".format(self.mass, self.pulsation)
    
    def infos(self) -> dict:
        """
            Returns a dictionary containing the informations about the mass and pulsation of the particle
        """
        return {
            Particle.STR_MASS: self.mass,
            Particle.STR_PULSATION: self.pulsation,
            Particle.STR_COLOR: self.color
        }
    
    def update(self):
        """
            Updates the particle
        """
        if self.fixed:
            self.pos.update(self._firstpos)
        else:
            raise NotImplementedError()

class Particles:
    """
        A wrapper of a list of particles to be used for the string simulation
    """

    particles: list[Particle]
    """ Contains the Particle objects """
    space_steps: int
    """ Number of cells (discretisation) that composes the string """
    free_particles: bool
    """ if True, at least one Particle is not fixed on the string """
    empty: bool
    """ if True, there is no Particle inside this object """

    def __init__(self, *particles: list[Particle], space_steps=0):
        """
            Initialise the list of particles

            :param space_steps: number of cells in the string
            :param particles: all the particles considered in a list
        """
        self.empty = True
        self.space_steps = space_steps
        self.particles_quantity = len(particles)
        self.free_particles = False # if True, at least one particle is moving 
        self.particles = []
        if self.particles_quantity != 0:
            self.space_steps = particles[0].space_steps
            self.empty = False
            self.particles = particles
            for p in particles:
                if not p.fixed:
                    self.free_particles = True
                if p.space_steps != self.space_steps:
                    raise ValueError("Some particles are on different strings!")
        else:
            if space_steps == 0:
                raise ValueError("No particles in the object: therefore 'space_steps' has to be entered manually!")
    
    def infos(self) -> list:
        """
            Returns list containing the dictionaries of the informations about the mass and pulsation of the particles
        """
        a = []
        for p in self.particles:
            a.append(p.infos())
        return a
    
    def __repr__(self):
        s = "[PARTICLES]    ;"
        for (p, i) in zip(self.particles, range(0, self.particles_quantity)):
            s += "{}: {}".format(i, str(p))
        return s
    
    def update(self):
        """
            Updates all the particles
        """
        if not self.empty:
            for p in self.particles:
                p.update()
    
    def list_pos(self, tstep=-1) -> list[int]:
        """
            Return a list where each entry corresponds to the index of the cell where a particle is, at the time step considered

            :param tstep: the time step considered
        """
        s = []
        if not self.empty:
            for p in self.particles:
                s.append(int(p.pos.get_val_time(tstep)))
        return np.array(s)
    
    def list_free(self, tstep=-1) -> list[int]:
        """
            Return a list where each entry corresponds to the index of the cell where a particle is NOT, at the time step considered (complementary of list_pos)

            :param tstep: the time step considered
            :type tstep: int

            :return: list containing the position of all free cells
            :rtype: list
        """
        s = [i for i in range(0, self.space_steps)]
        lp = self.list_pos(tstep=tstep)
        if not self.empty:
            for i in lp:
                s = np.delete(s, i)
        return np.array(s)
    
    def mass_density(self, tstep=-1, fixed=False) -> list[float]:
        """
            Return a list where each cell corresponds to a cell of the string. if cell == 0, no particle on this step. if cell >= 1, corresponds to the total mass at that cell
            
            :param tstep: the time step considered
            :param fixed: if True, will return only the masses that are fixed on the string
        """
        s = [0]*self.space_steps
        if not self.empty:
            for p in self.particles:
                pos = int(p.pos.get_val_time(tstep)) # get the position of each particle
                s[pos] += 0 if fixed and not p.fixed else p.mass # increment the vector where the particle is
        return np.array(s)

    def spring_density(self, tstep=-1, fixed=False) -> list[float]:
        """
            Return a list where each cell corresponds to a cell of the string. if cell == 0, no particle on this step. if cell > 0, the value corresponds to the spring stiffness attached to this cell
            
            :param tstep: the time step considered
            :param fixed: if True, will return only the masses that are fixed on the string
        """
        s = [0]*self.space_steps
        if not self.empty:
            for p in self.particles:
                pos = int(p.pos.get_val_time(tstep)) # get the position of each particle
                s[pos] += 0 if fixed and not p.fixed else p.mass*p.pulsation*p.pulsation # increment the vector where the particle is
        return np.array(s)

    def mass_presence(self, tstep=-1, fixed=False) -> list[bool]:
        """
            Return a list where each cell corresponds to a cell of the string. if cell == False, no particle on this step. if cell == True, at least 1 particle is present on this cell
            
            :param tstep: the time step considered
            :param fixed: if True, will return only the masses that are fixed on the string
        """
        s = np.array([False]*self.space_steps)
        md = self.mass_density(tstep=tstep, fixed=fixed)
        s[md != 0] = True
        return np.array(s)

