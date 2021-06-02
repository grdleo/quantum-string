import numpy as np

class OneSpaceField:
    """
        A class to create a 1D field that changes in time
    """
    def __init__(self, val_t0: list, val_t1: list):
        """
            Initialise the field

            :param val_t0: the value of the field at t=0×Δt
            :param val_t1: the value of the field at t=1×Δt
            :type val_t0: list
            :type val_t1: list
        """
        # val = list or float
        self.val = np.vstack((val_t0, val_t1))
    
    def time_steps(self):
        """
            Returns the number of time steps in the field

            :return: number of time steps for this field (aka the number of 1D fields in this class, or the number of rows in the matrix)
        """
        return self.val.shape[0]
    
    def pos_steps(self):
        """
            Returns the number of position steps in the field

            :return: number of distance steps of the field (aka the number of cells in the field, or the number of cols in the matrix)
        """
        return self.val[0].shape[0]
    
    def current_time_step(self):
        """
            Returns the value of the last time step of the field (aka the number of rows minus one)
        """
        return self.val.shape[0] - 1
    
    def update(self, newval: list):
        """
            Appends a new value of the field at the time t=t₁+Δt where t₁ is the current time step of the field

            :param newval: the next value for the field, has to have the same length as the previous entries
            :type newval: list
        """
        self.val = np.vstack((self.val, newval))
    
    def get_val_time(self, t: int):
        """
            Get the value of  the field at the step t×Δt

            :param t: a time step
            :type t: int

            :return: the 1D field at the time step considered
        """
        return self.val[t]
    
    def get_val_pos(self, n: int):
        """
            Get the list of the values taken by the cell at step n×Δx for all time steps

            :param n: index of a 1D field cell
            :type n: int

            :return: list of all the values taken by the field for each time steps
        """
        return self.val[:,n]

    def get_last(self):
        """
            Returns the field at the time t₁, where t₁ is the current time step of the field

            :return: field at the time t₁, where t₁ is the current time step of the field
        """
        return self.get_val_time(self.current_time_step())