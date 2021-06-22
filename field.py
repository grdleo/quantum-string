import numpy as np
from numpy import ndarray
import scipy

class OneSpaceField:
    """
        A class to create a 1D field that changes in time

        The memory can be limited if necessary
    """
    def __init__(self, init_val: ndarray, memory=np.inf):
        """
            Initialise the field

            :param init_val: initial value for the field. Each line is a value of the field at a certain time. First line is for t=0, second line is for t=Δt, etc
            :param memory: number of fields that will be saved at the same time (if inf -> no limit of memory)

            :type init_val: NumPy 2D array
            :type memory: int
        """
        if memory < 3:
            raise ValueError("'memory' has to be greater than 3!")

        # val = list or float
        try:
            (x, y) = init_val.shape
            if x <= 0:
                raise ValueError("Initial value for the field is not of a correct shape!")
            self._last_tstep = x - 1
            self.val = np.copy(init_val)
        except:
            raise ValueError("Initial value for the field is probably not a NumPy matrix!")

        self.memory = memory
    
    def pos_steps(self) -> int:
        """
            Returns the number of position steps in the field

            :return: number of distance steps of the field (aka the number of cells in the field, or the number of cols in the matrix)
        """
        return self.val[0].shape[0]
    
    def current_time_step(self) -> int:
        """
            Returns the value of the last time step of the field (aka the number of rows minus one)
        """
        return self._last_tstep
    
    def update(self, newval: list):
        """
            Appends a new value of the field at the time t=t₁+Δt where t₁ is the current time step of the field

            :param newval: the next value for the field, has to have the same length as the previous entries
            :type newval: list
        """
        self._last_tstep += 1
        self.val = np.vstack((self.val, newval))
        if self._last_tstep > self.memory: # if the amount of fields saved is superior to the memory allowed
            self.val = np.delete(self.val, 0, 0)
    
    def get_val_time(self, t: int) -> ndarray:
        """
            Get the value of  the field at the step t×Δt

            :param t: a time step
            :type t: int

            :return: the 1D field at the time step considered
        """
        tstep = t if (self._last_tstep < self.memory) or (t < 0) else t - self._last_tstep + self.memory
        if tstep < 0 and t >= 0: # therefore user tried to access a field that does not exist anymore due to memory restriction
            raise ValueError("Cannot access the field to the cell at time step {} because of memory restriction".format(t))
        return self.val[tstep]
    
    def get_val_pos(self, n: int) -> ndarray:
        """
            Get the list of the values taken by the cell at step n×Δx for all time steps

            :param n: index of a 1D field cell
            :type n: int

            :return: list of all the values taken by the field for each time steps
        """
        return self.val[:,n]

    def get_last(self) -> ndarray:
        """
            Returns the field at the time t₁, where t₁ is the current time step of the field

            :return: field at the time t₁, where t₁ is the current time step of the field
        """
        return self.get_val_time(self._last_tstep)
    
    def get_prev(self) -> ndarray:
        """
            Returns the field at the time t₁-1, where t₁ is the current time step of the field

            :return: field at the time t₁-1, where t₁ is the current time step of the field
        """
        return self.get_val_time(self._last_tstep - 1)
    
    def space_fft(self, xwindow: tuple, t: int, dx: float) -> tuple:
        a, b = int(xwindow[0]), int(xwindow[1])
        field = self.get_val_time(t)
        signal = field[a:b]
        n = signal.shape[-1]
        fft = scipy.ftt.ftt(signal)
        fftfreq = scipy.fft.fttfreq(n, d=dx)
        return fft, fftfreq
    
    def time_fft(self, twindow: tuple, x: int, dt: float) -> tuple:
        a, b = int(twindow[0]), int(twindow[1])
        field = self.get_val_pos(x)
        signal = field[a:b]
        n = signal.shape[-1]
        fft = scipy.ftt.ftt(signal)
        fftfreq = scipy.fft.fttfreq(n, d=dt)
        return fft, fftfreq

    @staticmethod
    def retrieve_sines(self, fft: list, fftfreq: list, h=0.0001):
        n = fftfreq.shape[-1]
        trunc_n = 0.5*n
        trunc_n += 1 if n % 2 != 0 else 0
        pos_fft = fft[trunc_n:n]
        pos_fftfreq = fftfreq[trunc_n:n]
        pos_spectrum = 2*np.abs(fft/n)[trunc_n:n]
        idx_peaks, _ = scipy.signal.find_peaks(pos_spectrum, height=h)
        freq_peaks = pos_fftfreq[idx_peaks]
        amp_peaks = pos_spectrum[idx_peaks]
        phase_peaks = np.angle(pos_fft)[idx_peaks]
        return {
            "frequency": freq_peaks,
            "amplitude": amp_peaks,
            "phase": phase_peaks
        }