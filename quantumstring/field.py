from __future__ import annotations

import numpy as np
from numpy import ndarray
import scipy
import scipy.signal

"""
    Classes for dealing with a 1D field in time
"""

class OneSpaceField:
    """
        A class to create a 1D field that changes in time

        The memory can be limited if necessary
    """

    val: ndarray
    """ Matrix representing the field. The n-th line corresponds to the field at time step n. """

    def __init__(self, init_val: ndarray, memory=np.inf):
        """
            Initialise the field

            :param init_val: initial value for the field. Each line is a value of the field at a certain time. First line is for t=0, second line is for t=Δt, etc
            :param memory: number of fields that will be saved at the same time (if inf -> no limit of memory)
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
            Returns the number of position steps in the field (aka the number of cells in the field, or the number of cols in the matrix)
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
        """
        self._last_tstep += 1
        self.val = np.vstack((self.val, newval))
        if self._last_tstep > self.memory: # if the amount of fields saved is superior to the memory allowed
            self.val = np.delete(self.val, 0, 0)
    
    def get_val_time(self, t: int) -> ndarray:
        """
            Get the value of  the field at the step t×Δt
        """
        tstep = t if (self._last_tstep < self.memory) or (t < 0) else t - self._last_tstep + self.memory
        if tstep < 0 and t >= 0: # therefore user tried to access a field that does not exist anymore due to memory restriction
            raise ValueError("Cannot access the field to the cell at time step {} because of memory restriction".format(t))
        return self.val[tstep]
    
    def get_val_pos(self, n: int) -> ndarray:
        """
            Get the list of the values taken by the cell at step n×Δx for all time steps
        """
        return self.val[:,n]

    def get_last(self) -> ndarray:
        """
            Returns the field at the time t₁, where t₁ is the current time step of the field
        """
        return self.get_val_time(self._last_tstep)
    
    def get_prev(self) -> ndarray:
        """
            Returns the field at the time t₁-1, where t₁ is the current time step of the field
        """
        return self.get_val_time(self._last_tstep - 1)
    
    def space_fft(self, t: int, dx: float, xwindow=(0, -1)) -> tuple:
        """
            Computes the FFT of the field at the time t, in the space window given.
            Returns a tuple where the first element is the FFT, and the second the space frequency axis
        """
        field = self.get_val_time(t)
        return OneSpaceField.fft(field, dx, window=xwindow)
    
    def time_fft(self, x: int, dt: float, twindow=(0, -1)) -> tuple:
        """
            Computes the FFT of the field at the position x, in the time window given.
            Returns a tuple where the first element is the FFT, and the second the time frequency axis
        """
        field = self.get_val_pos(x)
        return OneSpaceField.fft(field, dt, window=twindow)

    @staticmethod
    def fft(field: list[float], d: float, window=(0, -1), averages=1) -> tuple:
        """
            Computes the FFT of the given signal over a given window.
        """
        if averages != 1:
            raise NotImplementedError("FFT averages not implemented yet")
        
        a, b = 0, field.shape[0]
        if window != (0, -1):
            a, b = int(window[0]), int(window[1])
        signal = field[a:b]
        n = signal.shape[-1]
        fft = scipy.fft.fft(signal)
        fftfreq = scipy.fft.fftfreq(n, d=d)
        return fft, fftfreq

    @staticmethod
    def retrieve_sines(fft: list, fftfreq: list) -> dict:
        """
            Given a FFT, returns the caracteristics of the main sines (peaks) of the signal.
            Returns a dictionary: { "frequency", "amplitude", "phase" }
        """
        n = fftfreq.size
        pos_fft = fft[fftfreq >= 0]
        pos_fftfreq = fftfreq[fftfreq >= 0]
        pos_spectrum = 2*np.abs(pos_fft/n)
        idx_peak = np.argmax(pos_spectrum)
        freq_peaks = pos_fftfreq[idx_peak]
        amp_peaks = pos_spectrum[idx_peak]
        phase_peaks = np.angle(pos_fft)[idx_peak]
        return {
            "frequency": freq_peaks,
            "amplitude": amp_peaks,
            "phase": phase_peaks
        }
