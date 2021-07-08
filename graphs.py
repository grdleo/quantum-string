import numpy as np
from matplotlib import pyplot as plt

dt_space = np.array([1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001])
f = lambda w, dt: np.sqrt(4*np.sin(w*dt*0.5)**2 - 1)/dt

for deltatime in dt_space:
    limit = np.pi/3/deltatime
    omega = np.linspace(limit, 2*limit, 128)
    plt.plot(omega, f(omega, deltatime))
plt.show()