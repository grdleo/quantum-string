import matplotlib.pyplot as plt
import numpy as np

fig, (ax_mod, ax_angle) = plt.subplots(1, 2)

transmission = lambda a: 1/(1 - 1j/a)

ax_mod.set_title("$|t|$")
ax_mod.grid()

ax_angle.set_title("$arg(t)$")
ax_angle.grid()

tension = 10.0 # [N]
density = 0.001 # [kg/m]
celerity = np.sqrt(tension/density)
mass = [0.0, 0.1, 0.008, 0.007, 0.006, 0.005] # [kg]
stiff = [0.0, 2.0, 7.0, 8.0, 9.0, 10.0] # [N/m]
nb_lines = len(mass)*len(stiff)

pulsations = np.linspace(0.01, 100.0, 500)

get_cmap = lambda str_cmap, n: plt.cm.get_cmap(str_cmap, n)
for m, mi in zip(mass, range(0, len(mass))):
    for k, ki in zip(stiff, range(0, len(stiff))):
        color = (0, 0, 0)
        if k == 0.0:
            color = get_cmap("winter", len(mass))(mi)
        elif m == 0.0:
            color = get_cmap("autumn", len(stiff))(ki)

        label = ""
        if mi == 0 and ki == 1:
            label = "m=0"
        if ki == 0 and mi == 1:
            label = "K=0"
        
        if k != 0.0 and m != 0.0 or k == 0.0 and m == 0.0:
            pass
        else:
            alpha = 2*celerity*density*pulsations/(k - m*pulsations**2)
            t = transmission(alpha)

            ax_mod.plot(pulsations, np.abs(t), color=color, label=label)
            ax_angle.plot(pulsations, np.angle(t), color=color, label=label)
        
        if label != "":
            ax_mod.legend()
            ax_angle.legend()

plt.show()