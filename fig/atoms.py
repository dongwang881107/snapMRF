import ra
import csv
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import random
from scipy import ndimage
import math

atoms = ra.read('../fig/atoms.ra')
rows, cols = np.shape(atoms)
width = 1
fs = 20

fig, ax = plt.subplots(figsize=(12,9))
random.seed(9001)
ax.plot(abs(atoms[:,random.randint(1,cols-1)]), linewidth=width)
ax.plot(abs(atoms[:,random.randint(1,cols-1)]), linewidth=width)
ax.plot(abs(atoms[:,random.randint(1,cols-1)]), linewidth=width)
ax.plot(abs(atoms[:,random.randint(1,cols-1)]), linewidth=width)
ax.plot(abs(atoms[:,random.randint(1,cols-1)]), linewidth=width)
ax.plot(abs(atoms[:,random.randint(1,cols-1)]), linewidth=width)
ax.plot(abs(atoms[:,random.randint(1,cols-1)]), linewidth=width)
ax.plot(abs(atoms[:,random.randint(1,cols-1)]), linewidth=width)
ax.plot(abs(atoms[:,random.randint(1,cols-1)]), linewidth=width)
ax.plot(abs(atoms[:,random.randint(1,cols-1)]), linewidth=width)

ax.tick_params(axis='both', which='major', length=10, direction='in')
ax.tick_params(axis='both', which='minor', length=8, direction='in')

ax.set(xlabel='TR Index', ylabel='Normalized Signal Intensity')

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
              ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(fs)

# plt.show()
fig.savefig("../fig/atoms.eps",bbox_inches='tight')
