#-- plot 1 curve from csv file
import os
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.mlab import griddata

if len(sys.argv) < 2:
    print("Usage: python3 pltcurve.py <csv_file>")
    sys.exit(1)

x = np.genfromtxt(sys.argv[1], usecols=(0))
y = np.genfromtxt(sys.argv[1], usecols=(1))

plt.grid()
plt.plot(x, y)

plt.show()
