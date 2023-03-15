#-- NOTE: only python2 can run this code
#-- NOTE: python3.9.7 can also run this code
#-- plot temperature difference by taking two grid.steady files

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.patches as patches
from matplotlib.patches import Rectangle


if len(sys.argv) != 4:
    print("Usage: python3 pltflptdiff.py <floorplan_file> <grid.steady_file1> <grid.steady_file2")
    sys.exit(1)


#-- read in csv format voltage file 
#-- format: xloc\tyloc\tvoltage
def ReadCSVfile(filename):
    array = np.genfromtxt(filename)
    return array

#-- read floorplan file
#-- format: unitname\tdx\tdy\tx0\ty0
def readflpfile(file):
    blocks = list(csv.reader(open(file, 'r'), delimiter='\t'))
    return blocks

#-- floorplan file
blocks = readflpfile(sys.argv[1])


max_x = 0
max_y = 0
i = 0
for e in blocks:
    #-- skip blank lines
    if len(blocks[i]) == 0:
        i += 1
        continue
    #-- skip comment lines
    if blocks[i][0][0] == '#':
        i += 1
        continue
    name = blocks[i][0]
    left, width = float(blocks[i][3]), float(blocks[i][1])
    bottom, height = float(blocks[i][4]), float(blocks[i][2])
    right = left + width
    top = bottom + height
    if max_x < right:
        max_x = right
    if max_y < top:
        max_y = top
    i += 1


total_width = max_x
total_height = max_y



#np.random.seed(19680801)
#npts = 200
#ngridx = 100
#ngridy = 200

res_array1 = ReadCSVfile(sys.argv[2])
res_array2 = ReadCSVfile(sys.argv[3])
npts1 = len(res_array1)
npts2 = len(res_array2)
assert(npts1 == npts2)
gc = res_array1[:,0]
z = res_array1[:,1] - res_array2[:,1]
z_list = z.tolist()

error = z/res_array1[:,1]

min_index = z_list.index(min(z_list))
gridT1 = res_array1[min_index:min_index+1,1]
gridT2 = res_array2[min_index:min_index+1,1]


print("the grid temperature for the max Tdiff:", gridT1-273.15, gridT2-273.15)

#-- print statistics
print("Total grid count: %d" %gc.size)
print("Min grid temp: %f[C]" % np.min(z))
print("Avg grid temp: %f[C]" % np.average(z))
print("Max grid temp: %f[C]" % np.max(z))


#-- print error statistics
print("Total grid count: %d" %error.size)
print("Min grid temp: %f[C]" % np.min(error))
print("Avg grid temp: %f[C]" % np.average(error))
print("Max grid temp: %f[C]" % np.max(error))


fig, (ax) = plt.subplots(nrows=1)

'''
# -----------------------
# Interpolation on a grid
# -----------------------
# A contour plot of irregularly spaced data coordinates
# via interpolation on a grid.

# Create grid values first.
xi = np.linspace(-2.1, 2.1, ngridx)
yi = np.linspace(-2.1, 2.1, ngridy)

# Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
triang = tri.Triangulation(x, y)
interpolator = tri.LinearTriInterpolator(triang, z)
Xi, Yi = np.meshgrid(xi, yi)
zi = interpolator(Xi, Yi)

# Note that scipy.interpolate provides means to interpolate data on a grid
# as well. The following would be an alternative to the four lines above:
# from scipy.interpolate import griddata
# zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')

ax1.contour(xi, yi, zi, levels=14, linewidths=0.5, colors='k')
cntr1 = ax1.contourf(xi, yi, zi, levels=14, cmap="RdBu_r")

fig.colorbar(cntr1, ax=ax1)
ax1.plot(x, y, 'ko', ms=3)
ax1.set(xlim=(-2, 2), ylim=(-2, 2))
ax1.set_title('grid and contour (%d points, %d grid points)' %
              (npts, ngridx * ngridy))
'''

#x_grid_count = 64
#y_grid_count = 64
x_grid_count = 128
y_grid_count = 128
# x_grid_count = y_grid_count = np.sqrt(np.len(grid_idx))


#-- NOTE: hotspot's grid index is computed from top left corner
x = np.array([])
y = np.array([])
for i in range(y_grid_count):
    for j in range(x_grid_count):
        x = np.append(x, j*max_x/x_grid_count)
        y = np.append(y, total_height - i*max_y/y_grid_count)

'''
for i in range(x_grid_count):
    for j in range(y_grid_count):
        x = np.append(x, i*max_x/x_grid_count)
        y = np.append(y, j*max_y/y_grid_count)
'''

# ----------
# Tricontour
# ----------
# Directly supply the unordered, irregularly spaced coordinates
# to tricontour.

ax.tricontour(x, y, error, levels=14, linewidths=0.5, colors='k')
cntr2 = ax.tricontourf(x, y, error, levels=14, cmap="RdBu_r")



# build a rectangle for each unit in axes coords
i = 0
for e in blocks:
    #-- skip blank lines
    if len(blocks[i]) == 0:
        i += 1
        continue
    #-- skip comment lines
    if blocks[i][0][0] == '#':
        i += 1
        continue
    name = blocks[i][0]
    left, width = float(blocks[i][3]), float(blocks[i][1])
    bottom, height = float(blocks[i][4]), float(blocks[i][2])
    right = left + width
    top = bottom + height

    ax.add_patch(Rectangle((left, bottom), width, height,
        fill=False, clip_on=False,
        edgecolor='black', facecolor='blue', lw=1))
    ax.text(0.5*(left+right), 0.5*(bottom+top), name,
        fontsize=10, color='black')
    i += 1


fig.colorbar(cntr2, ax=ax)
ax.plot(x, y, 'ko', ms=3)
#ax2.set(xlim=(2, 2), ylim=(-2, 2))
#ax2.set_title('tricontour (%d points)' % npts)
#ax2.set_title('Voltage noise contour (%s) (%d points)' % sys.argv[1], npts)
ax.set_title('Temperature difference contour (%s)' % sys.argv[1])
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)


plt.subplots_adjust(hspace=0.5)
plt.show()


#-- plot temperature histogram
n_bins = 20

print("No. of nodes: %d" % len(z))

# Creating histogram
#fig, axs = plt.subplots(1, 1, figsize =(10, 7), tight_layout = True)
fig, axs = plt.subplots(1, 1)

axs.hist(z, bins = n_bins)

#plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
axs.set_title('Temperature difference histogram (%s)' % sys.argv[1])
axs.set_xlabel("Temperature difference [C]")
axs.set_ylabel("Grid count")

# Show plot
plt.show()

