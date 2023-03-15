#-- copmute chiplets' temperature based on characterized self- and mutual- thermal resistance
#-- and HotSpot's step file (to read chiplets coordinates)

from system import System_25D
import numpy as np 
import os
import sys
import csv
import math
import time
import config
import configparser
import util.fill_space
import subprocess
from scipy import interpolate


#-- remove blank lines in a file and dump to a new file
def rm_blank_line(filein, fileout):
    with open(filein, 'r') as r, open(fileout, 'w') as o:
        for line in r:
            #strip() function
            if line.strip():
                o.write(line)
    r.close()
    o.close()
    return


#-- read csv file
#-- format: entries separated by tabs
def readcsvfile(file):
    entries = list(csv.reader(open(file, 'r'), delimiter='\t'))
    return entries

# function to get unique values
def unique(list1):

    # initialize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


#-- read floorplan file
#-- return a list of chiplets info (name, coordinate, size)
#-- flp file format: unitname\tdx\tdy\tx0\ty0
def readflpfile(flpfile):
    global int_width, int_height    #-- interposer's width and height

    blocks = list(csv.reader(open(flpfile, 'r'), delimiter='\t'))
    max_x = 0
    max_y = 0
    i = 0
    chiplets_name = []
    chiplets_left = np.array([])
    chiplets_width = np.array([])
    chiplets_bottom = np.array([])
    chiplets_height = np.array([])
    for e in blocks:
        #-- skip blank lines
        if len(blocks[i]) == 0:
            i += 1
            continue
        #-- skip comment lines
        if blocks[i][0][0] == '#':
            i += 1
            continue
        chiplets_name += [blocks[i][0]]
        left = float(blocks[i][3])
        chiplets_left = np.append(chiplets_left, left)
        width = float(blocks[i][1])
        chiplets_width = np.append(chiplets_width, width)
        bottom = float(blocks[i][4])
        chiplets_bottom = np.append(chiplets_bottom, bottom)
        height = float(blocks[i][2])
        chiplets_height = np.append(chiplets_height, height)

        right = left + width
        top = bottom + height
        if max_x < right:
            max_x = right
        if max_y < top:
            max_y = top
        i += 1

    int_width = max_x
    int_height = max_y
    return chiplets_name, chiplets_left, chiplets_width, chiplets_bottom, chiplets_height


def compute_temp(cidx, chiplets_count, chiplets_left, chiplets_bottom, chiplets_width, chiplets_height, chiplets_power, rself, rmutu):
    global int_width, int_height

    assert(len(chiplets_left) == len(chiplets_bottom))
    assert(len(chiplets_width) == len(chiplets_height))
    assert(len(chiplets_width) == len(chiplets_power))

    x1 = rself[:,0]
    y1 = rself[:,1]
    z1 = rself[:,2]

    x2 = unique(x1)
    y2 = unique(y1)

    x2 = np.sort(x2)
    y2 = np.sort(y2)

    #-- [NOTE] assume x- y- step are the same
    step_count = len(x2)
    z2 = np.reshape(z1, (-1, step_count))
    f2d = interpolate.interp2d(x2, y2, z2)

    dist_array = rmutu[:,0]
    rmut_array = rmutu[:,1]
    assert(dist_array.shape == rmut_array.shape)

    cleft = chiplets_left[cidx]
    cbottom = chiplets_bottom[cidx]
    cwidth = chiplets_width[cidx]
    cheight = chiplets_height[cidx]
    cx = cleft + 0.5*cwidth
    cy = cbottom + 0.5*cheight

    #-- effective coordinate
    iw = 1.0e-3*intp_width    # [mm] -> [m]
    ih = 1.0e-3*intp_height    # [mm] -> [m]
    cx1 = cx if cx < 0.5*iw else (iw - cx)
    cy1 = cy if cy < 0.5*ih else (ih - cy)
    assert(0.0 < cx1)
    assert(0.5*iw >= cx1)
    assert(0.0 < cy1)
    assert(0.5*ih >= cy1)

    #-- self thermal
    rs = f2d(cx1, cy1)
    dtself = rs*chiplets_power[cidx]

    #-- mutual thermal
    dtmut = 0.0
    for i in range(0, chiplets_count, 1):
        if (i == cidx):
            i += 1
            continue
        else:
            ocleft = chiplets_left[i]
            ocbottom = chiplets_bottom[i]
            ocwidth = chiplets_width[i]
            ocheight = chiplets_height[i]
            ocx = ocleft + 0.5*ocwidth
            ocy = ocbottom + 0.5*ocheight
            dx = cx - ocx
            dy = cy - ocy
            dist = np.sqrt(dx*dx+dy*dy)
            assert(dist >= np.min(dist_array))
            assert(dist <= np.max(dist_array))
            rm = np.interp(dist, dist_array, rmut_array)
            dtmut += rm*chiplets_power[i]
            #print("name:", 'Chiplet_'+str(i), "dist:", dist, "rm:", rm, "power:", chiplets_power[i], "dt:", rm*chiplets_power[i])
            i += 1

    chiplet_temp = dtself + dtmut + TAMB

    return chiplet_temp


def clean_hotspot(path, stepfilename):
    os.system('rm ' + path + stepfilename + '{*.flp,*.lcf,*.ptrace,*.steady}')



if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python3 compute_temp.py <config_file> <step_file>")
        print("<config_file>: TAP2.5D config file, e.g. configs/sys_micro150.cfg")
        print("<step_file>: TAP2.5D step file, e.g. step_1")
        sys.exit(1)

    cfgfile = sys.argv[1]
    sys_name = os.path.splitext(cfgfile)[0]
    step_file = sys.argv[2]


    #-- global consts
    INTP_X_GRID_COUNT = 64
    INTP_Y_GRID_COUNT = 64
    KTOC = 273.15
    TAMB = 45.0


    #-- read in TAP2.5D config file, which contains chiplets info
    insys = config.read_config(cfgfile)

    chiplets_count = insys.chiplet_count
    intp_size = insys.intp_size
    intp_width = intp_size
    intp_height = intp_size
    #chiplets_width = insys.width    # [mm]
    #chiplets_height = insys.height    # [mm]
    chiplets_power = insys.power    # not used in characterization
    sys_path = insys.path    # result output file location

    #-- read floorplan file and get all chiplet's data
    #-- [NOTE] this step will be replaced by RL's data
    chiplets_name1, chiplets_left1, chiplets_width1, chiplets_bottom1, chiplets_height1 = readflpfile(sys_path+step_file+"L4_ChipLayer.flp")


    #-- filter out non-chiplet blocks read from flp file
    #-- [NOTE] this may not be necessary from RL call
    i = 0
    chiplets_name = []
    chiplets_left = np.array([])
    chiplets_bottom = np.array([])
    chiplets_width = np.array([])
    chiplets_height = np.array([])
    for cn in chiplets_name1:
        if ('Chiplet' in cn):
            chiplets_name += [cn]
            chiplets_left = np.append(chiplets_left, chiplets_left1[i])
            chiplets_bottom = np.append(chiplets_bottom, chiplets_bottom1[i])
            chiplets_width = np.append(chiplets_width, chiplets_width1[i])
            chiplets_height = np.append(chiplets_height, chiplets_height1[i])
        i += 1


    #-- load rself and rmutu table model from file
    rself_list = []
    rmutu_list = []
    for i in range(0, chiplets_count, 1):
        chiplet_name = "Chiplet_" + str(i)
        rself = np.loadtxt(sys_path+chiplet_name+".rself", delimiter='\t')
        rmutu = np.loadtxt(sys_path+chiplet_name+".rmutu", delimiter='\t')
        rself_list.append(rself)
        rmutu_list.append(rmutu)


    #-- this is the real temperature computation which will be called from RL
    tstart = time.time()
    chiplets_temp = np.array([])
    tmax = 0.0
    for i in range(0, chiplets_count, 1):
        chiplet_name = "Chiplet_" + str(i)
        t1 = compute_temp(i, chiplets_count, chiplets_left, chiplets_bottom, chiplets_width, chiplets_height, chiplets_power, rself_list[i], rmutu_list[i])
        chiplets_temp = np.append(chiplets_temp, t1)
        tmax = t1 if t1 > tmax else tmax

    tend = time.time()

    #-- print result
    for i in range(0, chiplets_count, 1):
        print("Chiplet: ",  "Chiplet_"+str(i), " Temp: ", chiplets_temp[i])
    print("Tmax:", tmax)
    print("runtime:", tend-tstart)

