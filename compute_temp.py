#-- copmute chiplets' temperature based on characterized self- and mutual- thermal resistance

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
import pandas as pd
from collections import defaultdict

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



def compute_temp(cidx, chiplets_count, intp_size, chiplets_left, chiplets_bottom, chiplets_width, chiplets_height, chiplets_power, rself, rmutu):
    intp_width = intp_size
    intp_height = intp_size

    INTP_X_GRID_COUNT = 64
    INTP_Y_GRID_COUNT = 64
    KTOC = 273.15
    TAMB = 45.0
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
    
    #-- 2D rmutu data processing
    xx1 = rmutu[:,0]
    yy1 = rmutu[:,1]
    zz1 = rmutu[:,2]
    
    xx2 = unique(xx1)
    yy2 = unique(yy1)

    xx2 = np.sort(xx2)
    yy2 = np.sort(yy2)

    #-- [NOTE] assume x- y- step are the same
    step_count1 = len(xx2)
    zz2 = np.reshape(zz1, (-1, step_count1))
    ff2dd = interpolate.interp2d(xx2, yy2, zz2)

#    dist_array = rmutu[:,0]
#    rmut_array = rmutu[:,1]
#    assert(dist_array.shape == rmut_array.shape)

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
#            dist = np.sqrt(dx*dx+dy*dy)
#            assert(dist >= np.min(dist_array))
#            assert(dist <= np.max(dist_array))
#            rm = np.interp(dist, dist_array, rmut_array)
            rm = ff2dd(dx, dy)
            dtmut += rm*chiplets_power[i]
            #print("name:", 'Chiplet_'+str(i), "dist:", dist, "rm:", rm, "power:", chiplets_power[i], "dt:", rm*chiplets_power[i])
            i += 1

    chiplet_temp = dtself + dtmut + TAMB

    return chiplet_temp


def clean_hotspot(path, stepfilename):
    os.system('rm ' + path + stepfilename + '{*.flp,*.lcf,*.ptrace,*.steady}')

def unique_WH(chiplets_widths, chiplets_heights):
    d = defaultdict(int)
    seen = set()
    idx = 0
    unique_widths, unique_heights = [], []
    for w, h in zip(chiplets_widths, chiplets_heights):
        if (w, h) not in seen:
            d[(w, h)] = idx
            idx += 1
            seen.add((w, h))
            unique_widths.append(w)
            unique_heights.append(h)
    return d, unique_widths, unique_heights  #-- d is a dictionary: {key: value} = {(w, h):group index}

def compute_tmax(cfgfile):

    sys_name = os.path.splitext(cfgfile)[0]
    
    #-- read in TAP2.5D config file, which contains chiplets info
    insys = config.read_config(cfgfile)

    chiplets_count = insys.chiplet_count
    intp_size = insys.intp_size
    intp_width = intp_size
    intp_height = intp_size
    chiplets_width = 1.0e-3*np.array(insys.width)    # [mm] -> [m]
    chiplets_height = 1.0e-3*np.array(insys.height)    # [mm] -> [m]
    assert(0 < len(insys.x))
    chiplets_x = 1.0e-3*np.array(insys.x)    # [mm] -> [m]
    assert(0 < len(insys.y))
    assert(len(insys.x) == len(insys.y))
    chiplets_y = 1.0e-3*np.array(insys.y)    # [mm] -> [m]
    chiplets_power = np.array(insys.power)
    sys_path = insys.path    # result output file location

    chiplets_left = np.array([])
    chiplets_bottom = np.array([])
    #chiplets_width = np.array([])
    #chiplets_height = np.array([])
    for i in range(0, chiplets_count, 1):
        chiplets_left = np.append(chiplets_left, chiplets_x[i]-0.5*chiplets_width[i])
        chiplets_bottom = np.append(chiplets_bottom, chiplets_y[i]-0.5*chiplets_height[i])

#    #-- load rself and rmutu table model from file
#    rself_list = []
#    rmutu_list = []
#    for i in range(0, chiplets_count, 1):
#        chiplet_name = "Chiplet_" + str(i)
#        rself = np.loadtxt(sys_path+chiplet_name+".rself", delimiter='\t')
#        rmutu = np.loadtxt(sys_path+chiplet_name+".rmutu", delimiter='\t')
#        rself_list.append(rself)
#        rmutu_list.append(rmutu)

#    #-- this is the real temperature computation which will be called from RL
#    tstart = time.time()
#    chiplets_temp = np.array([])
#    tmax = 0.0
#    for i in range(0, chiplets_count, 1):
#        chiplet_name = "Chiplet_" + str(i)
#        t1 = compute_temp(i, chiplets_count, intp_size, chiplets_left, chiplets_bottom, chiplets_width, chiplets_height, chiplets_power, rself_list[i], rmutu_list[i])
#        chiplets_temp = np.append(chiplets_temp, t1)
#        tmax = t1 if t1 > tmax else tmax
#
#    tend = time.time()

#----------------------------------------------------------------------------------------------------
    #-- load rself and rmutu table model from group number, map from the group number to each chiplet sequence number.
    rself_list = []
    rmutu_list = []
    WH_dict, unique_width, unique_height = unique_WH(chiplets_width, chiplets_height)
    chiplet_groupnum = len(WH_dict)

    for i in range(0, chiplet_groupnum, 1):
        chiplet_name = "Chiplet" + str(i)
        rself = np.loadtxt(sys_path+chiplet_name+".rself", delimiter='\t')
        rmutu = np.loadtxt(sys_path+chiplet_name+".rmutu", delimiter='\t')
        rself_list.append(rself)
        rmutu_list.append(rmutu)

    #-- this is the real temperature computation which will be called from RL
    tstart = time.time()
    chiplets_temp = np.array([])
    tmax = 0.0
    for i in range(0, chiplets_count, 1):
        wh = list(zip(chiplets_width, chiplets_height))
        listnum = WH_dict[(wh[i])] 
        t1 = compute_temp(i, chiplets_count, intp_size, chiplets_left, chiplets_bottom, chiplets_width, chiplets_height, chiplets_power, rself_list[listnum], rmutu_list[listnum])
        chiplets_temp = np.append(chiplets_temp, t1)
        tmax = t1 if t1 > tmax else tmax

    tend = time.time()



    return tmax, chiplets_temp


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python3 compute_temp.py <config_file>")
        print("<config_file>: TAP2.5D config file, e.g. configs/sys_micro150.cfg")
        sys.exit(1)

    cfgfile = sys.argv[1]
    sys_name = os.path.splitext(cfgfile)[0]
    #step_file = sys.argv[2]


    #-- global consts
    INTP_X_GRID_COUNT = 64
    INTP_Y_GRID_COUNT = 64
    KTOC = 273.15
    TAMB = 45.0

    #-- this is the real temperature computation which will be called from RL
    tstart = time.time()
    chiplets_temp = np.array([])
    tmax = 0.0
#    for i in range(0, chiplets_count, 1):
#        chiplet_name = "Chiplet_" + str(i)
#        t1 = compute_temp(i, chiplets_count, chiplets_left, chiplets_bottom, chiplets_width, chiplets_height, chiplets_power, rself_list[i], rmutu_list[i])
#        chiplets_temp = np.append(chiplets_temp, t1)
#        tmax = t1 if t1 > tmax else tmax

    Tmax, chiplets_temp = compute_tmax(cfgfile)
    tend = time.time()
    chiplets_count = len(chiplets_temp)
    #-- print result
    for i in range(0, chiplets_count, 1):
        print("Chiplet: ",  "Chiplet_"+str(i), " Temp: ", chiplets_temp[i])
    print("Tmax:", Tmax)
    print("runtime:", tend-tstart)


