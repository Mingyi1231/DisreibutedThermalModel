from system import System_25D
import numpy as np 
import os
import sys
import csv
import math
import util.fill_space
import subprocess
import char_thermal_r
import compute_temp
import config


#---------------------------save tables for characterized chiplets, it is called only once.
cfgfile = 'configs/case1.cfg'
sys_name = os.path.splitext(cfgfile)[0]
    
#-- global consts
INTP_X_GRID_COUNT = 64
INTP_Y_GRID_COUNT = 64
KTOC = 273.15
TAMB = 45.0
CHAR_STEP = 0
CHIPLET_CHAR_POWER = 100.0

#-- read in TAP2.5D config file, which contains chiplets info
insys = config.read_config(cfgfile)

chiplet_count = insys.chiplet_count
intp_size = insys.intp_size
intp_width = intp_size
intp_height = intp_size
chiplets_width = insys.width
chiplets_height = insys.height
chiplets_power = insys.power    # not used in characterization
sys_path = insys.path

#-- create a list of unique sizes of chiplets
WH_dict, unique_widths, unique_heights = char_thermal_r.unique_WH(chiplets_width, chiplets_height)
chiplet_groupnum = len(WH_dict)

for i in range(0, chiplet_groupnum, 1):
    chiplet_name = "Chiplet" + str(i)
    char_thermal_r.char_self_r(sys_path, sys_name, chiplet_name, intp_size, unique_widths[i], unique_heights[i], CHIPLET_CHAR_POWER)
    char_thermal_r.char_mutu_r(sys_path, sys_name, chiplet_name, intp_size, unique_widths[i], unique_heights[i], CHIPLET_CHAR_POWER)

for i, (w, h) in enumerate(zip(chiplets_width, chiplets_height)):
    chiplet_name = "Chiplet_" + str(i)
    index = WH_dict[(w, h)]
    os.system('cp ' + sys_path + "Chiplet" + str(index) + ".rself" + ' ' + sys_path + chiplet_name + ".rself")
    os.system('cp ' + sys_path + "Chiplet" + str(index) + ".rmutu" + ' ' + sys_path + chiplet_name + ".rmutu")


##-- loop through each chiplet in the system
#for i in range(0, chiplet_count, 1):
#    chiplet_name = "Chiplet_" + str(i)
#    char_thermal_r.char_self_r(sys_path, sys_name, chiplet_name, intp_size, chiplets_width[i], chiplets_height[i], CHIPLET_CHAR_POWER)
#    char_thermal_r.char_mutu_r(sys_path, sys_name, chiplet_name, intp_size, chiplets_width[i], chiplets_height[i], CHIPLET_CHAR_POWER)

#--------------------------this function can be called many times to compute each chiplets temperature.
#-- read in TAP2.5D config file, which contains chiplets info

tmax , chiplets_temp = compute_temp.compute_tmax(cfgfile)

#-- print result
for i in range(0, chiplet_count, 1):
    print("Chiplet: ",  "Chiplet_"+str(i), " Temp: ", chiplets_temp[i])
print("Tmax:", tmax)

