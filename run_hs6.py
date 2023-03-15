from system import System_25D
import os
import sys
import util.fill_space
import subprocess

if len(sys.argv) != 2:
    print("Usage: python3 run_hotspot.py <step_file>")
    print("<step_file>: TAP2.5D step, e.g. step_1")
    sys.exit(1)

def run_hotspot(path, filename):
    proc = subprocess.Popen(["./util/hotspot",
        "-c",path+"new_hotspot.config",
	"-f",path+filename+"L4_ChipLayer.flp",
	"-p",path+filename+"_chiplet6.ptrace",
	"-steady_file",path+filename+".steady",
	"-grid_steady_file",path+filename+".grid.steady",
	"-model_type","grid",
	"-detailed_3D","on",
	"-grid_layer_file",path+filename+"layers.lcf"], 
    stdout=subprocess.PIPE, stderr = subprocess.PIPE)
    stdout, stderr = proc.communicate()
    outlist = stdout.split()
    return (max(list(map(float,outlist[3::2])))-273.15)

def clean_hotspot(path, filename):
    os.system('rm ' + path + filename + '{*.flp,*.lcf,*.ptrace,*.steady}')

path = "./output/micro150"
temp_new = run_hotspot("./output/micro150/", sys.argv[1])

print("temp=", temp_new)

