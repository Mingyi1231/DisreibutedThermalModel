# DisreibutedThermalModel
The characterization have done for compute chiplet and interposer temperature, the result are in the "outputs" directory,  or you can run char_thermal_r.py file for another characterization which will take some time, all config files are in "configs" directory named by *.cfg, you can also add some config example in this directory. 

Run char_thermal_r.py by:
python3 char_thermal_r.py configs/case1.cfg

Then we can compute the gird temperature, compute_temp_grid.py is normal serial calculation of interpolation results, compute_temp_parallel.py is Improved parallel computation interpolation results, The running time is reduced by about half. 

Compute temperature by:
python3 compute_temp_grid.py configs/case1.cfg
python3 compute_temp_parallel.py configs/case1.cfg
