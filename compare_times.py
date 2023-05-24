import numpy as np
import matplotlib.pyplot as plt




openfoam_log = '/Users/sbarwey/Files/openfoam_cases/backward_facing_step/Backward_Facing_Step/speedup_analysis/Re_39076/out_grep'

dt_gnn = 1e-4




# Read the log:
file1 = open(openfoam_log, 'r')
lines = file1.readlines()

physical_time = lines[::2]
exec_time = lines[1::2]

N = len(physical_time)
if (N != len(exec_time)):
    print('something wrong')
    print('something wrong')
    print('something wrong')
    print('something wrong')

# Make float 
t_phys = np.zeros(len(physical_time))
t_exec = np.zeros(len(exec_time))

for i in range(len(physical_time)):

    # physical time
    temp = physical_time[i]
    val = temp.split(' = ')[-1]
    val = val.split('\n')[0]
    t_phys[i] = float(val)


    # execution time 
    temp = exec_time[i]
    temp = temp.split(' ClockTime ')[0]
    val = temp.split(' = ')[-1].split(' s ')[0]
    t_exec[i] = float(val)



# Re-format 
dt_phys = t_phys[1:] - t_phys[:-1]
t_exec = t_exec[1:] - t_exec[:-1]



# Get approximate speedup, based on dt_gnn  
# q: how long does it take to advance solution by dt_gnn 
cml_phys_time = 0 
cml_exec_time = 0
dt_final = []
t_exec_final = []
for i in range(len(dt_phys)):
    cml_phys_time += dt_phys[i] 
    cml_exec_time += t_exec[i]
    if cml_phys_time >= dt_gnn:
        dt_final.append(cml_phys_time)
        t_exec_final.append(cml_exec_time)
        cml_phys_time = 0
        cml_exec_time = 0

    


print('Openfoam average execution time for equivalent GNN forward pass = %g s' %(np.mean(t_exec_final)))


