import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

############################################################
# MOST BASIC, FOUNDATIONAL FUNCTIONS
############################################################
# Functions for basic 1D CTRW
def get_waiting_time(min_wait_time, anom_diff_exp):
    
    R = np.random.uniform(0,1)
    waiting_time = min_wait_time / (R**(1/anom_diff_exp))
    
    return waiting_time

def gaussian_step(D, t, ndim=1):
    
    var = 2*ndim*D*t
    dx = np.random.normal(scale=np.sqrt(var))
    return dx

def gauss_step_3d(xyz_array, diff_coeff, diff_time):
    
    new_coords = np.zeros(3)
    xyz_array_row = xyz_array[-1]
    xyz_array = np.vstack((xyz_array, xyz_array_row))

    for i,x_i in enumerate(xyz_array_row):
        new_coords[i] = x_i + gaussian_step(diff_coeff,diff_time)
        
    xyz_array = np.vstack((xyz_array,new_coords))
    return xyz_array

def get_distance_3d(xyz_array1,xyz_array2):
    #function to get distance between two sets of coordinates
    distance = 0
    
    for i,x_i in enumerate(xyz_array1):
        distance += (x_i - (xyz_array2[i]))**2
        
    return np.sqrt(distance)

def interaction_process(p):
    #simple call to uniform function if within distance
    #could be altered to consider probability based on distance
    repair = 0 
    q = np.random.uniform(0,1)
    if q < p:
        repair = 1
    return repair


###############################################################
# CHECKING FUNCTIONS

def check_waiting_times(nsamples, alpha, plot_range=None, min_waiting_time=1):
    # plotting probability distribution of waiting times as a check
    waiting_times = np.empty((nsamples))
    for i in range(nsamples):
        waiting_times[i] = get_waiting_time(min_waiting_time, alpha)
    plt.figure()
    plt.hist(waiting_times, bins=30, range=plot_range, density=True)

def check_steps(nsamples, D_coeff, t):
    # plotting probability distribution of step lengths as a check

    step_lengths = np.empty((nsamples))
    for i in range(nsamples):
        step_lengths[i] = gaussian_step(D_coeff, t)
    plt.figure()
    plt.hist(step_lengths, bins=30, density=True)
    
###############################################################
# COMPOSITE FUNCTIONS - WHOLE CTRWs

#1D CTRW
def ctrw(x_start, diff_coeff, diff_time, run_time, min_wait_time, anom_diff_exp, plot=0):
    """
    #Be consistent with units
    #run_time: simulation time 
    #diff_time: diffusion time
    #min_wait_time: minimum waiting time
    """
    time = 0
    x = x_start
    times = np.array([time])
    xs = np.array([x])
    while time < run_time:
        waiting_time = get_waiting_time(min_wait_time, anom_diff_exp)
        time += waiting_time
        times = np.append(times, time)
        xs = np.append(xs, xs[-1])
        x += gaussian_step(diff_coeff, diff_time)
        times = np.append(times, time)
        xs = np.append(xs, x)
    
    # correction so run time not exceeded
    times = times[:-2]
    xs = xs[:-2]
    times = np.append(times, run_time)
    xs = np.append(xs, xs[-1])
    
    data = {'t': times, 'x': xs}
    df = pd.DataFrame(data)

    return df

#3D CTRW
def ctrw_3d(initial_position, diff_coeff, diff_time, run_time, min_wait_time, anom_diff_exp, plot=0):

    time = 0
    x, y, z = initial_position
    times = np.array([time])
    xs = np.array([x])
    ys = np.array([y])
    zs = np.array([z])
    
    while time < run_time:
        waiting_time = get_waiting_time(min_wait_time, anom_diff_exp)
        time += waiting_time  # update current time
        times = np.append(times, time)
        xs = np.append(xs, xs[-1])
        ys = np.append(ys, ys[-1])
        zs = np.append(zs, zs[-1])
        x += gaussian_step(diff_coeff, diff_time)  # update current x position
        y += gaussian_step(diff_coeff, diff_time)  # update current y position
        z += gaussian_step(diff_coeff, diff_time)  # update current z position
        times = np.append(times, time)
        xs = np.append(xs, x)
        ys = np.append(ys, y)
        zs = np.append(zs, z)
    
    # correction so run time not exceeded
    times = times[:-2]
    xs = xs[:-2]
    ys = ys[:-2]
    zs = zs[:-2]
    times = np.append(times, run_time)
    xs = np.append(xs, xs[-1])
    ys = np.append(ys, ys[-1])
    zs = np.append(zs, zs[-1])
    
    data = {'t': times, 'x': xs, 'y': ys, 'z': zs}
    df = pd.DataFrame(data)

    return df

# 3D CTRW with interactions
def ctrw_3d_interaction(initial_pos, diff_coeff, diff_time, run_time, min_wait_time, anom_diff_exp,
                        int_length, delay_time, interaction_p=1, plot=0, return_trajectories=False):
    """
    initial_pos: 2d array e.g. [(x1,y1,z1), (x2,y2,z2)]
    """
    repair = 0 
    time1 = 0
    time2 = 0 
    
    x1, y1, z1 = initial_pos[0]
    x2, y2, z2 = initial_pos[1]
    
    times1 = np.array([time1])
    times2 = np.array([time2])
    
    coords1 = np.array([[x1, y1, z1]])
    coords2 = np.array([[x2, y2, z2]])
    
    interaction_coords = np.array([[0,0,0,0,0,0]])
    interaction_times = np.array([0])
    interaction_count = 0
    int_count_arr = np.array(interaction_count)
    repair_arr = np.array([repair])
        
    while (time1 < run_time and time2 < run_time) and repair==0:
        
        # the particle that is behind in time takes a step
        if time1 < time2:
            waiting_time1 = get_waiting_time(min_wait_time, anom_diff_exp)
            time1 += waiting_time1
            times1 = np.append(times1, time1)
            times1 = np.append(times1, time1)
            coords1 = gauss_step_3d(coords1, diff_coeff, diff_time)
        else:
            waiting_time2 = get_waiting_time(min_wait_time, anom_diff_exp)
            time2 += waiting_time2
            times2 = np.append(times2, time2)
            times2 = np.append(times2, time2)
            coords2 = gauss_step_3d(coords2, diff_coeff, diff_time)
        
        distance = get_distance_3d(coords1[-1],coords2[-1])
        
        if distance < int_length:
            
            if time1 < time2 and time1!=0:
                int_time = time1
            else:
                int_time = time2
               
            if int_time > delay_time:
                interaction_count += 1
                int_coords_temp = np.append(coords1[-1],coords2[-1])
                repair = interaction_process(interaction_p)
            
            if interaction_count == 1:
                interaction_times = np.array([int_time])
                interaction_times = np.reshape(interaction_times,(1,len(interaction_times)))
                interaction_coords = np.reshape(int_coords_temp,(1,len(int_coords_temp)))
                int_count_arr = np.array([interaction_count])
                repair_arr = np.array([repair])
                
            elif interaction_count > 0: 
                interaction_times = np.vstack((interaction_times,np.array([int_time])))
                interaction_coords = np.vstack((interaction_coords,int_coords_temp))
                int_count_arr = np.append(int_count_arr,interaction_count)
                repair_arr = np.append(repair_arr,repair)
    
    #######################################
    
    if repair==1:
        # record final position at time of repair 
        if time1 > time2:
            times2 = np.append(times2, time1)
            coords2 = np.vstack((coords2, coords2[-1]))
        else:
            times1 = np.append(times1, time2)
            coords1 = np.vstack((coords1, coords1[-1]))
    
    if repair==0:
        # correction so run time not exceeded
        if time1 > run_time:
            times1 = times1[:-2]
            coords1 = coords1[:-2]
        if time2 > run_time:
            times2 = times2[:-2]
            coords2 = coords2[:-2]
        if times1[-1] < run_time:
            times1 = np.append(times1, run_time)
            coords1 = np.vstack((coords1, coords1[-1]))
        if times2[-1] < run_time:
            times2 = np.append(times2, run_time)
            coords2 = np.vstack((coords2, coords2[-1]))
    
    data1 = {'t': times1, 'x': coords1[:,0], 'y': coords1[:,1], 'z': coords1[:,2]}
    df1 = pd.DataFrame(data1)
    data2 = {'t': times2, 'x': coords2[:,0], 'y': coords2[:,1], 'z': coords2[:,2]}
    df2 = pd.DataFrame(data2)

    if return_trajectories:
        return interaction_times, interaction_coords, int_count_arr,interaction_count, repair_arr, df1, df2
    else:
        return interaction_times, interaction_coords, int_count_arr, repair_arr
    