import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import CTRW_base as CTRW

####################################################################
# Update on Older base functions for 4 CTRW model
####################################################################
def gauss_step_3d_bounded(xyz_array, diff_coeff, diff_time,r_nuc=3000):
    
    new_coords = np.zeros(3)
    
    xyz_array_row = xyz_array[-1]
    
    distance_centre = r_nuc + 1
    
    while distance_centre > r_nuc:
        
        for i,x_i in enumerate(xyz_array_row):
        
            new_coords[i] = x_i + CTRW.gaussian_step(diff_coeff,diff_time)
            
        distance_centre = CTRW.get_distance_3d(new_coords,np.array([0,0,0]))
            
    xyz_array = np.vstack((xyz_array,new_coords))
    return xyz_array

def gauss_step_3d_pair(xyz_array1, xyz_array2, diff_coeff, diff_time):
    
    new_coords1 = np.zeros(3)
    new_coords2 = np.zeros(3)
    
    xyz_array_row1 = xyz_array1[-1]
    xyz_array_row2 = xyz_array2[-1]
    
    # two particles move together
    for i in range(3):
        dx = CTRW.gaussian_step(diff_coeff, diff_time)
        new_coords1[i] = xyz_array_row1[i] + dx
        new_coords2[i] = xyz_array_row2[i] + dx
        
    xyz_array1 = np.vstack((xyz_array1, new_coords1))
    xyz_array2 = np.vstack((xyz_array2, new_coords2))
    
    return xyz_array1, xyz_array2

def gauss_step_3d_pair_bounded(xyz_array1, xyz_array2, diff_coeff, diff_time,r_nuc=3000):
    
    new_coords1 = np.zeros(3)
    new_coords2 = np.zeros(3)
    
    xyz_array_row1 = xyz_array1[-1]
    xyz_array_row2 = xyz_array2[-1]
    
    distance_centre1 = r_nuc + 1
    distance_centre2 = r_nuc + 1
    
    while distance_centre1 > r_nuc or distance_centre2 > r_nuc:
        # two particles move together
        for i in range(3):
            dx = CTRW.gaussian_step(diff_coeff, diff_time)
            new_coords1[i] = xyz_array_row1[i] + dx
            new_coords2[i] = xyz_array_row2[i] + dx
            
        distance_centre1 = CTRW.get_distance_3d(new_coords1,np.array([0,0,0]))
        distance_centre2 = CTRW.get_distance_3d(new_coords2,np.array([0,0,0]))
        
    xyz_array1 = np.vstack((xyz_array1, new_coords1))
    xyz_array2 = np.vstack((xyz_array2, new_coords2))
    
    return xyz_array1, xyz_array2

def interaction_process(p):
    #simple call to uniform function if within distance
    #could be altered to consider probability based on distance
    repair = 0 
    q = np.random.uniform(0,1)
    if q < p:
        repair = 1
    return repair

def repair_process(interaction_index, interaction_p):
    """
    Updated Function to also consider misrepairs
    """
    repair = interaction_process(interaction_p)
    
    #index 0 = i12, index 5 = i34, which are ends of same pair
    # if not these indices, misrepair between diff strands
    if interaction_index != 0 and interaction_index != 5:
        repair = repair*(-1)
        
    return repair

##############################################################################
# Function to generate breaksites randomly in bounding of nucleus
##############################################################################

def gen_breaksites(r_nuc=3000,site_separation=500):

    u1 = np.random.uniform(0,1)
    u2 = np.random.uniform(0,1)

    inclination = np.arccos(1-2*u1)
    azimuth = 2*(np.pi)*u2
    
    #cartesian conversion
    x1 = r_nuc*np.cos(azimuth)*np.sin(inclination)
    y1 = r_nuc*np.sin(azimuth)*np.sin(inclination)
    z1 = r_nuc*np.cos(inclination)

    d_centre = r_nuc+1
    while d_centre > r_nuc:
        
        u3 = np.random.uniform(0,1)
        u4 = np.random.uniform(0,1)
    
        inclination = np.arccos(1-2*u3)
        azimuth = 2*(np.pi)*u4
       
        rx2 = site_separation*np.cos(azimuth)*np.sin(inclination)
        ry2 = site_separation*np.sin(azimuth)*np.sin(inclination)
        rz2 = site_separation*np.cos(inclination)
        
        x2 = x1 + rx2
        y2 = y1 + ry2
        z2 = z1 + rz2
        
        d_centre = CTRW.get_distance_3d([(x1,y1,z1),(x2,y2,z2)])
    
    coords_list = [(x1,y1,z1),(x2,y2,z2)]
    
    return coords_list

##############################################################################
# Preliminary functions for 4 CTRW 
##############################################################################

def take_step_coupled_ctrw(sim_clock, move_index, coords1, coords2, coords3, coords4, D_break_sites, D_break_ends, diff_time,
                           run_time, min_wait_time, anom_diff_exp):
    """
    Depending on move_index, either one of the break ends will move individually (smaller motion) or the 
    two break ends of one break site will move together (bigger motion).
    """
    # Should we keep it as bounded???

    if move_index==0:
        # break end 1 moves (smaller motion)
        coords1 = gauss_step_3d_bounded(coords1, D_break_ends, diff_time)
        # ends 2,3,4 stay
        coords2 = np.vstack((coords2, coords2[-1]))
        coords3 = np.vstack((coords3, coords3[-1]))
        coords4 = np.vstack((coords4, coords4[-1]))
        
    elif move_index==1:
        # break end 2 moves (smaller motion)
        coords2 = gauss_step_3d_bounded(coords2, D_break_ends, diff_time)
        # ends 1,3,4 stay
        coords1 = np.vstack((coords1, coords1[-1]))
        coords3 = np.vstack((coords3, coords3[-1]))
        coords4 = np.vstack((coords4, coords4[-1]))
        
    elif move_index==2:
        # break end 3 moves (smaller motion)
        coords3 = gauss_step_3d_bounded(coords3, D_break_ends, diff_time)
        # ends 1,2,4 stay
        coords1 = np.vstack((coords1, coords1[-1]))
        coords2 = np.vstack((coords2, coords2[-1]))
        coords4 = np.vstack((coords4, coords4[-1]))
    
    elif move_index==3:
        # break end 4 moves (smaller motion)
        coords4 = gauss_step_3d_bounded(coords4, D_break_ends, diff_time)
        # ends 1,2,3 stay
        coords1 = np.vstack((coords1, coords1[-1]))
        coords2 = np.vstack((coords2, coords2[-1]))
        coords3 = np.vstack((coords3, coords3[-1]))
        
    elif move_index==4:
        # break site 12 moves (bigger motion) - ends 1 and 2 move together        
        coords1, coords2 = gauss_step_3d_pair_bounded(coords1, coords2, D_break_sites, diff_time)
        # ends 3,4 stay
        coords3 = np.vstack((coords3, coords3[-1]))
        coords4 = np.vstack((coords4, coords4[-1]))
        
    elif move_index==5:
        # break site 34 moves (bigger motion) - ends 3 and 4 move together        
        coords3, coords4 = gauss_step_3d_pair_bounded(coords3, coords4, D_break_sites, diff_time)
        # ends 1,2 stay
        coords1 = np.vstack((coords1, coords1[-1]))
        coords2 = np.vstack((coords2, coords2[-1]))
    
    return coords1, coords2, coords3, coords4

def check_for_interaction(sim_t, c1, c2, c3, c4, int_range, int_in, delay_time, interaction_p, int_count):
    """
    sim_t is current simulation time
    Function to take positions, times and check for repair.
    t1-t4 are current walk times, c1-c4 are their coordinates xyz
    int_range is interaction range, int_in is exisiting interaction array
    int_in has form: [i12,i13,i14,i23,i24,i34,repair]
    where i12 is a tally of interactions between 1 and 2 etc, repair is 1,0,-1
    for repair, no repair, or misrepair
    interaction_p is repair probability
    """
    #array of combinations of times and coordinates to iterate through later
    coords = np.array([[c1,c2],[c1,c3],[c1,c4],[c2,c3],[c2,c4],[c3,c4]],dtype=object)
    distances = np.zeros(6)
    
    for i,cc in enumerate(coords):
        distances[i] = CTRW.get_distance_3d(cc[0],cc[1])
        
    # use minimum distance to check for interaction between the two nearest particles
    min_dist = np.min(distances)
    interaction_index = np.argmin(distances)
    
    repair = 0
    int_t = 0
    int_out = int_in
        
    if min_dist < int_range and sim_t > delay_time:
        int_count += 1
        repair = repair_process(interaction_index, interaction_p)
        int_t = sim_t  # interaction time is the current simulation time
        int_out[-1] = repair
        int_out[interaction_index] = 1
                 
    return int_t, int_out, repair, int_count

##########################################################################
# 4 CTRW 
#########################################################################

def coupled_ctrw(initial_pos, D_break_sites, D_break_ends, diff_time, run_time, min_wait_time, anom_diff_exp, int_length,
                 delay_time, interaction_p):
    """
    Separation of the two break ends of one DSB is assumed to be 0.
    initial_pos: 2d array, e.g. [(x1,y1,z1), (x2,y2,z2)], describes initial positions of the two break sites
    """    
    # ends of first break site
    x1, y1, z1 = initial_pos[0]
    x2, y2, z2 = initial_pos[0]
    
    # ends of second break site
    x3, y3, z3 = initial_pos[1]
    x4, y4, z4 = initial_pos[1]
    
    coords1 = np.array([[x1, y1, z1]])
    coords2 = np.array([[x2, y2, z2]])
    coords3 = np.array([[x3, y3, z3]])
    coords4 = np.array([[x4, y4, z4]])
    
    #possible correct repair counts
    i12,i34 = 0,0
    #possible misrepair counts
    i13,i14,i23,i24 = 0,0,0,0
    
    repair = 0 
    int_t = 0
    #int_arr = np.array([int_t,i12,i13,i14,i23,i24,i34,repair])
    int_arr = np.array([i12,i13,i14,i23,i24,i34,repair])
    # keep int_t separate from int_arr, otherwise it will get rounded to integer
    
    individual_clocks = np.zeros(6)
    for i in range(6):
        individual_clocks[i] = CTRW.get_waiting_time(min_wait_time, anom_diff_exp)
    
    sim_clock = 0  # current time in the simulation (for all particles)
    sim_times = np.array([0])  # time array to be used for all particles
    
    int_count = 0
    
    while sim_clock < run_time and repair == 0:
        
        sim_clock = min(individual_clocks)
        move_index = np.argmin(individual_clocks)
        
        # record current simulation time
        sim_times = np.append(sim_times, sim_clock)
        # record position of each particle at this simulation time
        coords1, coords2, coords3, coords4 = take_step_coupled_ctrw(sim_clock, move_index, coords1, coords2, coords3, coords4,
                                                                    D_break_sites, D_break_ends, diff_time, run_time,
                                                                    min_wait_time, anom_diff_exp)
        
        c1, c2, c3, c4 = coords1[-1], coords2[-1], coords3[-1], coords4[-1]
        
        int_t, int_arr, repair, int_count= check_for_interaction(sim_clock, c1, c2, c3, c4, int_length, int_arr, delay_time,
                                                                 interaction_p, int_count)
        
        # update relevant individual clock
        waiting_t = CTRW.get_waiting_time(min_wait_time, anom_diff_exp)
        individual_clocks[move_index] += waiting_t
       
    
    data1 = {'t': sim_times, 'x': coords1[:,0], 'y': coords1[:,1], 'z': coords1[:,2]}
    df1 = pd.DataFrame(data1)
    data2 = {'t': sim_times, 'x': coords2[:,0], 'y': coords2[:,1], 'z': coords2[:,2]}
    df2 = pd.DataFrame(data2)
    data3 = {'t': sim_times, 'x': coords3[:,0], 'y': coords3[:,1], 'z': coords3[:,2]}
    df3 = pd.DataFrame(data3)
    data4 = {'t': sim_times, 'x': coords4[:,0], 'y': coords4[:,1], 'z': coords4[:,2]}
    df4 = pd.DataFrame(data4)

    return int_t, int_arr, df1, df2, df3, df4, repair, sim_clock, int_count

def coupled_ctrw_for_mc(initial_pos, D_break_sites, D_break_ends, diff_time, run_time, min_wait_time, anom_diff_exp, int_length,
                        delay_time, interaction_p):
    """
    Separation of the two break ends of one DSB is assumed to be 0.
    initial_pos: 2d array, e.g. [(x1,y1,z1), (x2,y2,z2)], describes initial positions of the two break sites
    Slimmed down to not record pandas and originally didnt have plotting while earlier one did
    """    
    # ends of first break site
    x1, y1, z1 = initial_pos[0]
    x2, y2, z2 = initial_pos[0]
    
    # ends of second break site
    x3, y3, z3 = initial_pos[1]
    x4, y4, z4 = initial_pos[1]
    
    coords1 = np.array([[x1, y1, z1]])
    coords2 = np.array([[x2, y2, z2]])
    coords3 = np.array([[x3, y3, z3]])
    coords4 = np.array([[x4, y4, z4]])
    
    #possible correct repair counts
    i12,i34 = 0,0
    #possible misrepair counts
    i13,i14,i23,i24 = 0,0,0,0
    
    repair = 0 
    int_t = 0
    #int_arr = np.array([int_t,i12,i13,i14,i23,i24,i34,repair])
    int_arr = np.array([i12,i13,i14,i23,i24,i34,repair])
    # keep int_t separate from int_arr, otherwise it will get rounded to integer
    
    individual_clocks = np.zeros(6)
    for i in range(6):
        individual_clocks[i] = CTRW.get_waiting_time(min_wait_time, anom_diff_exp)
    
    sim_clock = 0  # current time in the simulation (for all particles)
    sim_times = np.array([0])  # time array to be used for all particles
    
    int_count = 0
    
    while sim_clock < run_time and repair == 0:
        
        sim_clock = min(individual_clocks)
        move_index = np.argmin(individual_clocks)
        
        # record current simulation time
        sim_times = np.append(sim_times, sim_clock)
        # record position of each particle at this simulation time
        coords1, coords2, coords3, coords4 = take_step_coupled_ctrw(sim_clock, move_index, coords1, coords2, coords3, coords4,
                                                                    D_break_sites, D_break_ends, diff_time, run_time,
                                                                    min_wait_time, anom_diff_exp)
        
        c1, c2, c3, c4 = coords1[-1], coords2[-1], coords3[-1], coords4[-1]
        
        int_t, int_arr, repair, int_count= check_for_interaction(sim_clock, c1, c2, c3, c4, int_length, int_arr, delay_time,
                                                                 interaction_p, int_count)
        
        # update relevant individual clock
        waiting_t = CTRW.get_waiting_time(min_wait_time, anom_diff_exp)
        individual_clocks[move_index] += waiting_t

    return int_t, int_arr, int_count