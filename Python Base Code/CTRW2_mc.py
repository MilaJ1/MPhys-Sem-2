import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import CTRW_base as CTRW

def ctrw_3d_interaction_mc(nsamples, initial_pos, diff_coeff, diff_time, run_time, min_wait_time, anom_diff_exp,
                           int_length, delay_time, interaction_p=1, plot=0):

    j = 0
    for i in np.arange(nsamples):        
        ts_temp,xyzs_temp,counts_temp,repair_temp = CTRW.ctrw_3d_interaction(initial_pos, diff_coeff, diff_time, run_time,
                                                                        min_wait_time, anom_diff_exp, int_length, 
                                                                        delay_time, interaction_p=interaction_p, plot=plot)
     
        repair_temp = np.reshape(repair_temp,(len(repair_temp),1))
        counts_temp = np.reshape(counts_temp,(np.size(counts_temp),1))

        if counts_temp[0] != 0:
            mc_sample = np.full((len(counts_temp),1),i)
            data_temp = np.hstack((mc_sample,ts_temp,xyzs_temp,counts_temp,repair_temp))

            if j == 0:
                repair_data = data_temp
            else: 
                repair_data = np.vstack((repair_data,data_temp))

            j+=1
                
    if j != 0:
        repair_df = pd.DataFrame(data=repair_data, 
                    columns=['mc_step', 't','x1','y1','z1','x2','y2','z2','interaction','repair'])
    else:
        repair_df = pd.DataFrame()
        
    return repair_df

def ctrw_interaction_mc_D(nsamples_per_D, repeats, initial_pos, diff_coeffs, diff_time, run_time, min_wait_time, anom_diff_exp,
                          int_length, delay_time, interaction_p=1, plot=0,print_out=0):
    
    repair_avgs = np.empty(len(diff_coeffs))
    repair_stds = np.empty(len(diff_coeffs))
    for k,D in enumerate(diff_coeffs):
        if print_out ==1:
            print('-------------')
            print(D)
            print('-------------')

        temp_repair_rates = np.empty(repeats)
        for i in np.arange(repeats):
            
            data = ctrw_3d_interaction_mc(nsamples_per_D, initial_pos, D, diff_time, run_time, min_wait_time, anom_diff_exp,
                                          int_length, delay_time, interaction_p=interaction_p, plot=plot)
            
            if len(data.index) == 0:
                temp_repair_rates[i] = np.NaN
            else:  
                repair_events = len(data[data['repair']==1.0])
                temp_repair_rates[i] = repair_events
            
        repair_avgs[k] = np.nanmean(temp_repair_rates)
        repair_stds[k] = np.nanstd(temp_repair_rates)
        
    return repair_avgs, repair_stds

def ctrw_interaction_mc_D_r(nsamples_per_D, repeats, separations, diff_coeffs, diff_time, run_time, min_wait_time, anom_diff_exp,
                           int_length, delay_time, interaction_p=1, plot=0,print_out=0):
    
    N_rows = int(len(diff_coeffs)*len(separations))    
    repair_data_D_r = np.empty((N_rows,4)) 

    for j,r in enumerate(separations):

        if print_out ==1:
            print("-----------------------")
            print(r)
            print("-----------------------")
                
        initial_pos = [(0,0,0),(0,0,r)]  
        repair_avgs = np.empty(len(diff_coeffs))
        repair_stds = np.empty(len(diff_coeffs))
    
        for k,D in enumerate(diff_coeffs):
            
            if print_out ==1:
                print(j*len(diff_coeffs)+k)
                print(r,D)

            temp_repair_rates = np.empty(repeats)

            for i in np.arange(repeats):
                data = ctrw_3d_interaction_mc(nsamples_per_D, initial_pos, D, diff_time, run_time, min_wait_time, anom_diff_exp,
                                              int_length, delay_time, interaction_p=interaction_p, plot=plot)
                
                if len(data.index) == 0:
                    temp_repair_rates[i] = 0
                else:  
                    repair_events = len(data[data['repair']==1.0])
                    temp_repair_rates[i] = repair_events

            repair_avg = np.nanmean(temp_repair_rates)
            repair_std = np.nanstd(temp_repair_rates)
            repair_data_row = np.array([r,D,repair_avg,repair_std])
            repair_data_D_r[j*len(diff_coeffs)+k] = repair_data_row

    repair_df_D_r = pd.DataFrame(data=repair_data_D_r, columns=['r','D','Repair Rate','Repair Rate Std'])                 
        
    return repair_df_D_r

def ctrw2_mc_D_delay_t(nsamples_per_D, repeats, delay_times, diff_coeffs, diff_time, run_time, min_wait_time, anom_diff_exp,
                           int_length, separation=0, interaction_p=1, plot=0,print_out=0):
    
    
    N_rows = int(len(diff_coeffs)*len(delay_times))    
    repair_data_D_delay_t = np.empty((N_rows,4)) 
    print(N_rows)

    for j,delay_t in enumerate(delay_times):

        if print_out ==1:
            print("-----------------------")
            print(delay_t)
            print("-----------------------")
                
        initial_pos = [(0,0,0),(0,0,separation)]  
    
        for k,D in enumerate(diff_coeffs):
            
            if print_out ==1:
                print(j*len(diff_coeffs)+k)
                print(delay_t,D)

            temp_repair_rates = np.empty(repeats)

            for i in np.arange(repeats):
                data = ctrw_3d_interaction_mc(nsamples_per_D, initial_pos, D, diff_time, run_time, min_wait_time, anom_diff_exp,
                                              int_length, delay_t, interaction_p=interaction_p, plot=plot)
                
                if len(data.index) == 0:
                    temp_repair_rates[i] = 0
                else:  
                    repair_events = len(data[data['repair']==1.0])
                    temp_repair_rates[i] = repair_events

            repair_avg = np.nanmean(temp_repair_rates)
            repair_std = np.nanstd(temp_repair_rates)
            repair_data_row = np.array([delay_t,D,repair_avg,repair_std])
            repair_data_D_delay_t[j*len(diff_coeffs)+k] = repair_data_row

    repair_df_D_delay_t = pd.DataFrame(data=repair_data_D_delay_t, columns=['Delay Time','D','Repair Rate','Repair Rate Std'])                 
        
    return repair_df_D_delay_t
