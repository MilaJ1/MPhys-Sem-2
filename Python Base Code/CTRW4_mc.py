import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import CTRW_base as CTRW
import CTRW4 

###########################################################
# Supporting Functions
###########################################################

def print_full(df):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    df
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')
    
############################################################
# Monte Carlo Functions to generate data in 4 CTRW framework
############################################################

#Basic CTRW4 mc function, collects relevant data for runs of CTRW4 simulation
def ctrw4_interaction_mc(nsamples, initial_pos, D_sites,D_ends, diff_time, run_time, min_wait_time, anom_diff_exp,
                         int_length, delay_time, interaction_p):
    
    mc_repair_arr = np.zeros((nsamples,12))
    
    for i in np.arange(nsamples):

        rep_t, rep_arr, int_count = CTRW4.coupled_ctrw_for_mc(initial_pos, D_sites, D_ends, diff_time, run_time, min_wait_time,
                                                              anom_diff_exp, int_length, delay_time, interaction_p)
    
        mc_repair_arr[i][0] = i
        mc_repair_arr[i][1] = D_sites
        mc_repair_arr[i][2] = D_ends
        mc_repair_arr[i][3] = int_count
        mc_repair_arr[i][4:-1] = rep_arr
        mc_repair_arr[i][-1] = rep_t
        
    repair_df = pd.DataFrame(data=mc_repair_arr, columns=['mc_step','D_sites','D_ends','int_count',
                                                          'i12','i13','i14','i23','i24','i34','repair','repair_t'])
    return repair_df

# MC function to vary the Diffusion Coefficient of the break ends and break sites
def ctrw4_De_Ds_mc(samples_per_rep,repeats, initial_pos, D_sites,D_ends, diff_time, 
                   run_time, min_wait_time, anom_diff_exp,int_length, delay_time,
                   interaction_p):
    
    nrows = len(D_sites)*len(D_ends)
    repair_data_De_Ds = np.zeros((nrows,6))
    print(nrows)
    
    for j,De in enumerate(D_ends):
        
        print('----------------')
             
        repair_avgs = np.empty(len(D_sites))
        repair_stds = np.empty(len(D_sites))
    
        for k,Ds in enumerate(D_sites):

            print(j*len(D_sites)+k)
            print(De,Ds)

            repair_rates = np.zeros(repeats)
            misrepair_rates = np.zeros(repeats)
            for i in np.arange(repeats):
                data = ctrw4_interaction_mc(samples_per_rep, initial_pos, Ds,De, 
                                            diff_time, run_time, min_wait_time, anom_diff_exp,
                                            int_length, delay_time, interaction_p)
                
                repair_events = len(data[data['repair']==1.0])
                misrepair_events = len(data[data['repair']==-1.0])
                repair_rates[i] = repair_events
                misrepair_rates[i] = misrepair_events

            repair_avg = np.nanmean(repair_rates)
            misrepair_avg = np.nanmean(misrepair_rates)
            repair_std = np.nanstd(repair_rates)
            misrepair_std = np.nanstd(misrepair_rates)
            
            repair_data_row = np.array([De,Ds,repair_avg,repair_std,misrepair_avg,misrepair_std])
            repair_data_De_Ds[j*len(D_sites)+k] = repair_data_row
            
    repair_df_De_Ds = pd.DataFrame(data=repair_data_De_Ds, 
                                   columns=['D_ends','D_sites','Repair Rate',
                                            'Repair Rate Std','Misrepair Rate',
                                            'Misrepair Rate Std']) 
    
    return repair_df_De_Ds
