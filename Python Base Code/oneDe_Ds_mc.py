# -*- coding: utf-8 -*-
import numpy as np

### Import our Own CTRW Code 
import CTRW_base as CTRW
import CTRW4
import CTRW4_mc

alpha = 0.5
#D_ends = 2.8e14  # nm^2/s
#D_sites = 2.8e15  #nm^2/s
jump_t = 1e-12  # jump time in s
min_t = 0.001  # minimum waiting time in s
run_t = 86400  # simulation run time in s
delay_t = 2.5  # initial delay time in s
int_length = 25 #nm
separation = 500 #nm
int_p = 0.6

D_site_vals = 2.8*np.logspace(10,17,num=8)

D_end = 2.8e15   # value to change

nreps = 5

CTRW4_mc.ctrw4_oneDe_Ds_mc(1, nreps, [(0,0,0), (0,0,separation)], D_site_vals,
                           D_end, jump_t, run_t, min_t, alpha, int_length,
                           delay_t, int_p)
