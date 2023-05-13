import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

################################################################################
################################################################################
#Trajectory Plotting Functions
################################################################################
################################################################################

#Base CTRW and CTRW2 Functions 
################################################################################

#1D Plot
def plot_1D_ctrw(ctrw_df):
    plt.figure()
    plt.plot(ctrw_df['t'], ctrw_df['x'])

#3D CTRW2 plot - no interactions
def plot_3d_ctrw(ctrw_df):
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot3D(ctrw_df['x'], ctrw_df['y'], ctrw_df['z'], c='k')
    ax.scatter(ctrw_df['x'][0], ctrw_df['y'][0], ctrw_df['z'][0], c='g')
    ax.scatter(ctrw_df['x'].iat[-1], ctrw_df['y'].iat[-1], ctrw_df['z'].iat[-1], c='r')

#3D CTRW2 plot - with interactions
def plot_3d_ctrw_interaction(interaction_times, interaction_coords, int_count_arr, int_count, repair_arr,df1,df2):
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection ='3d')
    ax.plot3D(df1['x'], df1['y'], df1['z'])
    ax.plot3D(df2['x'], df2['y'], df2['z'])
        
    ax.set_title('Interaction count: {},   Repair: {} \n Ran for {} s'.format(int_count, repair_arr[-1], (df1['t'].values)[-1]))
    ax.set_xlabel('x (nm)')
    ax.set_ylabel('y (nm)')
    ax.set_zlabel('z (nm)')
    ax.scatter(df1['x'][0], df1['y'][0], df1['z'][0], marker='x', c='lime')
    ax.scatter(df2['x'][0], df2['y'][0], df2['z'][0], marker='x', c='lime')
    ax.scatter(df1['x'].iat[-1], df1['y'].iat[-1], df1['z'].iat[-1], marker='x', c='red')
    ax.scatter(df2['x'].iat[-1], df2['y'].iat[-1], df2['z'].iat[-1], marker='x', c='red')
    if int_count > 0:
        for i, coords in enumerate(interaction_coords):
            ax.plot3D([coords[0], coords[3]], [coords[1], coords[4]], [coords[2], coords[5]], c='k', ls='dotted')
            if repair_arr[-1]==1 and i==int_count-1:
                ax.plot3D([coords[0], coords[3]], [coords[1], coords[4]], [coords[2], coords[5]], c='m', ls='dashed')

################################################################################
#CTRW4 plotting 
################################################################################
#3D CTRW4 plot - with interactions               
def plot_3d_4ctrw_interaction(int_t, int_arr, df1, df2, df3, df4, repair, sim_clock, int_count):

        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection ='3d')
        ax.plot3D(df1['x'], df1['y'], df1['z'], c='b', label='end 1')
        ax.plot3D(df2['x'], df2['y'], df2['z'], c='cyan', ls='dotted', label='end 2')
        ax.plot3D(df3['x'], df3['y'], df3['z'], c='g', label='end 3')
        ax.plot3D(df4['x'], df4['y'], df4['z'], c='lime', ls='dotted', label='end 4')
        ax.set_title('Repair: {} \n Ran for {} s \n Interaction count: {}'.format(repair, sim_clock, int_count))
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
        ax.set_zlabel('z (nm)')
        ax.scatter(df1['x'].iat[-1], df1['y'].iat[-1], df1['z'].iat[-1], marker='x', c='red', label='finish')
        ax.scatter(df2['x'].iat[-1], df2['y'].iat[-1], df2['z'].iat[-1], marker='x', c='red')
        ax.scatter(df3['x'].iat[-1], df3['y'].iat[-1], df3['z'].iat[-1], marker='x', c='red')
        ax.scatter(df4['x'].iat[-1], df4['y'].iat[-1], df4['z'].iat[-1], marker='x', c='red')
        ax.scatter(df1['x'][0], df1['y'][0], df1['z'][0], marker='x', c='black', label='start')
        ax.scatter(df2['x'][0], df2['y'][0], df2['z'][0], marker='x', c='black')
        ax.scatter(df3['x'][0], df3['y'][0], df3['z'][0], marker='x', c='black')
        ax.scatter(df4['x'][0], df4['y'][0], df4['z'][0], marker='x', c='black')
        ax.legend()


################################################################################
################################################################################
# MC data plotting functions
################################################################################
################################################################################

#CTRW2 MC data
################################################################################

#Plot repair for different values of repair and Diffusion Coefficient
#Plot graphs separately on different subplots
def plot_r_and_D_subplots(repair_df_D_r,reps,r_vals):
    
    fig,axes = plt.subplots(2,3,figsize=(10,8))
    plt.suptitle('Repair Rate vs Diffusivity Coefficient For Varying Initial Separations')
    for i,r in enumerate(r_vals):

        data = repair_df_D_r[repair_df_D_r['r']==r]

        axes.flat[i].errorbar(x=data['D'],y=data['Repair Rate']/reps,yerr=data['Repair Rate Std']/reps)
        axes.flat[i].set_xscale('log')
        axes.flat[i].set_xlabel('Diffusivity Coefficient')
        axes.flat[i].set_ylabel('Repair Rate')
        
        axes.flat[i].set_title('Initial Separation: {} nm'.format(r))

#Plot repair for different values of repair and Diffusion Coefficient
#Plot graphs on single plot 
def plot_r_and_D(repair_df_D_r,reps,r_vals):
    
    fig,ax = plt.subplots(figsize=(10,8))
    plt.suptitle('Repair Rate vs Diffusivity Coefficient For Varying Initial Separations')
    for i,r in enumerate(r_vals):

        data = repair_df_D_r[repair_df_D_r['r']==r]
        #area = scipy.integrate.simpson(data['Repair Rate']/reps, x=data['D'], axis=-1, even='avg')
        ax.errorbar(x=data['D'],y=data['Repair Rate']/reps,
                    yerr=data['Repair Rate Std']/reps,label='Initial Separation: {} nm, Area Under Curve: {}'.format(r,'area'))
        ax.set_xscale('log')
        ax.set_xlabel('Diffusivity Coefficient')
        ax.set_ylabel('Repair Rate')
        ax.legend()

def plot_delay_t_and_D_subplots(repair_df_D_delay_t,reps,dim_1=2,dim_2=3):
    
    delay_ts = np.unique((repair_df_D_delay_t['Delay Time'].to_numpy()))

    fig,axes = plt.subplots(dim_1,dim_2,figsize=(10,10))
    plt.suptitle('Repair Rate vs Diffusivity Coefficient For Varying Initial Delay Times')
    for i,delay_t in enumerate(delay_ts):

        data = repair_df_D_delay_t[repair_df_D_delay_t['Delay Time']==delay_t]

        axes.flat[i].errorbar(x=data['D'],y=data['Repair Rate'],yerr=data['Repair Rate Std']/np.sqrt(reps),fmt='.',ls='none')
        axes.flat[i].set_xscale('log')
        axes.flat[i].set_xlabel('Diffusivity Coefficient')
        axes.flat[i].set_ylabel('Repair Rate')
        
        axes.flat[i].set_title('Delay Time: {0:.2f} s'.format(delay_t))
        
    fig.tight_layout()
     

###############################################################################
#CTRW4 MC data
################################################################################

def plot_De_and_Ds(repair_df_D_r,reps=1,skip=1):
    
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,8))
    plt.suptitle('Repair and Misrepair Rate vs Break Site Diffusivity Coefficient For Varying Break End Diffusivity')
    
    De_vals = np.unique((repair_df_D_r['D_ends'].to_numpy()))
    print(De_vals)
    
    for i,De in enumerate(De_vals[::skip]):

        data = repair_df_D_r[repair_df_D_r['D_ends']==De]

        #area = scipy.integrate.simpson(data['Repair Rate']/reps, x=data['D'], axis=-1, even='avg')
        #ax.errorbar(x=data['D'],y=data['Repair Rate'],
         #           yerr=data['Repair Rate Std']/reps,label='Initial Separation: {} nm, Area Under Curve: {}'.format(r,'area'))
        ax1.errorbar(x=data['D_sites'],y=data['Repair Rate'],yerr = data['Repair Rate Std']/np.sqrt(reps), label='D_end = {}'.format(De))
        ax1.set_xscale('log')
        ax1.set_xlabel('Break Site Diffusivity Coefficient')
        ax1.set_ylabel('Repair Rate')
        ax1.set_title('Repair Rate vs Site Diffusion Coefficient')
        
        ax2.errorbar(x=data['D_sites'],y=data['Misrepair Rate'],yerr=data['Misrepair Rate Std']/np.sqrt(reps),label='D_end = {:.2e}'.format(De))
        ax2.set_xscale('log')
        ax2.set_xlabel('Break Site Diffusivity Coefficient')
        ax2.set_ylabel('Misrepair Rate')
        ax2.set_title('Misrepair Rate vs Site Diffusion Coefficient')

        fig.tight_layout()
        
        plt.legend()

def plot_De_and_Ds_subplots(repair_df_De_Ds,plot_dim_1=2,plot_dim_2=3,reps=1):
    
    fig,axes = plt.subplots(plot_dim_1,plot_dim_2,figsize=(10,8))
    plt.suptitle('Repair and Misrepair Rate vs Break Site Diffusivity Coefficient For Varying Break End Diffusivities')
    
    De_vals = np.unique((repair_df_De_Ds['D_ends'].to_numpy()))
    
    for i,De in enumerate(De_vals):

        data = repair_df_De_Ds[repair_df_De_Ds['D_ends']==De]

        axes.flat[i].errorbar(x=data['D_sites'],y=data['Repair Rate'],yerr=data['Repair Rate Std']/np.sqrt(reps),color ='k',
                              label='Repair Rate')
        axes.flat[i].errorbar(x=data['D_sites'],y=data['Misrepair Rate'],yerr=data['Misrepair Rate Std']/np.sqrt(reps),
                          color='r',linestyle='dashed',label='Misepair Rate')
        
        axes.flat[i].set_xscale('log')
        
        if i == 0 or i==plot_dim_2:
            axes.flat[i].set_ylabel('Repair Rate')
        
        axes.flat[i].set_xlabel('Break Site Diffusivity Coefficient')
        axes.flat[i].set_title('D_ends: {:.2e} '.format(De))
        
        plt.legend()

        fig.tight_layout()

def heatmap_repair_misrepair_subplots(repair_misrepair_df,N_de,N_ds,misrep_multiplier=1,
                                    extent_list=[2.8e13,28e13,28e13,2.8e13]):
    
    #extent: [leftmost value,rightmost value, bottom-most value, top-most value]
    arr_repair = np.array(repair_misrepair_df['Repair Rate'].values)
    arr_repair = np.reshape(arr_repair,(N_de,N_ds))

    arr_misrepair = np.array(repair_misrepair_df['Misrepair Rate'].values)
    arr_misrepair = np.reshape(arr_misrepair,(N_de,N_ds))

    arr_repair = np.flip(arr_repair,axis=0)
    arr_misrepair = np.flip(arr_misrepair,axis=0)

    repair_misrepair_arr = np.array([arr_repair,arr_misrepair*misrep_multiplier])

    De_vals = np.unique((repair_misrepair_df['D_ends'].to_numpy()))
    Ds_vals = np.unique((repair_misrepair_df['D_sites'].to_numpy()))

    titles = np.array(['Repair Rate','Misrepair Rate X 10'])
    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10,8))
    #plt.ticklabel_format(style='plain')
    fig.suptitle('Heatmaps for Repair and Misrepair Rates, Varying Break Site and Break End Diffusivities')
    for i,ax in enumerate(axes.flat):
        im = ax.imshow(repair_misrepair_arr[i], cmap='hot', vmin=0,vmax=1,
                        extent=extent_list)
        ax.set_title('{}'.format(titles[i]))
        ax.set_xlabel('Break Site Diffusion Coefficient')
        ax.plot()
        
        if i == 0:
            ax.set_ylabel('Break End Diffusion Coefficient')

        ax.set(xticks=np.linspace(extent_list[0], extent_list[1], N_de), xticklabels=Ds_vals,
                yticks=np.linspace(extent_list[2], extent_list[3], N_ds), yticklabels=De_vals)
        #ax.ticklabel_format(style='sci',scilimits=(10,17),axis='both')

    fig.colorbar(im, ax=axes.ravel().tolist(),orientation='horizontal',label='Rate')

    plt.show()

    