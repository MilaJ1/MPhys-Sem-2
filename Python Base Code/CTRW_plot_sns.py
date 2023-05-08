# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_heatmaps(df, ylabel, annotate=False):
    """
    Function to plot repair and misrepair heatmaps side by side.
    """

    heading1 = df.columns[0]
    heading2 = df.columns[1]

    col_labels = np.unique(df[heading2])
    row_labels = np.unique(df[heading1])[::-1]

    repair_grid = np.zeros([len(row_labels), len(col_labels)])
    misrepair_grid = np.zeros([len(row_labels), len(col_labels)])

    for i, y in enumerate(row_labels):
        for j, Ds in enumerate(col_labels):
            temp_df = df[df[heading1]==y]
            temp_df = temp_df[temp_df[heading2]==Ds]
            repair_rate = temp_df['Repair Rate']
            misrepair_rate = temp_df['Misrepair Rate']
            repair_grid[i,j] = repair_rate
            misrepair_grid[i,j] = misrepair_rate

    col_labels = ['{:.0e}'.format(x) for x in col_labels]
    row_labels = ['{:.1f}'.format(x) for x in row_labels]

    repair_df = pd.DataFrame(data=repair_grid)
    repair_df.columns = col_labels
    repair_df.index = row_labels

    misrepair_df = pd.DataFrame(data=misrepair_grid)
    misrepair_df.columns = col_labels
    misrepair_df.index = row_labels

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4))
    ax1.set_title('Repair Fraction')
    ax2.set_title('Misrepair Fraction')
    ax1 = sns.heatmap(repair_df, ax=ax1, cmap='inferno', annot=annotate)
    ax1.set(xlabel=r'$D_s$ (nm$^2$/s)', ylabel=ylabel)
    ax2 = sns.heatmap(misrepair_df, ax=ax2, cmap='viridis', annot=annotate)
    ax2.set(xlabel=r'$D_s$ (nm$^2$/s)', ylabel=ylabel)
    plt.tight_layout()