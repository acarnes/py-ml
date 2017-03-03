#############################################
# train.py                                   #
##############################################
# train some classifiers, then optimize them #
#                                            #
#                                            #
##############################################

#============================================
# import
#============================================

import sys 
sys.path.insert(0, '../lib')

# Basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import table

#============================================
# load data
#============================================

csvs = ['counts_run1_all', 'counts_run1_final', 'counts_run1_geo', 'counts_run1_w_intermediate']

for csv in csvs:
    # Get the data set for algo timing info (this one has more events but unequal #'s of y values in each class) 
    df = pd.read_csv('data/'+csv+'.csv', index_col=0)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
#    print df
    
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(1,1,1) # no visible frame
    #ax.xaxis.set_visible(False)  # hide the x axis
    #ax.yaxis.set_visible(False)  # hide the y axis
    ax.axis('off')
    
    t = table(ax, df, loc='center')  # where df is your data frame
    t.set_fontsize(12)
    t.scale(1, 1.5)
    
    plt.savefig(csv+'.png')
