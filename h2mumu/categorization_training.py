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
import seaborn as sns

#============================================
# load data
#============================================

# MET,N_valid_bjets,N_valid_electrons,N_valid_extra_leptons,N_valid_extra_muons,N_valid_jets,bjet0_eta,bjet0_pt,bjet1_eta,bjet1_pt,dEta_bb,dEta_jj,dEta_jj_mumu,dPhi,dimu_pt,
# electron0_eta,electron0_pt,electron1_eta,electron1_pt,extra_muon0_eta,extra_muon0_pt,extra_muon1_eta,extra_muon1_pt,is_signal,jet0_eta,jet0_pt,jet1_eta,jet1_pt,mT_b_MET, 
# m_bb,m_jj,mu0_eta,mu0_pt,mu1_eta,mu1_pt,phi_star,weight,zep

csv = 'categorization_training.csv'

# Get the data set for algo timing info (this one has more events but unequal #'s of y values in each class) 
df = pd.read_csv('data/'+csv)
print df

df_0 = df[df.is_signal <= 0]
df_1 = df[df.is_signal > 0]

#ax = df_0[['jet0_pt','jet1_pt']].rename(columns={'jet0_pt': 'jet0_pt_bg', 'jet1_pt': 'jet1_pt_bg'}).plot.density()
#df_1[['jet0_pt','jet1_pt']].rename(columns={'jet0_pt': 'jet0_pt_signal', 'jet1_pt': 'jet1_pt_signal'}).plot.density(ax=ax)
#plt.show()

ax = df_0[['m_jj']].rename(columns={'m_jj': 'm_jj_bg'}).plot.density()
df_1[['m_jj']].rename(columns={'m_jj': 'm_jj_signal'}).plot.density(ax=ax)
plt.show()
