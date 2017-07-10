##############################################
# samples.py                                 #
##############################################
# grab some samples                          #
# concatenate them into one .csv             #
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

#smap = { 'GGF'    : 'H2Mu_gg_bdt_training.csv',
#         'VBF'    : 'H2Mu_VBF_bdt_training.csv',
#         'WHn'    : 'H2Mu_WH_neg_bdt_training.csv',
#         'WHp'    : 'H2Mu_WH_pos_bdt_training.csv',
#         'ZH'     : 'H2Mu_ZH_bdt_training.csv',
#         'ttll'   : 'tt_ll_AMC_bdt_training.csv',
#         'ttW'    : 'ttW_bdt_training.csv',
#         'ttZ'    : 'ttZ_bdt_training.csv',
#         'tWn'    : 'tW_neg_bdt_training.csv',
#         'tWp'    : 'tW_pos_bdt_training.csv',
#         'tZq'    : 'tZq_bdt_training.csv',
#         'WW'     : 'WW_bdt_training.csv',
#         'WZ'     : 'WZ_3l_bdt_training.csv',
#         'DY'     : 'ZJets_AMC_bdt_training.csv',
#         'ZZ2l2q' : 'ZZ_2l_2q_bdt_training.csv',
#         'ZZ2l2u' : 'ZZ_2l_2v_bdt_training.csv',
#         'ZZ4l'   : 'ZZTo4L_bdt_training.csv' }

smap = {
         'GGF'    : 'H2Mu_VBF_bdt_training_dyAMC-J.csv',
         'VBF'    : 'H2Mu_gg_bdt_training_dyAMC-J.csv',
         'ZH'     : 'H2Mu_ZH_bdt_training_dyAMC-J.csv',
         'WHp'    : 'H2Mu_WH_pos_bdt_training_dyAMC-J.csv',
         'WHn'    : 'H2Mu_WH_neg_bdt_training_dyAMC-J.csv',
         'DY0J'   : 'ZJets_AMC_0j_bdt_training_dyAMC-J.csv',
         'DY1J'   : 'ZJets_AMC_1j_bdt_training_dyAMC-J.csv',
         'DY2J'   : 'ZJets_AMC_2j_bdt_training_dyAMC-J.csv',
         'tt'     : 'tt_ll_AMC_bdt_training_dyAMC-J.csv'
#         'ttW'    : 'ttW_bdt_training_dyAMC-J.csv',
#         'ttZ'    : 'ttZ_bdt_training_dyAMC-J.csv',
#         'tWp'    : 'tW_pos_bdt_training_dyAMC-J.csv',
#         'tWn'    : 'tW_neg_bdt_training_dyAMC-J.csv',
#         'tZq'    : 'tZq_bdt_training_dyAMC-J.csv',
#         'ZZ2l2v' : 'ZZ_2l_2v_bdt_training_dyAMC-J.csv',
#         'ZZ2l2q' : 'ZZ_2l_2q_bdt_training_dyAMC-J.csv',
#         'ZZ4l'   : 'ZZ_4l_bdt_training_dyAMC-J.csv',
#         'WZ2l'   : 'WZ_2l_bdt_training_dyAMC-J.csv',
#         'WZ3l'   : 'WZ_3l_bdt_training_dyAMC-J.csv',
#         'WW'     : 'WW_bdt_training_dyAMC-J.csv'
}
#         'RunB'   : 'RunB_bdt_training_dyAMC-J.csv'
#         'RunC'   : 'RunC_bdt_training_dyAMC-J.csv'
#         'RunD'   : 'RunD_bdt_training_dyAMC-J.csv'
#         'RunE'   : 'RunE_bdt_training_dyAMC-J.csv'
#         'RunF1'  : 'RunF_1_bdt_training_dyAMC-J.csv'
#         'RunF2'  : 'RunF_2_bdt_training_dyAMC-J.csv'
#         'RunG'   : 'RunG_bdt_training_dyAMC-J.csv'
#         'RunH'   : 'RunH_bdt_training_dyAMC-J.csv'


df_all = pd.DataFrame()

# Get the data set for algo timing info (this one has more events but unequal #'s of y values in each class) 
for key,val in smap.iteritems():
    print "%s,%s" % (key,val)
    df = pd.read_csv('data/indiv/'+val)
    df_all = df_all.append(df)

print df_all.describe()
df_all.to_csv('data/sig_and_dy-ttbar-bkg_in_110_to_160_GeV.csv')
