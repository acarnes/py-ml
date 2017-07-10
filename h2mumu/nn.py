##############################################
# nn.py                                      #
##############################################
# neural net for h2mu sig/bkg discrimination #
#                                            #
##############################################

# All vars w/ normalization (div max-min) gives 66% accuracy with 0.734 roc auc
# w/ counts but no bjetsN, jetsN gives 64.5% with 0.711 roc auc
# w/ all vars and weights gives 66% accuracy with 0.734 roc auc  
# w/ all vars and div std rather than max-min 66% accuracy with 0.74 roc auc
# w/ all vars and weights and -999 -> 0 66% accuracy with 0.744 roc auc
# w dropout at 0.1 -- 67.8%, 0.757
# 3x100 nodes dropout at 0.1 -- 68%, 0.76
# 300,200,100,25 nodes dropout at 0.1 -- 68.6%, 0.768
# 300,300,300,100 nodes no dropout -- 70%, 0.785
# test on sig vs dy, ttbar only
# try other architectures
    # try batch norm
# try multiclass

#------------------------------------------------------------------
#------------------------------------------------------------------

# Should train only on dy, ttbar, sig
    # waste of resources and time and probably worse results otherwise
# Try weighting the events 
# Reduce dropout to increase complexity
# Counts without jetsN, bjetsN (these have -999 values)... are they helping even?
# Try to replace -999 values with 0 and train
    # Leave out counts
    # Leave counts in
# Remove large jetsN features
# Try batch norm
# Can try different architectures
# Can also try multiclass


#------------------------------------------------------------------
#------------------------------------------------------------------

#============================================
# import
#============================================

import sys 
sys.path.insert(0, '../lib')

# Basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Get ROC AUC
def roc_auc(y, y_score):
    fpr, tpr, _ = roc_curve(y, y_score)
    roc_auc = auc(fpr, tpr)
    return roc_auc

# Plot the ROC curve
def plot_roc(y, y_score):
    fpr, tpr, _ = roc_curve(y, y_score)
    rocauc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % rocauc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Baseline Comparison: ROC for BDT Score')
    plt.legend(loc="lower right")
    plt.show()
    return roc_auc

# Read in data for training
df = pd.read_csv('data/all_sig_and_bkg_in_110_to_160_GeV.csv')
#df = pd.read_csv('data/sig_and_dy-ttbar-bkg_in_110_to_160_GeV.csv')
df = df[df.bin >= 0] # only look at those in signal region
df[df == -999] = 0   # replace -999 values with 0
print df.shape

# Get sig/bkg dataframes
bkgdf = df[df.is_signal <= 0]
sigdf = df[df.is_signal > 0]
print bkgdf.describe()

#for col in bkgdf.columns:
#    print col

# organize sets of variables
vars_mu = ['dimu_pt', 'dimu_eta', 'dimu_abs_dEta', 'dimu_abs_dPhi']
vars_bdt = ['bdt_score']
vars_met = ['MET']
vars_bjets1 = ['bjet0_eta', 'bjet0_pt']
vars_bjets2 = ['bjet1_eta', 'bjet1_pt', 'dEta_bb', 'mT_b_MET']
vars_jet_counts = ['nBMed', 'nJets', 'nJetsFwd', 'nJetsCent']
vars_jets1 = ['jet1_eta', 'jet1_pt']
vars_jets2 = ['jet2_eta', 'jet2_pt' ,'dEta_jj_mumu', 'dPhi_jj_mumu', 'dijet1_abs_dEta', 'dijet1_mass']
vars_jets3 = ['jet3_eta', 'jet3_pt', 'dijet2_abs_dEta', 'dijet2_mass']
vars_jets4 = ['jet4_eta', 'jet4_pt', 'dijet3_abs_dEta', 'dijet3_mass', 'dijet4_abs_dEta', 'dijet4_mass']
y = ['is_signal']

# plot the ROC score for the BDT 
# this acts as the baseline comparison
#plot_roc(df['is_signal'], df['bdt_score'])
print "bdt roc_auc: %f" % roc_auc(df['is_signal'], df['bdt_score'])

# Add the variables we want to the set of training features
features = vars_mu
features.extend(vars_met)
features.extend(vars_jet_counts)
features.extend(vars_jets1)
features.extend(vars_jets2)
features.extend(vars_jets3)
features.extend(vars_jets4)
features.extend(vars_bjets1)
features.extend(vars_bjets2)

# normalize the dataframe
#df_norm = (df - df.mean()) / (df.max() - df.min())
df_norm = (df - df.mean()) / df.std()

# Training features and target
X = df_norm[features].values
Y = df[y].values

# Neural Net architecture
model = Sequential()
model.add(Dense(300, input_dim=len(features), activation='relu'))
#model.add(Dropout(0.1))
model.add(Dense(300, activation='relu'))
#model.add(Dropout(0.1))
model.add(Dense(300, activation='relu'))
#model.add(Dropout(0.1))
model.add(Dense(100, activation='relu'))
#model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

for i in range(20):
    print "========== Training interation %d ===================\n" % i
    #model.fit(X, Y, epochs=5, batch_size=100, class_weight = {0: 1.0, 1: bkgdf.shape[0]/sigdf.shape[0]})
    model.fit(X, Y, epochs=5, batch_size=100)
    scores = model.evaluate(X, Y)
    Y_nn = model.predict(X)
    rocauc = roc_auc(Y, Y_nn)
    
    print Y_nn

    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    print "roc_auc: %f" % rocauc
    print ""
