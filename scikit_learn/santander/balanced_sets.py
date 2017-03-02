##############################################
# balanced_sets.py                           #
##############################################
# make some plots, run PCA, etc              #
# get to know the data see how using         #
# sets changes things.                       #
##############################################

#============================================
# import
#============================================

import sys
sys.path.insert(0, '../lib')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import histograms as histos
from mi import mutual_information
from sklearn import decomposition
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score

# ===========================================
# Load Data
# ===========================================

df = pd.read_csv('data/train.csv', index_col=0)

# ===========================================
# Clean Data
# ===========================================

# Remove variables with zero standard deviation
#print "\n#### Features with zero variance...\n"
#print(df.loc[:, df.std() == 0].std())
df = df.loc[:, df.std() != 0]

# Extract the two types of target into separate data sets so that we can make balanced sets
df0 = df.loc[df['TARGET'] == 0]
df1 = df.loc[df['TARGET'] == 1]

# the number of entries in each data set
n0 = df0.shape[0]
n1 = df1.shape[0]

nsets = n0/n1
print nsets

frames0 = []
for i in xrange(0,nsets):
    frames0.append(df0.iloc[i*n1:(i+1)*n1])

print len(frames0)

for i in xrange(0,len(frames0)):
    print frames0[i].shape

#print "\n#### Description of the different features...\n"
#with pd.option_context('display.max_rows', 999, 'display.max_columns', 10):
#    print(df.describe().transpose())

#print "#### Dataframe Unique Value Counts...\n"
#with pd.option_context('display.max_rows', 999, 'display.max_columns', 3):
#    print(df.apply(pd.Series.nunique))

# concatenate equal numbers of target == 0 and target == 1, vertically
dfbalanced = pd.concat([df1, frames0[1]], axis=0)

## ===========================================
## PCA
## ===========================================
#
pca = decomposition.PCA(n_components=100)
xbalanced = dfbalanced.ix[:, dfbalanced.columns != 'TARGET']
xall = df.ix[:, df.columns != 'TARGET']
xbalanced = pd.DataFrame(scale(xbalanced))
xall = pd.DataFrame(scale(xall))

#with pd.option_context('display.max_rows', 999, 'display.max_columns', 3):
#    print(x.std())
#    print(x.mean())

yall = df['TARGET']
ybalanced = dfbalanced['TARGET']
pca.fit(xbalanced)

# Check out the eigenvectors and eigenvalues for PCA
print "\n#### PCA Eigenvalues...\n"
print(pca.explained_variance_ratio_)  #Eigenvalues
print "\n#### PCA Variance Cumsum...\n"
print(np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)) # see how many pca vectors we need to explain all of the variance (looks like 13)
print "\n#### PCA Eigenvectors...\n"
eigenvectors = pca.components_        #Eigenvectors are the rows
print(eigenvectors)                   
print(eigenvectors.shape)

#print "\n#### x before PCA transformation...\n"
#print(x.ix[:,:3])
dfallpca = pd.DataFrame(pca.transform(xall))
dfbalancedpca = pd.DataFrame(pca.transform(xbalanced))
#print "\n#### x after PCA transformation...\n"
dfallpca['TARGET'] = pd.Series(yall.values)
dfbalancedpca['TARGET'] = pd.Series(ybalanced.values)
#print(dfpca.ix[:,:3])
#print(dfpca['TARGET'])

# choose the working data frame
dfw = dfbalancedpca

# ===========================================
# Mutual Information
# ===========================================

mi = mutual_information(dfw, dfw['TARGET'], bins=[25,25])
mi[:15].plot(kind='barh', title='Mutual Information')
plt.show()

#print "\n#### Top mutual information vars...\n"
#print(mi[:15])
#print "\n#### Bottom mutual information vars...\n"
#print(mi[-15:])
#print "\n"

# Plot the top/bottom MI variables if we want
#histos.histogram_all_vars(dftopv)
#histos.p_y_given_x_all_vars(dftopv, dftopv['TARGET'])
#histos.p_x_given_y_all_vars(dftopv, dftopv['TARGET'])
#histos.p_y_given_x_all_vars(dflastv, dftopv['TARGET'])
#histos.p_x_given_y_all_vars(dflastv, dftopv['TARGET'])

#plt.show()

# ===========================================
# Select Vars
# ===========================================

dfw = dfallpca

# get a dataframe of the top N mutual info variables 
dftopv = pd.DataFrame()
dftopv = dfw[list(mi[:50].index)]

# get a dataframe of the bottom N mutual info variables 
dflastv = pd.DataFrame()
dflastv = dfw[list(mi[-15:].index)]

x = dftopv.ix[:, dftopv.columns != 'TARGET']
x = pd.DataFrame(scale(x))

#print "\n#### Std Deviation and Mean after PCA transform...\n"
#print(x.std())
#print(x.mean())

y = dftopv['TARGET']

# ===========================================
# Run Classification
# ===========================================

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size = 0.2, random_state=0)
weights = {0:0.04, 1:0.96}
lr = LogisticRegression(C=1.0, class_weight=weights)
#lr = LogisticRegression(C=1.0)
#
lr.fit(x_train,y_train)
#
y_true, y_scores = y_test, lr.predict_proba(x_test)
score = roc_auc_score(y_test, y_scores[:,1])
#
print "\n#### ROC AUC Score...\n"
print score


#df.ix[:5,0:10] # row begin:row end, column begin:column end
#num_unique_values = col.value_counts().size
#df = df.reindex(np.random.permutation(df.index))

