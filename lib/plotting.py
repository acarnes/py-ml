# Method to calculate mutual information between variables
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import offsetbox
from time import time


############################################################
# Plot 2d projections of Feature Reduction Transformations
############################################################
def plot_embedding(X,y):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
