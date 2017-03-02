# =====================================
# Restricted Boltzmann Machine
# =====================================
# visible units are grouped into a row
# hidden units are grouped into a vector
# E(v,h;w) = w_ij*v_i*h_j = v*w*h
# P(v,h) = exp(-E(v,h))/SUM(-E(v,h))
# delta_E(h_j = 1) = -w_ij*v_i
# delta_E(v_i = 1) = -w_ij*h_j
# P(x = 1) = 1/(1+exp(delta_E(x = 1)))
# =====================================


# This is a machine learning algorithm based upon Statistical Mechanics.
# We use the boltzmann distribution to describe the probability of
# a microstate with a specific energy, the higher the energy the less probable.
# The RBM is basically an ising model. Interactions between sites have different
# interaction energies. We want to set the interaction energies to make certain
# microstates more likely. If we are trying to represent the data we want to 
# make the microstates representing the data to be the most likely. We update the
# wij values to make this so. Namely, we want the system to tend towards
# data points in thermal equilibrium. Moreover, if we start with a data point
# and let the system run to thermal equilibrium the system should not run away 
# from the data point. If the system runs away, this implies that the data point 
# is not more likely than the surrounding states. So we put a data point into
# the system and look to see if it is stable as the system tends towards equilibrium.


import numpy as np
import random
import math

n_hidden = 3
n_visible = 4

# Generate some data to play with
def generate_data(n_features=4, n_events=10):

    m = []
    for j in range(0,n_events): 
        # make sure the visible units have a 1 in the first location
        # sGo that the hidden units connect to a bias
        v = [1]
        for i in range(0,n_features):
            v.append(random.randint(0,1))
        m.append(v)

    return m

# we will have (n_visible-1) features and 1 y value
v_data_set = generate_data(n_features=n_visible-1, n_events=100000)

# Add a y value that depends on the x values
# x,x,x,x,y in this case
for row in v_data_set:
    row.append((row[1] and row[2]) or (row[2] and row[3]))
    if row[4] != ((row[1] and row[2]) or (row[2] and row[3])): 
        print "ERROR: y value assignment messed up"
        print "    ", row
        print row[4], " != (", row[1], " and ", row[2], ") or (", row[2], " and ", row[3], ") = ", (row[1] and row[2]) or (row[2] and row[3]) 
        print row[4] != (row[1] and row[2]) or (row[2] and row[3])
        print ""

# Turn the list list into a matrix
v_data_set = np.matrix(v_data_set)
print "visible units test data"
print v_data_set
print ""


# Initialize the w values
# make sure to make into an appropriate matrix
def initialize_w(n_visible, n_hidden):
    w = np.random.rand(n_visible+1, n_hidden+1) - 0.5*np.ones((n_visible+1, n_hidden+1))
#    w = np.ones((n_visible+1, n_hidden+1))
    w=np.matrix(w)
    # set the 0,0 location to zero so that the bias units don't interact
    w[0,0]=0
    return w


# Initialize the h values
# make sure to make into an appropriate matrix/row/vector
def random_h(n_hidden):
    h = [1]
    for i in range(0, n_hidden):
        # make sure the hidden units have a 1 in the first location
        # so that the visible units connect to a bias
        h.append(random.randint(0,1)) 
    # h is a vector
    h = np.matrix(h).transpose()
    return h

# assumes h is a column, v is a row
# the 0th entry in this defininition of delta_E deals with the bias unit
# which is always on, so defining a delta_E for the on vs off state
# doesn't make sense. Make sure not to use the first delta_E entry because of this.
# It is simply convenient to have this calculated along with the other meaningful
# delta_E values.
# take the transpose so that the energy collections are the same shape as the
# vectors or columns they came from
def calculate_delta_Ev(w, h):
   return -1*np.dot(w,h).transpose()

def calculate_delta_Eh(w, v):
   return -1*np.dot(v,w).transpose()

def p(x):
    return 1/(1+math.exp(x))

def p_up(delta_E):
    vecp = np.vectorize(p)
    p_up_ = vecp(delta_E)
    # impose the constraint that the bias unit is always on
    p_up_[0,0] = 1
    return p_up_


# Given the probability p of being on vs being off
# Determine whether the value should be on or off
def flip_unit(p):
    if random.random() < p:
        return 1
    else:
        return 0

def flip_v(w,h):
    p = p_up(calculate_delta_Ev(w,h))
    flip_units = np.vectorize(flip_unit)
    return flip_units(p)
    

def flip_h(w,v):
    p = p_up(calculate_delta_Eh(w,v))
    flip_units = np.vectorize(flip_unit)
    return flip_units(p)


def update_w(w,v):
    v_new = v[:,:]
    h = flip_h(w,v)
    v_new = flip_v(w,h)
    p_h_new = p_up(calculate_delta_Eh(w,v_new))
    delta_w = np.outer(v.transpose(), h.transpose()) - np.outer(v_new.transpose(), p_h_new.transpose()) 
    delta_sum=0.0
    for i in np.nditer(delta_w[:,1:]):
        delta_sum+=abs(i)
    w += 0.1*delta_w
    w[0,0] = 0
    return delta_sum


def test_accuracy(w,v_data_set):
    sum_correct=0.0
    total_sum=0.0
    for v in v_data_set:
        v_new = v[:,:]
        v_new[0,-1] = 0.5
        for i in range(0,10):
            h = flip_h(w,v_new)
            v_new = flip_v(w,h)
        if v_new[0,-1] == ((v[0,1] and v[0,2]) or (v[0,2] and v[0,3])):
            sum_correct+=1.0
        total_sum+=1.0

    print "total_accuracy = ", sum_correct/total_sum
    print ""

print "w before update"
w = initialize_w(n_visible, n_hidden)
print w

test_accuracy(w, v_data_set[0:500,:])

for v in v_data_set:
#    print"w before update"
#    print w
    delta_sum = update_w(w,v)
#    if delta_sum < 0.001:
#        break
#    print"w after update"
#    print w

print "w after update"
print w

test_accuracy(w, v_data_set[0:500,:])
