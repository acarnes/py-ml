import numpy as np
import random
import math

class RBM:
    """A Restricted Boltzmann Machine
    """

    def __init__(self):
        self.verbosity = 1
        self.n_hidden = 3
        self.n_visible = 4
        self.w = np.matrix([])
        self.hidden_units = np.matrix([])
        self.visible_units = np.matrix([])

    def calculate_delta_Ev(self):
        """ Calculate the energy needed to turn a visible unit from off to on. Do this for each visible unit.
            Returns a row where the ith entry is the energy needed for the ith visible unit"""
        return -1*np.dot(self.w, self.hidden_units).transpose()

    def calculate_delta_Eh(self):
        """ Calculate the energy needed to turn a hidden unit from off to on. Do this for each hidden unit.
             Returns a column vector where the ith entry is the energy needed for the ith hidden unit"""
        return -1*np.dot(self.visible_units, self.w).transpose()

    def calculate_p_up_scalar(self, delta_E):
        """ Calculate the probability of a single unit being on, given the scalar delta_E for that unit."""
        return 1/(1+math.exp(delta_E))

    def calculate_p_up(self, delta_E):
        """ Calculate the on probabilities for either the hidden or visible units. The function is given a row or column vector of delta_E values
            and returns a row or column vector of probabilities. If given a row it returns a row. If given a column vector
            it returns a column vector. The delta_E collection is a row for visible and a column for hidden."""
        vecp = np.vectorize(self.calculate_p_up_scalar)
        p_up = vecp(delta_E)
        # make sure the bias unit is always on
        p_up[0,0] = 1
        return p_up

    def scalar_flip(self, p):
        """ If the generated random number in [0,1) is less than the probability of turning the unit on
        then turn the unit on. If not turn the unit off. """
        if random.random() < p:
            return 1
        else:
            return 0

    # Make sure the visible and hidden units matrices are saved correctly. I'm not sure if the
    # assignment to vecflip(p_up) will fail when vecflip(p_up) goes out of scope.
    def flip_v(self):
        """ Flip all the visible units in the collection according to scalar_flip. 
            The current values of the hidden units determine the probability of a visible unit
            landing as on."""
        p_up = self.calculate_p_up( self.calculate_delta_Ev() )
        vecflip = np.vectorize(self.scalar_flip)
        self.visible_units = vecflip(p_up)
        #return p_up

    def flip_h(self):
        """ Flip all the hidden units in this collection according to scalar flip.
            The current values of the visible units determine the probability of a hidden unit
            landing as on. """
        p_up = self.calculate_p_up( self.calculate_delta_Eh() )
        vecflip = np.vectorize(self.scalar_flip)
        self.hidden_units = vecflip(p_up)
        #return p_up

    def initialize_w(self):
        """ Set the initial weights for the connections visible-hidden connections
        """
        self.w = np.random.rand(self.n_visible+1, self.n_hidden+1) - 0.5*np.ones((self.n_visible+1, self.n_hidden+1))
        self.w[0,0] = 0

    def load_visible_units(self, v):
        """ load a vector into the visible units """
        temp = np.append([1],v[0,:])
        self.visible_units = np.matrix(temp)

    def update_w(self, v):
        """ This is the main part of the algorithm. Update the weights in the direction of the likelihood gradient,
            so that they converge to the values that make the data most likely. This will produce a system that models
            the training data.  """
        h = np.matrix([0])
        self.load_visible_units(v)
        v = self.visible_units[:,:]
        if self.verbosity > 1:
            print ""
            print "load_v"
            print "v:             ", v 
            print "h:             ", h.transpose() 
            print "visible_units: ", self.visible_units
            print "hidden_units:  ", self.hidden_units.transpose()
            print ""
        self.flip_h()
        if self.verbosity > 1:
            print "flip_h"
            print "v:             ", v 
            print "h:             ", h.transpose()
            print "visible_units: ", self.visible_units
            print "hidden_units:  ", self.hidden_units.transpose()
            print ""
        h = self.hidden_units[:,:]
        if self.verbosity > 1:
            print "store_h"
            print "v:             ", v 
            print "h:             ", h.transpose() 
            print "visible_units: ", self.visible_units
            print "hidden_units:  ", self.hidden_units.transpose()
            print ""
        self.flip_v()
        if self.verbosity > 1:
            print "flip_v"
            print "v:             ", v 
            print "h:             ", h.transpose() 
            print "visible_units: ", self.visible_units
            print "hidden_units:  ", self.hidden_units.transpose()
            print ""
        v_new = self.visible_units[:,:]
        if self.verbosity > 1:
            print "store_v"
            print "v:             ", v 
            print "h:             ", h.transpose() 
            print "visible_units: ", self.visible_units
            print "hidden_units:  ", self.hidden_units.transpose()
            print ""
        p_h_new = self.calculate_p_up( self.calculate_delta_Eh() )
        if self.verbosity > 1:
            print "v:             ", v 
            print "h:             ", h 
            print "visible_units: ", self.visible_units
            print "hidden_units:  ", self.hidden_units
            print ""
            print "v_new:         ", v_new
            print "p_h_new:       ", p_h_new
            print ""


        delta_w = np.outer(v.transpose(), h.transpose()) - np.outer(v_new.transpose(), p_h_new.transpose()) 
        delta_sum=0.0
        for i in np.nditer(delta_w[:,1:]):
            delta_sum+=abs(i)
        self.w += 0.1*delta_w
        self.w[0,0] = 0 

        if self.verbosity > 0:
            print delta_w
            print "delta_sum = ", delta_sum

        return delta_sum

    def train(self, v_data_set):
        self.initialize_w()
        for v in v_data_set:
    #    print"w before update"
    #    print w
            delta_sum = self.update_w(v)
    #        if delta_sum < 0.001:
    #            break
    #    print"w after update"
    #    print w

    def run_to_equilibrium(self, v, n_iterations):
        if self.verbosity > 1: print "run to equilibrium "
        if self.verbosity > 1: print "input_v: ", v
        self.load_visible_units(v)
        if self.verbosity > 1: print "visible_units: ", self.visible_units
        
        for i in range(0,n_iterations):
            self.flip_h()
            self.flip_v()
        
        return self.visible_units[:,1:] 


rbm = RBM()

# Generate some data to play with
def generate_data(n_features=4, n_events=10):

    m = []
    for j in range(0,n_events):
        v = []
        for i in range(0,n_features):
            v.append(random.randint(0,1))
        m.append(v)

    return m


def calculate_error():
    pass

def test_accuracy(rbm, v_data_set):
    sum_correct=0.0
    total_sum=0.0
    for v in v_data_set:
        v_new = v[:,:]
        v_new[0,-1] = 0.5
        v_new = rbm.run_to_equilibrium(v_new, 10)
        if v_new[0,-1] == ((v[0,1] and v[0,2]) or (v[0,2] and v[0,3])):
            sum_correct+=1.0
        total_sum+=1.0

    print "total_accuracy = ", sum_correct/total_sum
    print ""

# we will have (n_visible-1) features and 1 y value
v_data_set = generate_data(n_features=rbm.n_visible-1, n_events=1000000)

# Add a y value that depends on the x values
# x,x,x,x,y in this case
for row in v_data_set:
    row.append((row[0] and row[1]) or (row[1] and row[2]))
    if row[3] != ((row[0] and row[1]) or (row[1] and row[2])):
        print "ERROR: y value assignment messed up"
        print "    ", row
        print row[3], " != (", row[0], " and ", row[1], ") or (", row[1], " and ", row[2], ") = ", (row[0] and row[1]) or (row[1] and row[2])
        print row[3] != (row[0] and row[1]) or (row[1] and row[2])
        print ""

# Turn the list list into a matrix
v_data_set = np.matrix(v_data_set)
print "visible units test data"
print v_data_set
print ""

rbm.initialize_w()
print rbm.w
test_accuracy(rbm, v_data_set[:500,:])
rbm.verbosity=0
rbm.train(v_data_set)
print rbm.w
test_accuracy(rbm, v_data_set[:500,:])
