# Template to run a simulation study for inventory management models.
# Zero Lead Time
#   tbd. 
import scipy.stats as sp
import numpy as np
import itertools
import multiechelonpls.des.discreteeventsystem as des

# AdBS method template
def adbs_evaluation(wh_range, store_range):
    # For each store length
    pass


# LC method template
def lc_evaluation():
    pass

# CBS method template
def cbs_evaluation(wh_range, store_range):
    scores = {}
    # Iterate over all possible combinations
    combinations = list(itertools.product([i for i in wh_range], [j for j in store_range]))
    
    # Run through combinations on the DES
    scores[combinations] = np.random.rand() # random numver for now just illustration
    # Return best
    return min(scores, key=scores.get)


if __name__ == '__main__':
    # Run each method on the des
    method = cbs_evaluation
    print(method(range(15), range(4,10)))



