import numpy as np

def random_positive_weight(size):
    v = np.random.rand(size) 
    return v  

def random_weight(size):
    v = np.random.rand(size)
    for x in range(size):
        b = np.random.rand(1)
        if b > 0.5:
            v[x] = v[x] * -1
    return v  

def uniform_weight(size):
    v = np.random.uniform(-1,1,size)
    
    return v    