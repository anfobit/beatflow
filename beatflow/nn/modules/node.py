import numpy as np
from .module import Module
from .. import function as F

class Node(Module):
    _positive = True
    _in_size = 0
    _out_size = 0
    _out_dim = 1

    _out_spike_pos = 0
    def spike_delay(self):
        self._out_spike_pos = min(self._out_spike_pos + 1, self._out_dim-1)
    def spike_speed_up(self):
        self._out_spike_pos = min(self._out_spike_pos - 1, self._out_dim-1) 

    _threshold = 1
    _potential = 0

    _weight = []
    _enable = []

    _rest_state = 0

    _leak_rate = 0

    _memory = []

    def __init__(self,
                 in_size,
                 out_size,
                 threshold = None,
                 weight=[],
                 out_dim = 1,
                 positive=True,
                 leak_rate=0.1):
        super(Node, self).__init__()
        self._positive = positive
        self._memory = np.zeros((self.max_dim,in_size))
        self._in_size = in_size
        self._out_size = out_size
        self._out_dim = out_dim
        if weight == []:
            self._weight = F.random_positive_weight(in_size)
        else:
            self._weight = weight

        self._enable = np.ones(in_size)
        if threshold == None:
            self._threshold = in_size 
        else:    
            self._threshold = threshold

        self._leak_rate = leak_rate
    #SHAPE -> DIM, IN_SIZE || [ [x,x,x] [n,n,n]]
    def forward(self, x):
        # split data
        x_input = x[:1].copy()    
        x_memory = self._memory[:1].copy() 

        #memory exist
        if len(x_memory) > 0:
            x_input = x_input + x_memory

        #new memory
        nx_input = x[1:].copy()    
        if nx_input.size > 0:  
            nx = [] 
            nx_memory = self._memory[1:]
            if len(nx_input) > len(nx_memory):
                nx = nx_input.copy()
                nx[:len(nx_memory)] += nx_memory
            else:
                nx = nx_memory.copy()
                nx[:len(nx_input)] += nx_input  
            self._memory = nx
        else:
            self._memory = self._memory[1:].copy()  
        #input
        x = x_input[0]   

        if self._rest_state > 0:
            self._rest_state = self._rest_state - 1
        else:    
            y = x * self._weight * self._enable
            y = sum(y) + self._potential

            #FIRE
            if y >= self._threshold:
                y_vector = np.zeros((self._out_dim , self._out_size))
                y_vector[self._out_spike_pos] = np.ones(self._out_size)
                self._potential = 0
                self._rest_state = self.rest_step
                if self._positive:
                    return y_vector
                else:
                    return y_vector * -1   
            #UPDATE POTENTIAL    
            else:
                self._potential = max(y - self._leak_rate, 0 )

        #ELSE RETURN 0    
        y_vector = np.zeros((self._out_dim , self._out_size))    
        return y_vector    




