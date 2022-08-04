from .module import Module
from .node import Node
import numpy as np

class Sync(Module):
    _learn_step = -1
    _step_sync = 0
    def __init__(self):
        super(Sync, self).__init__()

        self.cell1 = Node(in_size = 1, out_size = 2, weight = np.ones(1), leak_rate = 2)
        self.cell2a = Node(in_size = 1, out_size = 1, out_dim = 10, weight = np.ones(1), leak_rate = 2)
        self.cell2b = Node(in_size = 1, out_size = 1, out_dim = 10, weight = np.ones(1), leak_rate = 2)
        self.cell3 = Node(in_size = 2, out_size = 1, weight = np.ones(2), leak_rate = 2)

    def forward(self, x):
        x = self.cell1(x)  
        xa = np.reshape( x[:, 0], (1,1))
        xb = np.reshape( x[:, 1], (1,1)) 
        a = self.cell2a(xa)
        b = self.cell2b(xb)
        ab = np.column_stack((a,b)) 
        y = self.cell3(ab)

        

        if self._learn_step == -1:
            self._learn_step = 0
            self.cell2a.spike_delay()
        elif self._learn_step == 0:
            if sum(y) >= 1:  
                self._learn_step = 1
                #self.cell2a.spike_speed_up()         
            elif sum(a) >= 1: 
                if self._step_sync == 0:
                    self._step_sync = 1
                else:    
                    self.cell2a.spike_delay()   
                    self._step_sync = 0      
              
        return y