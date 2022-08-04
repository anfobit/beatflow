class Module:
    def __init__(self): 
        pass
    def forward(self, x): 
        pass
    def __call__(self, x):
        return self.forward(x)       

    # 0 train, 1 test
    model_status = 0    
    def train(self): 
        self.model_status = 0
    def test(self):  
        self.model_status = 1  

    rest_step = 1

    learning_rate = 0.005

    max_dim = 50
 

   

    

     
