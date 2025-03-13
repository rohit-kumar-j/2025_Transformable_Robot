class MyRelu:
    def __init__(self):
        self.layer_li=["linear","linear_1","linear_2","linear_3",]
        self.params=self.init_params()
        
    def init_params(self):
        from ulab import numpy
        
        params={}
        params["linear"]={"w":numpy.zeros((3, 8)),"b":numpy.zeros((8,))}
        params["linear_1"]={"w":numpy.zeros((8, 8)),"b":numpy.zeros((8,))}
        params["linear_2"]={"w":numpy.zeros((8, 8)),"b":numpy.zeros((8,))}
        params["linear_3"]={"w":numpy.zeros((8,1)),"b":numpy.zeros((1,))}
        
        #this dict is not sorted list is needed
        return params
    @property
    def networks(self):
        def add(a,b):return a+b
        from ulab import numpy as np
    
        layer_name="linear"
        yield np.dot,self.params[layer_name]["w"]
        yield  add , self.params[layer_name]["b"]
        yield np.maximum,0
    
        layer_name="linear_1"
        yield np.dot,self.params[layer_name]["w"]
        yield  add , self.params[layer_name]["b"]
        yield np.maximum,0
    
        layer_name="linear_2"
        yield np.dot,self.params[layer_name]["w"]
        yield  add , self.params[layer_name]["b"]
        yield np.maximum,0
    
        layer_name="linear_3"
        yield np.dot,self.params[layer_name]["w"]
        yield  add , self.params[layer_name]["b"]
        
    def forward(self,inp):
        for func,num in self.networks:
            inp=func(inp,num)
        return inp
    
    def __call__(self,obs):
        from ulab import numpy
        return self.forward(numpy.array(obs))
    
    def params_unpack(self,*li):
        def prod(shape):
            if len(shape)==1:return shape[0]
            elif len(shape)==2:return shape[0]*shape[1]
            else:raise NotImplementedError
        
        from ulab import numpy
        params=self.params
        for layer in self.layer_li:
            things=self.params[layer]
            
            g_k=things["w"]
            shape=g_k.shape
            g_len=prod(shape)
            things["w"]=numpy.array(li[:g_len]).reshape(shape)
            li=li[g_len:]


            g_k=things["b"]
            shape=g_k.shape
            g_len=prod(shape)
            things["b"]=numpy.array(li[:g_len]).reshape(shape)
            li=li[g_len:]