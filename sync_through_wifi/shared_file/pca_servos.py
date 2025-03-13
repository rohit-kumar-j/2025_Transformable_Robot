from pca9685 import PCA9685
from machine import Pin,I2C

class Servo:
    def __init__(self,duty_func,range_min=-1, range_max=1):
        self.range_min = range_min
        self.range_max = range_max
        
        self.min,self.max=80,490
        self.offset=0
        
        
        self.duty=duty_func
        
    last_ret=0
    def __call__(self, *args, **kwargs):
        get = args[0]
        get+=self.offset

        b = min(max(self.range_min, get), self.range_max)

        b -= (self.range_max + self.range_min) / 2
        b /= (self.range_max - self.range_min) / 2
        # 归一化(-1,1)
        
        
        b *= (self.max - self.min) / 2
        b += (self.max + self.min) / 2
        
        b=int(b)
        ret=b
        if self.last_ret==ret:return
        self.last_ret=ret

        #TODO:
        # 省电模式
        # 减缓上次改变的速度，增加按钮式滤波延迟


        self.duty(ret)

    def off(self):
        self.duty(0)

class Leg:
    def __init__(self,s1,s2,offset_s1=0,offset_s2=0):
        # offset should be in range (1,-1)
        # maybe 0.1,-0.1
        # float
        
        s1.offset=offset_s1
        s2.offset=offset_s2

        self.s1,self.s2=s1,s2
    
    def to_neutral(self):
        self.go(0,0)
        
        
    def go(self,a,b):
        self.s1(a)
        self.s2(b)
    
    dry_run=False
    def __call__(self, *args, **kwargs):
        if len(args) == 1:
            up = 0
            change = args[0]
        else:
            change, up = args
        
        a=up + change
        b=up - change
        
        if self.dry_run:
            print(*[round(i,2) for i in (up,change,b)],sep="\t")
        else:
            self.go(a,b)
        

class PcaMover:
    def __init__(self,I):
        self.servos=[Servo(self.give_duty_func(i)) for i in range(12)]
        
        self.board = PCA9685(I,64)
        self.board.freq(50)
        
        self.legs=[Leg(self.servos[i*2],self.servos[i*2+1]) for i in range(5)]
    
    def give_duty_func(self,num):
        def go(x):
            self.board.duty(num,x)
        return go
    
    def to_neutral(self):
        return [i.to_neutral() for i in self.legs]
    
    def off(self):
        return [i.off() for i in self.servos]
    
