from full_udp import *
from ulab import numpy as np
from machine import Pin,I2C,PWM



class RealMotor:
    def __init__(self,esp):
        self.esp=esp
        # import d1motor
        # esp.m0 = d1motor.Motor(0, esp.I)
        # esp.m1 = d1motor.Motor(1, esp.I)
        
        self.max_torque = 2.0
        # self.max_torque = 1.0
        
        self.last_raw_obs=None
        
        self.trun_flag=False
        
        
        self.a=PWM(Pin(39), freq=50, duty=0)
        self.b=PWM(Pin(37), freq=50, duty=0)
        self.c=PWM(Pin(35), freq=50, duty=0)
        self.d=PWM(Pin(33), freq=50, duty=0)
    
    def run(self,u):
        u=int(u)
        
        # esp.m0.speed(-u)
        
        
        a=max(0,u)
        b=max(0,-u)

        self.a.duty(aa)
        self.b.duty(bb)
    
    def get_raw_obs(self,):
        # return self.get_angle()
        import time
        return self.get_angle(),time.ticks_ms()
    
    def __call__(self,torques,boost=1):
        esp=self.esp
        from ulab import numpy as np
        
        u = np.clip(torques, -self.max_torque, self.max_torque)[0]
        
        u*=8*boost
        u=int(u)
        
        
        
        
    async def go_to(self,to,speed):
        # TODO need to use that ANN
        esp=self.esp
        
        angle=self.get_angle()
        
        di=to-angle
        # 3.14左右
        
        self([0,])
        await asyncio.sleep(0.5)
        
        if speed<0:
            self([-di,],boost=1.5)
            await asyncio.sleep(0.5)
        
        self([di,],boost=1.5)
        await asyncio.sleep(0.5)
        
        self([0,])
        wait=speed
        wait*=0.5
        wait = np.clip(wait,0,1)
        await asyncio.sleep(wait)
        return
        
    def get_angle(self,):
        print("should be replaced")
        
    @property
    def state(self,):
        import time
        if self.last_raw_obs is None:
            self.last_raw_obs=self.get_raw_obs()

        ang0,time0=self.last_raw_obs
        # ang0=self.last_raw_obs
        self.last_raw_obs=self.get_raw_obs()
        ang1,time1=self.last_raw_obs
        # ang1=self.last_raw_obs
        
        
        timed=time.ticks_diff(time0,time1)/1000
        # timed=0.05
        angd=ang0-ang1
        angd=angle_normalize(angd)
        
        def precision_protection(angd):
            import math
            if angd>math.pi/2:
                self.trun_flag=True
                print("!",end="")
                print(angd,end="")
        precision_protection(angd)
        
        return np.array([ang1, angd/timed])
        # as long as it is reguler
        # time should not varient
    
def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
        