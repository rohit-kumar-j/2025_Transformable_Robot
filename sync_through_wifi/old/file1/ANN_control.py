from full_udp import *

from machine import Pin,I2C,PWM,ADC

esp=Responder()
R=esp

# esp.watch

esp.light=PWM(Pin(15,Pin.OUT))
esp.light.freq(1000)
from drums import *


a=PWM(Pin(39), freq=50, duty=0)
b=PWM(Pin(37), freq=50, duty=0)
c=PWM(Pin(35), freq=50, duty=0)
d=PWM(Pin(33), freq=50, duty=0)
# esp.I=I2C(0,scl=Pin(35),sda=Pin(33))
# esp.I=I2C(0,scl=Pin(35),sda=Pin(33),freq=1000000)
# esp.I=I2C(0,scl=Pin(35),sda=Pin(33),freq=800000)
esp.I=I2C(0,scl=Pin(7),sda=Pin(9),freq=800000)
# 瓶颈在i2c上？

@go_loop(1)
async def mem_manage():
    import gc
    # Necessary for long term stability
    gc.collect()
    gc.threshold(gc.mem_free() // 4 + gc.mem_alloc())


from as5600 import AS5600
esp.as1=AS5600(esp.I,54)
# def get_angle(as5600,offset=50):
def get_angle(offset=-14):
    as5600=esp.as1
    g=as5600.ANGLE
    # 0~4096
    g=g/4096*360
    # 0~360
    g+=offset
    g%=360
    g-=180
    # -180~+180
    import math
    return g/180*math.pi
    return g

from real_env import RealMotor


# TODO: need to change after things?
esp.real=RealMotor(esp)


esp.state_and_action_queue=Buff(maxlen=5)
def get_trajectory():
    if len(esp.state_and_action_queue):
        return esp.state_and_action_queue[-1]
esp.trajectory=get_trajectory

from env_actor import *
esp.pi=MyRelu()
esp.env=PendulumEnv()

esp.real.get_angle=get_angle

esp.env.move=esp.real

async def learn_loop():
    esp.state=await esp.env.reset()
    while 1:
        # await asyncio.sleep(0.02)
        # await learn(env,pi,s,state_and_action_queue)
        
        await learn(esp)
        # send_trajectory
        esp.send_m()
        # print(".",end="")
        
esp.params=esp.pi.params_unpack

def get_times(t=10):
    import time
    g=time.time()
    for _ in range(t):
        esp.real.get_angle()
    return time.time()-g

async def go(loop,K):
    while K.wifi is not 1:await asyncio.sleep(0.1)
    
    import network
    wlan = network.WLAN(network.STA_IF)
    con=False
    if '192.168.2.106' in wlan.ifconfig():con=True
    if con:
        pass
    else:
        print("CON")
    
    
    # R.watchdog=Watch(R.get_last_call)
    # loop.create_task(R.watchdog.foo(off))
    
    
    # loop.create_task(send_mpu_loop())
    print(wlan.ifconfig())
    
    esp.x=loop.create_task(esp.loop_recv())
    # esp.y=loop.create_task(esp.loop_send())
    
    if True:
        g=I_am_the_doctor(*syl(*sleep_state(give_state_func(esp.light))))
        esp.drums=loop.create_task(g)
    
    
    
    # esp.touched_a=loop.create_task(clicks(list(),touch_a,do_relay))
    # esp.touched_b=loop.create_task(clicks(esp.go_pin_clicked,touch_b,do_relay_off))
    
    
    
    
    
    
    
    
    esp.mem=loop.create_task(mem_manage())
    
    esp.done=True
