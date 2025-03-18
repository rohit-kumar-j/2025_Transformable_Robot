from full_udp import *

from machine import Pin,I2C,PWM,UART

esp=Responder()
R=esp

# esp.watch

esp.light=PWM(Pin(15,Pin.OUT))
esp.light.freq(1000)
from drums import *

esp.I=I2C(0,scl=Pin(35),sda=Pin(33))

# from ulab import numpy as np

@go_loop(1)
async def mem_manage():
    import gc
    # Necessary for long term stability
    gc.collect()
    gc.threshold(gc.mem_free() // 4 + gc.mem_alloc())


# from as5600 import AS5600

# from tmc2209_simple import TMC_2209


async def go(loop,K):
    while K.wifi is not 1:await asyncio.sleep(0.1)
    
    import network
    wlan = network.WLAN(network.STA_IF)
    con=False
    if '192.168.2.122' in wlan.ifconfig():con=True
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
    
    from pca9685 import PCA9685
    esp.board = PCA9685(esp.I,64)
    esp.board.freq(5000)
    
    # from pca_servos import PcaMover
    
    # esp.pca=PcaMover(esp.I)
    # esp.pca.servos[4].offset=-0.05
    # esp.pca.servos[5].offset=-0.16
 
    # from debug_util import timed_function
    # @timed_function
    # 10ms
    # def pca_moves(li):
    #     for num,i in enumerate(li):
    #         esp.pca.servos[num](i)
    # esp.pca_moves=pca_moves
    
    def pca_moves(li):
        for num,i in enumerate(li):
            esp.board.duty(num,i)
    esp.pca_moves=pca_moves
    
    
    esp.mem=loop.create_task(mem_manage())
    
    esp.done=True
