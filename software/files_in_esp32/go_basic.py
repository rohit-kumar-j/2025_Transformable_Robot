from full_udp import *

from machine import Pin,I2C,PWM
btn = Pin(39, Pin.IN)

pin_a=Pin(26)
pin_b=Pin(32)

pwm_a = PWM(pin_a)
pwm_a.freq(50)
pwm_b = PWM(pin_b)
pwm_b.freq(50)

from pca_servos import Servo
ser_a=Servo(pwm_a.duty)
ser_b=Servo(pwm_b.duty)

ser_a.min=25
# ser_a.max=135
ser_a.max=125
ser_b.min=25
ser_b.max=125

from matrix_5x5 import np

def change_light(*byt,change=False):
    np.buf=bytearray(byt)
    if change:np.write()
    
esp=Responder()
# esp.watch
esp.np=np
esp.change_light=change_light

esp.ser_a=ser_a
esp.ser_b=ser_b

R=esp


from fuse_mpu import *



async def go(loop,K):
    while K.wifi is not 1:await asyncio.sleep(0.1)
    
    import network
    wlan = network.WLAN(network.STA_IF)
    con=False
    if '192.168.2.123' in wlan.ifconfig():con=True
    if con:
        pass
    else:
        print("CON")
    

    await fuse.start()
    esp.fuse=fuse
    esp.imu=imu
    
    # R.watchdog=Watch(R.get_last_call)
    # loop.create_task(R.watchdog.foo(off))
    
    
    # loop.create_task(send_mpu_loop())
    print(wlan.ifconfig())
    
    
    esp.x=loop.create_task(esp.loop_recv())
    # y=loop.create_task(esp.loop_send())
    
    # if 'wh' in S.__dict__:
    #     g=I_am_the_doctor(*syl(*sleep_state(give_state_func(S.light))))
    #     loop.create_task(g)