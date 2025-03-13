from full_udp import *

from machine import Pin,I2C,PWM,TouchPad


def touch_IO(touch,threshold=200):
    def go():
        nonlocal touch,threshold
        ret=touch.read()
        if abs(ret-threshold)>20:
            if ret>threshold:threshold+=1
            if ret<threshold:threshold-=1
        return int(ret>threshold)
    return go

touch_b=touch_IO(TouchPad(Pin(33)),700)


esp=Responder()
R=esp

# esp.watch

esp.light=PWM(Pin(16,Pin.OUT))
esp.light.freq(1000)
from drums import *

# esp.I=I2C(0,scl=Pin(22),sda=Pin(23))

    
    
@go_loop(0.05)
async def clicks(clicked,read_func,do_func):
    #开灯开servo
    clicked.append(read_func())
    if len(clicked)>10:clicked.pop(0)
    if len(clicked)<10:return
    
    changes=[i-ii for i,ii in zip(clicked[0:9],clicked[1:10])]
    main_change=changes[-1]
    
    
    su=sum(changes)
    if su==0:return

    k=tuple(i for i, x in enumerate(changes) if x==1)
    if not len(k):k=0
    else:k=k[0]

    k=len(clicked)-k-2
    #k range from 0 to clicked-1 
    #to start from 0

    if su==1:
        do_func(k)
        
        
esp.go_pin_clicked=[]  
def do_relay(num):
    if len(esp.go_pin_clicked)<10:return
    # if sum(esp.go_pin_clicked)!=len(esp.go_pin_clicked):return
    if num==8:
        R.s.sendto('Cesp.relay.off();_=0',('192.168.2.223',8080))
        esp.drums.cancel()
    
def do_relay_off(num):
    # print(num)
    if num==0:
        R.s.sendto('Cesp.relay.on();_=0',('192.168.2.223',8080))
        
        if esp.drums.done():
            g=I_am_the_doctor(*syl(*sleep_state(give_state_func(esp.light))))
            esp.drums=esp.loop.create_task(g)
    
# loop.create_task(go(loop,K))
# loop.run_forever()

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
    
    
    # R.watchdog=Watch(R.get_last_call)
    # loop.create_task(R.watchdog.foo(off))
    
    
    # loop.create_task(send_mpu_loop())
    print(wlan.ifconfig())
    
    esp.x=loop.create_task(esp.loop_recv())
    # y=loop.create_task(esp.loop_send())
    
    if True:
        g=I_am_the_doctor(*syl(*sleep_state(give_state_func(esp.light))))
        esp.drums=loop.create_task(g)
    
    
    
    esp.touched_a=loop.create_task(clicks(list(),touch_b,do_relay))
    esp.touched_b=loop.create_task(clicks(esp.go_pin_clicked,touch_b,do_relay_off))
    
    
    
    
    def loop_dance():
        @go_loop(0.2)
        async def dance():
            try:
                esp.currents.append(esp.ina.current)
            except:return
        
            if len(esp.currents)>10:esp.currents.pop(0)
            if len(esp.currents)<10:return
            # error[Errno 116] ETIMEDOUT
            
            # if sum(esp.currents)<1:relay.off()
            
        return dance()
    
    # esp.loop_dance=loop.create_task(loop_dance())
    
    
    
    
    esp.loop=loop
    esp.done=True