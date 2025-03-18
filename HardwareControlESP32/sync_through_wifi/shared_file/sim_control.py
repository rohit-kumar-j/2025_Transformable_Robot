from full_mqtt import *
from machine import Pin,I2C,PWM,ADC
import aioespnow

from drums import *

@go_loop(1)
async def mem_manage():
    import gc
    # Necessary for long term stability
    gc.collect()
    gc.threshold(gc.mem_free() // 4 + gc.mem_alloc())

import network
wlan = network.WLAN(network.STA_IF)

class E:pass
esp=E()

esp.I=I2C(0,scl=Pin(2),sda=Pin(3))
# esp.I=I2C(0,scl=Pin(33),sda=Pin(35))
from pca9685 import PCA9685


async def go(loop,K):
    global esp
    while K.wifi is not 1:await asyncio.sleep(0.1)
                
    Res=Responder(("10.144.113.5",1885),"ho1")
    Res.to="ho1_ret"
    
    esp.self_echo_stop=0
    esp.esp_now_count=0
    
    # esp.light=PWM(Pin(15,Pin.OUT))
    esp.light=PWM(Pin(8,Pin.OUT))
    esp.light.freq(1000)

    # esp.espnow_server=asyncio.create_task(echo_server(e))
    # esp.espnow_heart=asyncio.create_task(heartbeat(e, peer, 3))
    
    # esp.test_seq=asyncio.create_task(test_sequence())
    
    print(wlan.ifconfig())
    
    esp.x=loop.create_task(Res.loop_recv())
    # esp.y=loop.create_task(esp.loop_send())
    
    if 1:
        g=I_am_the_doctor(*syl(*sleep_state(give_state_func(esp.light))))
        esp.drums=loop.create_task(g)



    from pca_servos import PcaMover
    esp.pca=PcaMover(esp.I)
    
    esp.pca.servos[0].offset=0.17
    esp.pca.servos[1].offset=0.17
    esp.pca.servos[2].offset=0.17
    esp.pca.servos[3].offset=0.17
    esp.pca.servos[4].offset=0.17
    esp.pca.servos[5].offset=0.17
    def pca_moves(li):
        for num,i in enumerate(li):
            # esp.board.duty(num,i)
            esp.pca.servos[num](i)
    esp.pca_moves=pca_moves
    
    
    # esp.mem=loop.create_task(mem_manage())
    
    esp.done=True

    print("local_done")
