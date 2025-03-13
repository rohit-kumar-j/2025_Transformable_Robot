from full_udp import *
from machine import Pin,I2C,PWM,ADC
import aioespnow

e = aioespnow.AIOESPNow()  # Returns AIOESPNow enhanced with async support
e.active(True)
peer = b"H'\xe2V\x00>"
e.add_peer(peer)

async def heartbeat(e, peer, period=3):
    while True:
        if not await e.asend(peer, b'ping'):
            print("Heartbeat: peer not responding:", peer)
        else:
            print("Heartbeat: ping", peer)
        await asyncio.sleep(period)

import time
async def echo_server(e):
    async for mac, msg in e:
        # print("E: ", msg,"t: ",time.ticks_ms())
        # print(int(msg))
        # print("loop_time(ms):", int(msg)-time.ticks_ms())
        if time.ticks_ms()<esp.self_echo_stop:
            esp.esp_now_count+=1
            await e.asend(mac, msg, False)
        else:
            try:
                # print("the_messag: ",int(msg))
                # print("now_tic_ms: ",time.ticks_ms())
                print("loop_time(ms):", time.ticks_ms()-int(msg))
                print("count: ",esp.esp_now_count)
            except:
                if msg==b'hihi':
                    print(msg)
                pass
                # print(msg)
async def test_sequence():
    # test_for_a__min
    print("test seq")

    slee=1
    e.send(peer,str(time.ticks_ms()).encode())
    esp.self_echo_stop=time.ticks_ms()+slee*1000
    esp.esp_now_count=0
    await asyncio.sleep(slee*2)
    
    
    e.send(peer,b't')
    
    #disconnect wifi?
    print("wifi down")
    wlan.disconnect()
    e.active(False)
    await asyncio.sleep(1)
    e.active(True)
    e.add_peer(peer)
    
    await asyncio.sleep(1)
    
    
    e.send(peer,str(time.ticks_ms()).encode())
    esp.self_echo_stop=time.ticks_ms()+slee*1000
    esp.esp_now_count=0
    await asyncio.sleep(slee*2)
    
    print("test sequence done")
    wlan.connect("Gadgets&Games", "")
    
    
    await asyncio.sleep(1)
    
    # print(wlan.ifconfig())
    
    
            
            
esp=Responder()

esp.self_echo_stop=0
esp.esp_now_count=0

R=esp

#esp.light=PWM(Pin(15,Pin.OUT))
esp.light=PWM(Pin(8,Pin.OUT))
esp.light.freq(1000)
from drums import *

@go_loop(1)
async def mem_manage():
    import gc
    # Necessary for long term stability
    gc.collect()
    gc.threshold(gc.mem_free() // 4 + gc.mem_alloc())

import network
wlan = network.WLAN(network.STA_IF)

async def go(loop,K):
    while K.wifi is not 1:await asyncio.sleep(0.1)

    esp.espnow_server=asyncio.create_task(echo_server(e))
    # esp.espnow_heart=asyncio.create_task(heartbeat(e, peer, 3))
    
    #esp.test_seq=asyncio.create_task(test_sequence())
    
    print(wlan.ifconfig())
    
    # esp.x=loop.create_task(esp.loop_recv())
    # esp.y=loop.create_task(esp.loop_send())
    
    if False:
        g=I_am_the_doctor(*syl(*sleep_state(give_state_func(esp.light))))
        esp.drums=loop.create_task(g)
    
    
    # esp.mem=loop.create_task(mem_manage())
    
    esp.done=True

    print("local_done")
