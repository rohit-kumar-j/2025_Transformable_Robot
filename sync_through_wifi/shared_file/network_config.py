SSIDS= {
    "iot_lab": "44FEC4DDB9", 
    #"Gadgets&Games": "",
}
dhcp_hostname="esp32_s2_3"

import network
wlan = network.WLAN(network.STA_IF)
wlan.active(True)


def print_config():
    print(wlan.config('dhcp_hostname'))
    print('network config:', wlan.ifconfig())
    
def after_connect():
    import webrepl
    webrepl.start()
    
import uasyncio as asyncio

network.WLAN(network.AP_IF).active(False)

async def do_connect(loop,status):        
    retry=5
    retry_loop_time=5
    wait_loop_time=10
    wait_for=5
    tried=0
    
    status.wifi=0
    
    # wlan = network.WLAN(network.STA_IF)
    
    while 1:
        if wlan.isconnected():
            status.wifi=1
            #test_loop
            await asyncio.sleep(wait_loop_time)
            #I mean, if the network is down, it is down.
            #trying the loop too frequently will not help.
            #so knowing that will not help
            if tried:tried=0
            continue
        
        if tried:
            await asyncio.sleep(retry_loop_time)
        tried+=1
            
        wlan.active(True)
        await asyncio.sleep(0.1)
        wlan.config(dhcp_hostname=dhcp_hostname)
        
        
        S=[str(i[0],"utf-8") for i in sorted(wlan.scan(),key=lambda x:x[3],reverse=True)]
        for s in SSIDS:
            #由SSIDS优先
            if s in S:
                print(f"connecting to {s}")
                wlan.connect(s, SSIDS[s])
                for i in range(wait_for):
                    if wlan.isconnected():
                        break
                    await asyncio.sleep(1)
                else:print("connection time out")

        if wlan.isconnected():
            print_config()
            after_connect()
            continue
        else:print("trying again")
        if tried>retry:break
        else:
            print("closing for " + str(retry_loop_time) + "seconds")
            wlan.active(False)
    
    wlan.active(False)
    ap = network.WLAN(network.AP_IF)
    ap.active(True)
    ap.config(essid='TOMSESP',password="13501594")
    status.wifi=2