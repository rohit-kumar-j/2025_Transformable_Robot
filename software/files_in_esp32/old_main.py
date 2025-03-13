import network,utime
# import webrepl
# webrepl.start()
SSIDS={"iot_lab": "44FEC4DDB9"}

network.WLAN(network.AP_IF).active(False)

def do_connect(name=None,repeat=3,stop=5):
    import network
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if name!=None:
        if type(name)==str:
            for s in SSIDS:
                if name in s:
                    wlan.connect(s, SSIDS[s])
                    return
        elif type(name)==int:repeat=name

    print(wlan.config('dhcp_hostname'))
    # print(wlan.config(dhcp_hostname="tom123"))
    rep=0
    while 1:
        rep+=1
        wlan.active(True)

        S=[str(i[0],"utf-8") for i in sorted(wlan.scan(),key=lambda x:x[3],reverse=True)]
        for s in S:
            if s in SSIDS:
                wlan.active(True)
                wlan.connect(s,SSIDS[s])
                start = utime.time()
                while not wlan.isconnected():
                    utime.sleep(1)
                    if utime.time() - start > 5:
                        print("connect timeout!")
                        break

                if wlan.isconnected():
                    print('network config:', wlan.ifconfig())
                    return

        wlan.active(True)
        if not wlan.isconnected():
            print("tried "+str(repeat))
            utime.sleep(1+rep)
        if not wlan.isconnected():
            print("tryingagain")
        if rep>repeat:break
        else:
            print("closing for " + str(stop) + "seconds")
            wlan.active(False)
            utime.sleep(stop)
    if not wlan.isconnected():
        wlan.active(False)
        wlan = network.WLAN(network.AP_IF)
        wlan.active(True)
        wlan.config(essid='TOMSESP',password="13501594")
try:
    do_connect()
    import webrepl
    webrepl.start()
except Exception as e:
    try:
        network.telnet.start(user="tom",password="135015")
        network.ftp.start(user="tom",password="135015")
    except:
        print("in Main")
        print(e)
        import machine
        machine.reset()
