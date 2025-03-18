from network_config import *

class UDP_Connection:
    def __init__(self,port=61234):
        import socket, network
        wlan = network.WLAN(network.STA_IF)

        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.bind((wlan.ifconfig()[0], port))
        self.s.setblocking(False)

    to=[]
    def __call__(self, *args, **kwargs):
        try:
            data,host = self.s.recvfrom(8192)
            if not host in self.to: self.to.append(host)
            return data
        except Exception as e:
            if e.args[0]==11:return


    def send(self,data):
        self.s.sendto(data,self.to[0])

udp_f=UDP_Connection(61234)
from util import *
@go_loop(0.1)
async def fail_safe(loop):
    # raise a big error when udp recieved a pack
    r=udp_f()
    if r is None:return
    udp_f.send(b"recved")
    print(r.decode())
    if r==b"stop":
        loop.stop()
    

from self_aware import Keeper
K=Keeper()

import uasyncio as asyncio
async def have_online(loop):
    await asyncio.gather(do_connect(loop,K),fail_safe(loop))
    
    
