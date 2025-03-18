from util import *
class KeepStatus:
    def __init__(self, send_rate=100,):
        import network
        self.wlan = network.WLAN(network.STA_IF)
        
        self.counts_in_a_sec=0
        self.counts_in=0
    
    def get_rssi(self):
        return self.wlan.status("rssi")
    
    async def run(self,loop):
        # loop.create_task(self.test_fresh_rate())
        await asyncio.gather(self.loop_timer(),)
    
    loop_time=0
    async def loop_timer(task,sleep_interval=0.1,keep_len=10):
        from collections import deque
        timed_list=deque((),keep_len)
        import time

        for _ in range(keep_len):
            await asyncio.sleep(sleep_interval)
            timed_list.append(time.ticks_ms())

        while 1:
            await asyncio.sleep(sleep_interval)
            t=time.ticks_ms()
            task.loop_time=t-timed_list.popleft()
            timed_list.append(t)
    
    
    
class Keeper(KeepStatus):
    def fo(self):
        pass

        
        
