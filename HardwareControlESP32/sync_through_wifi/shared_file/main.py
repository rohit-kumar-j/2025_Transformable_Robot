from network_online import *

loop=asyncio.get_event_loop()

print(K)

from go import *
# from go_simple import *

import uasyncio as asyncio
async def main_loop(loop,K):
    loop.create_task(have_online(loop))
    await asyncio.sleep(0.1)
    loop.create_task(K.run(loop))
    
    loop.create_task(go(loop,K))
    
    
    print("="*50)
    print("all_task_created")
    print("="*50)
    
    
loop.create_task(main_loop(loop,K))
try:
    loop.run_forever()
except:
    pass
finally:
    if wlan.isconnected():
        # C.close_client()
        pass