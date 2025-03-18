import uasyncio as asyncio

def go_loop(t=0.001):
    def wrapper(func):
        # @wraps(func)
        async def wrapped(*args, **kwargs):
            while 1:
                await asyncio.sleep(t)
                ret=await func(*args, **kwargs)
                if ret:break
            return
        return wrapped
    return wrapper



def do_exec(payload):
    try:
        exec(payload)
    except Exception as e:
        print("error"+str(e))
        return True
    
class Watch:    
    def __init__(self,watch_func):
        self.last=False
        self.watch_func=watch_func
    
    @go_loop(0.1)
    async def foo(self,func):
        if self.watch_func()<=0:
            if self.last:
                func()
                self.last=False
        else:self.last=True


class Dog:
    last_call = 0

    @go_loop(0.05)
    async def watch(self):
        if self.last_call > 0: self.last_call -= 1

    def get_last_call(self):
        return self.last_call
