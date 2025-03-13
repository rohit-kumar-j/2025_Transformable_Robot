import uasyncio as asyncio
# from functools import wraps

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

def try_loop(t=0.001,skip_errors=(),jump_errors=(asyncio.CancelledError,KeyboardInterrupt)):
    def wrapper(func):
        # @wraps(func)
        async def wrapped(*args, **kwargs):
            while 1:
                await asyncio.sleep(t)
                try:
                    ret=await func(*args, **kwargs)
                    if ret:break
                except tuple(jump_errors):
                    return
                except tuple(skip_errors):
                    pass
                except Exception as e:
                    print("")
                    print(e)
                    print("continue running")
            return
        return wrapped
    return wrapper

# try doing this lead to 30% of efficiency loss
# def get_time():
    # import time
    # return time.ticks_ms()