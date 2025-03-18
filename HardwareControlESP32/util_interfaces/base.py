import asyncio
from functools import wraps


def loop(t=0.001):
    def wrapper(func):
        @wraps(func)
        async def wrapped(*args, **kwargs):
            import asyncio
            while 1:
                await asyncio.sleep(t)
                if asyncio.iscoroutinefunction(func):
                    ret=await func(*args, **kwargs)
                else:
                    ret=func(*args, **kwargs)
                if ret:break
            return ret
        return wrapped
    return wrapper


def try_loop(t=0.001,skip_errors=(),jump_errors=(asyncio.CancelledError,KeyboardInterrupt)):
    def wrapper(func):
        @wraps(func)
        async def wrapped(*args, **kwargs):
            while 1:
                await asyncio.sleep(t)
                try:
                    if asyncio.iscoroutinefunction(func):
                        ret=await func(*args, **kwargs)
                    else:
                        ret=func(*args, **kwargs)
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


import logging
def handle_task_result(task: asyncio.Task) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        pass  # Task cancellation should not be logged as an error.
    except Exception:  # pylint: disable=broad-except
        logging.exception('Exception raised by task = %r', task)


class State:
    funclist = []  # for TaskHandler


class TaskHandler:
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.task = None
        self.args = args
        self.kwargs = kwargs

    def start(self, l=None):
        if not self.is_running:
            self.task = asyncio.create_task(self.func(*self.args, **self.kwargs))
            self.task.add_done_callback(handle_task_result)

    error = None

    @property
    def is_running(self):
        if self.task is None: return False
        if not self.task.done(): return True
        self.end()
        del self.task
        self.task = None

    def cancel(self):
        if not self.is_running: return
        self.task.cancel()

    def stop(self, l=None):
        self.cancel()

    def end(self):
        try:
            self.task.result()
        except asyncio.CancelledError:
            pass
        except asyncio.InvalidStateError:
            print("result not good")
        except Exception as e:
            self.error = e

    name = ""

    def __str__(self):
        return self.name

    @classmethod
    def add_to_state(cls,state):
        def wrapper(func):
            @wraps(func)
            def wrapped(*args, **kwargs):
                state.funclist.append(cls(func, *args, **kwargs))
                return
            return wrapped
        return wrapper

    @classmethod
    def add_to_state_button(cls,state, widgets):
        def wrapper(func):
            @wraps(func)
            def wrapped(name, *args, **kwargs):
                thing=cls(name, state, func, widgets, *args, **kwargs)
                state.funclist.append(thing)
                return thing
            return wrapped
        return wrapper



class StateT(State):
    def __init__(self):
        print("""info
        
        Always be aware that this version of dashboard is instanced when defination.
        Which means that there cannot be two dashboard in the same ipynb kernel.
        
        """)

    def find_func_by_name(self, name):
        for i in self.funclist:
            if i.name == name: return i

    def start_func(self):
        ...

    def display(self):
        for i in self.funclist:
            i.display()


s = StateT()

