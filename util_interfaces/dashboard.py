from base import *
import ipywidgets


class TaskHandlerButton(TaskHandler):
    def __init__(self, name, state, func, widgets, *args, **kwargs):
        self.name = name
        self.state = state

        self.func = func
        self.task = None
        self.args = [self, *args]
        self.kwargs = kwargs

        self.widgets = widgets(self)

        self.box=None
        self.create_buttons(self.widgets)

    def change_name(self, name):
        self.name = name
        self.name_button.description = self.name

    @property
    def default_layout(self):
        return ipywidgets.Layout(width='auto', height='30px')

    def button(self, *args, **kwargs):
        if "layout" not in kwargs:
            kwargs["layout"] = self.default_layout
        return ipywidgets.Button(*args, **kwargs)

    def wfloat(self, *args, **kwargs):
        if "layout" not in kwargs:
            kwargs["layout"] = self.default_layout
        return ipywidgets.BoundedFloatText(*args, **kwargs)

    def wtext(self, *args, **kwargs):
        if "layout" not in kwargs:
            kwargs["layout"] = self.default_layout
        return ipywidgets.Text(*args, **kwargs)

    def create_buttons(self, widgets):
        name_button = self.button(description=self.name, disabled=True)
        start_button = self.button(description="start")
        start_button.on_click(self.start)
        stop_button = self.button(description="stop")
        stop_button.on_click(self.stop)
        # self.box=ipywidgets.Box([self.start_button,self.stop_button])
        default_buttons = name_button, start_button, stop_button
        # (setattr(i,"default",True) for i in default_buttons)
        self.box = ipywidgets.Box([*default_buttons, *widgets])

    def display(self):
        from IPython.display import display
        display(self.box)


def add_to_state_button(state, widgets):
    def wrapper(func):
        @wraps(func)
        def wrapped(name, *args, **kwargs):
            state.funclist.append(TaskHandlerButton(name, state, func, widgets, *args, **kwargs))
            return
        return wrapped
    return wrapper

async def sync_value(from_value, to_value,
                     from_value_name="value", to_value_name="value",
                     sync_time=0.01):
    while 1:
        await asyncio.sleep(sync_time)
        if hasattr(to_value, to_value_name) and hasattr(from_value, from_value_name): break
    while 1:
        await asyncio.sleep(sync_time)
        x = getattr(from_value, from_value_name)
        if x == getattr(to_value, to_value_name): continue
        setattr(to_value, to_value_name, x)


def text_widgets(task):
    return (task.wfloat(),)

@add_to_state_button(s, text_widgets)
async def loop_timer(task, sleep_interval=0.1, keep_len=10):
    # the variable used is called "loop_time"

    # TODO: better timing function to reduce execution time
    from collections import deque
    import time
    timed_list = deque((), keep_len)

    # state.loop_time=-1
    # old version no load loop time:1.0171284675598145
    # used "asyncio.create_task(sync_value(s,t,"loop_time"))" at outside

    # new version no load loop time:1.0043690204620361
    text_time = task.widgets[0]

    for _ in range(keep_len):
        await asyncio.sleep(sleep_interval)
        timed_list.append(time.time())

    while 1:
        await asyncio.sleep(sleep_interval)
        t = time.time()
        text_time.value = t - timed_list.popleft()
        timed_list.append(t)


# loop_timer("loop_timer")
