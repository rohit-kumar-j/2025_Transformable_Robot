from dashboard import *


class TaskHandlerButtonAdvanced(TaskHandlerButton):
    class Button(ipywidgets.Button):
        def __init__(self,task,name):
            if name=="":
                super().__init__(layout=ipywidgets.Layout(width='auto', height='30px'))
            else:
                super().__init__(description=name,layout=ipywidgets.Layout(height='30px'))

            self.name=name
            self.task=task
            # self.state=task.state

        default=False

    class StateButton(Button):
        def __init__(self,task=None,name="",click=True,color1="lightblue", color2="red"):
            super().__init__(task,name,)

            self.style.button_color = 'gray'
            self.state_color = color1
            self.non_state_color = color2
            if click:
                self.on_click(self.called)

        @property
        def state(self):
            return self.style.button_color == self.state_color

        def change_state(self):
            if self.state:
                self.style.button_color = self.non_state_color
            else:
                self.style.button_color = self.state_color

        def called(self, things=None):
            if not self.state:
                self.task.start()
            else:
                self.task.stop()

            # TODO: change_state default, sync_state with running
            self.change_state()

        @loop()
        def async_state(self, func,args=()):
            self.sync_state(func,args)

        def sync_state(self, func,args=()):
            if callable(func):f=func
            else:f=lambda: func

            if f(*args) != self.state:
                self.change_state()


        @property
        def value(self):
            return self.state

        @value.setter
        def value(self, value):
            self.sync_state(value)

        def run_sub_task(self,awaitable):
            asyncio.create_task(awaitable)

        def self_task(self):
            self.sub_task=self.run_sub_task(self.async_state(getattr,(self.task,"is_running")))


    def create_buttons(self, widgets):
        r_but=self.StateButton(self,click=False)
        r_but.self_task()

        default_buttons=[self.StateButton(self,self.name),r_but]
        [setattr(i,"default",True) for i in default_buttons]
        self.box = ipywidgets.Box([*default_buttons, *widgets])