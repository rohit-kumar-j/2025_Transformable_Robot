#drums
# 4 in 6 xxxx__

# Doctor Who, I am the doctor.
# 6 in 7 XX XX XX _


def give_state_func(pwm):
    # alternative partial func
    def have_state(state):
        def pwm_state_func():
            # pwm = PWM(Pin(2))
            # pwm.freq(1000)
            if state==0:
                pwm.duty(800)
            else:
                pwm.duty(0)
        return pwm_state_func
    return have_state(0),have_state(1)

import uasyncio as asyncio

def sleep_state(li,s=0.1):
    def st(do):
        async def st():
            # await do()
            do()
            await asyncio.sleep(s)
        return st
    return [st(i) for i in li]

def syl(st1,st2):
    async def act():
        await st1()
        await st2()
    async def deact(l=2):
        for _ in range(l):
            await st2()
    return act,deact

def drums(a,d,loop=True):
    while 1:
        await a()
        await a()
        await a()
        await a()
        await d()
        await d()
        if not loop:break
        
    
def I_am_the_doctor(a,d):
    while 1:
        await a()
        await a()
        await a()
        await d(1)
    
