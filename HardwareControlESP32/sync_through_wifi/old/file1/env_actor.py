from reluer import MyRelu
from ulab import numpy as np
from full_udp import *


class PendulumEnv:
    def __init__(self, ):
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.02
        # 就当它是0.05
        self.dt = 0.05
        self.g = 10
        self.m = 1.0
        self.l = 1.0
        
        self.max_step=200
        self.now_step=0
        
        # self.states=Buff(maxlen=10)
        self.states=Buff(maxlen=5)
        
    
    @property
    def state(self,):
        s=self.move.state
        self.states.append(s[0])
        self.destabilize()
        return self.move.state

    def destabilize(self,):
        if len(self.states)<4:return
        if self.now_step<10:return
        
        def check_stable(states):
            return min(states)==max(states)
        
        if not check_stable(self.states):return
        
        self.now_step=200
    
    async def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        # dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        
        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)
        
        self.move([u,])
        # await asyncio.sleep(0.02)
        await asyncio.sleep(0.01)
        ###
        # fake model
        
        # print("This is a dream")
        # newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u) * dt
        # newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        # newth = th + newthdot * dt
        # self.state = np.array([newth, newthdot])
        
        ###
        
        
        
        
        self.now_step+=1
        if self.now_step>self.max_step:
            return self.get_obs(), -costs, False, True
        return self.get_obs(), -costs, False, False
    
    async def reset(self):
        self.now_step=0
        import random
        to=random.uniform(-3.14, 3.14),random.uniform(-1, 1)
        
        await self.move.go_to(*to)
        
        return self.get_obs()

    def get_obs(self):
        theta, thetadot = self.state
        
        if self.move.trun_flag:
            self.now_step=200
            self.trun_flag=False
        
        return np.array([np.cos(theta), np.sin(theta), thetadot])


def angle_normalize(x):
    return x
    return ((x + np.pi) % (2 * np.pi)) - np.pi
        
    
    
    
# in ANN_controller.py
# s=env.reset()
async def learn(esp):
    env,pi,s,state_and_action_queue=esp.env,esp.pi,esp.state,esp.state_and_action_queue
    from ulab import numpy
    if True:
        a=pi(s)
        
        s_next, r, done, truncated = await env.step(a)
        
        done=truncated or done
        
        # reward should be defined outside esp32
        state_and_action_queue.append(numpy.concatenate((s,s_next,a,)).tolist()+[r,done])
        # state_and_action_queue.append([s,s_next,a])
        # (s, a, r, done)

        if done or truncated:
            esp.state = await env.reset()
            return

        esp.state = s_next