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

def to_pop(li):
    def poper(ret_len=5):
        p=li[:ret_len]
        li.pop(0)
        return p
    return poper

def add_num(func):
    num=-1
    def go_num():
        nonlocal num
        num+=1
        # r=num,*func()
        # incompatible
        r=[num]
        r.extend(func())
        return r
    return go_num

class Buff(list):
    def __init__(self, iterable=(),maxlen=128):
        self.maxlen=maxlen
        assert len(iterable)<=maxlen
        super().__init__(iterable)
    def extend(self):
        raise NotImplementedError
    
    def append(self, item):
        if len(self)>=self.maxlen:
            self.pop(0)
        super().append(item)

        
import ustruct as struct
def spliter(format,buffer):
    data=buffer[:struct.calcsize(format)]
    buffer=buffer[struct.calcsize(format):]
    ret=struct.unpack(format,data)
    if len(ret)==1:ret=ret[0]
    return ret,buffer

def get_wifi_location():
    import network
    wlan = network.WLAN(network.STA_IF)
    location=(wlan.ifconfig()[0], 8080)
    return location


import time

class UDP_Connection:
    def __init__(self,location=None):
        import socket
        if location is None:
            location=get_wifi_location()
        
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.bind(location)
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
        if len(self.to)==0:return   
        
        # use asyncio to handle loop send
        self.s.sendto(data,self.to[0])
        
from collections import deque;ret=deque((),1)
def exer(R,payload):
    global ret
    if b"ret" in payload:
        print("using 'ret' causes problem, skipping")
        return True
    
    to_ret=False
    if not b"=" in payload:
        payload=b"ret.append("+payload+b")"
        to_ret=True
        
    try:
        exec(payload)
        if to_ret:
            try:
                # g=ret[0]
                g=ret.popleft()
                try:
                    R.send_ret(g)
                except:
                    R.send_ret(str(g))
            except IndexError:g=None
    except Exception as e:
        R.send_ret("error"+str(e))
        return True
    
    return True



class Responder(UDP_Connection):
    def __init__(self, *args,send_rate=100,):
        super().__init__(*args)
        self.encode_list=[]
        self.decode_list=[]
        self.encode_list.append(["4B","get_s_time"])
        # self.encode_list.append(["b","get_rssi"])

        self.decode_list.append(["4B","rec_s_time"])
        # ESP不支持assignment

        self.mode=0
        
        self.ret_list=Buff(maxlen=10)


        
    def add_control_set(self,lis):
        self.decode_list.append(lis)
    def add_recv_set(self,lis):
        self.encode_list.append(lis)
        
        
    last_s_time=None
    s_number=0
    def get_s_time(self):
        s_time=time.localtime()[3:6]
        if s_time!=self.last_s_time:
            s_number=0
            last_s_time=s_time
        else:s_number+=1
        # ret=list(*s_time)+[s_number]
        ret=list(s_time)+[s_number]
        return ret

    
    #ESP不支持assignment
    dict={}
    def decode(self,buffer):
        for form,name in self.decode_list:
            data,buffer=spliter(form,buffer)
            
            #不支持drop掉比最新packet旧的packet
            try:
                if hasattr(self, name):
                    if type(data)==tuple:
                        getattr(self, name)(*data)
                    else:getattr(self, name)(data)
                else:
                    self.dict[name]=data
            except Exception as e:
                print(form)
                print(data)
                print(e)


    def encode(self):
        buffer=b""
        for form, name in self.encode_list:
            if name in self.__dict__ and not callable(self.__dict__[name]):
                data = self.__dict__[name]
            elif hasattr(self, name):
                data = getattr(self, name)()
            try:

                if type(data) in [tuple, list]:
                    buffer += struct.pack(form, *data)
                else:
                    buffer += struct.pack(form, data)
            except Exception as e:
                print(form)
                print(data)
                print(e)
        return buffer

    def send_ret(self,data,header=b"R"):
        import json
        data=json.dumps(data).encode()
        super().send(header+data)
        
    
    def send_exe(self,data,header=b"C"):
        if type(data)==str:data=data.encode()
        super().send(header+data)
        
    def send_m(self,header=b"M"):
        if header in b"MF":
            super().send(header+self.encode())
        
    @go_loop(0.01)         
    async def loop_send(self):
        if len(self.to)==0:return

        self.send_m()
            
    # @go_loop(0.01)         
    @go_loop(0.001)         
    async def loop_recv(self):
        #not_working_at_unix
        #?check later
        
        data=self()
        if not data:return
        
        header=data[0]
        data=data[1:]
        if header in b"MF":
            self.decode(data)
        if header in b"C":
            exer(self,data)
        if header in b"R":
            import json
            data=json.loads(data.decode())
            self.ret_list.append(data)
            
            
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