import websocket
import time
import logging
import threading
from collections import deque
class littlesock():
    def __init__(self,sock,logger:logging.getLogger=logging,buffer=10):
        self.logger=logger
        self.messages=deque(maxlen=buffer)
        websocket.enableTrace(True)
        self.web = websocket.WebSocketApp(sock, on_message=self.OnMess, on_error=self.OnError)
        self.T=threading.Thread(target=self.web.run_forever)
        self.T.daemon = True
        # self.web.keep_running = 1
        self.T.start()
        conn_timeout = 5
        try:
            while not self.web.sock.connected and conn_timeout:
                time.sleep(1)
                conn_timeout -= 1
        except AttributeError:
            self.messages.append("failed")
    def __enter__(self):
        return self

    def OnMess(self,ws,message=""):
        if not message:message=ws
        logging.info(message)
        self.messages.append(message)

    def OnError(self,ws,error=""):
        logging.warning(error)

    def __exit__(self, exc_type="", exc_val="", exc_tb=""):
        self.web.close()
        #可以不用↓
        self.T.join()
        self.logger.info(exc_type)
        self.logger.info(exc_val)
        self.logger.info(exc_tb)
        print("messg::",self.messages,end=" ")
        return self.messages

#fake
class littlesock1():
    messages=deque(maxlen=10)

import re,socket
class espweber():
    board=0

    def __enter__(self):
        return self

    normal=1
    def __init__(self,place,normal=0,logger=logging):
        try:place=int(place)
        except:
            if not re.search("\.",place):place+=".mshome.net"
            logger.debug(place)
            try:place=socket.gethostbyname(place)
            except:place=socket.gethostbyname("ESP_"+place)
        self.plac=place
        if type(place)==int:
            place=str(place)
            place="ws://192.168.1." + place + ":8266/"
        else:
            place="ws://"+place+":8266/"
        logger.info(place)
        self.WS = littlesock(place,logger,buffer=100)
        for k in range(3):
            if len(self.WS.messages):
                break
            time.sleep(0.5)
        else:
            raise Exception("NoResponse")
        self.saylines("135015")
        self.saylines("\x03",1)
        self.saylines("\x03",1)
        self.normal = normal
        if not self.normal:self.saylines("\x01",1)
        else:self.saylines("\x02",1)


    #TODO:测试：查看板载形式
    def check(self):
        self.saylines("import esp32")
        # if error
        #No module
        self.board=32

    def 超频(self):
        if self.board==32:
            self.saylines("import machine;machine.freq(240000000)")
            #esp32默认是该频率
        elif self.board==8266:
            self.saylines("import machine;machine.freq(160000000)")

    def didline(self,p):
        if self.normal:
            return b"\r\n"
        if p:return b"\x04"
        # return b""
        return b"\r\n"
        return b"\x04"


    def saylines(self,p,pure=0):
        if pure:
            self.WS.web.send(bytes(p,"utf-8"))
        else:
            print("saylines" + str(p))
            if type(p) == list:
                for k in p: self.saylines(k)
            else:
                if type(p) == str:
                    if not re.search("\r\n", p) and re.search("\n", p):
                        print("DiDi")
                        p = re.split("\n", p)
                        print("not imlement")
                        return self.saylines(p)


                    p = bytes(p, "utf-8")
                self.WS.web.send(p + self.didline(p))
    on=1
    def turnoff(self,restart=0):
        if not self.on:return 
        if restart==1:
            self.saylines("import machine")
            self.saylines("machine.reset()")
            self.saylines("print('done')")
        self.saylines("\x02", 1)
        if restart==4:self.saylines('\x04', 1)
        # Not for webrepl, so do not do that
        self.WS.__exit__()
        self.on=0
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.turnoff()


#fake
class espweber1():
    def __init__(self,*args):
        print(*args)
        self.WS=littlesock1()
    def saylines(self,s):
        print("said "+s)
    def turnoff(self):
        pass
    def start(self):
        pass

def Pincontroller():
    C.saylines("from machine import Pin;P=Pin(0,Pin.OUT)")
    while 1:
        n = input()
        if n == "1" or n == "0":
            print("said {}".format(n))
            C.saylines("P.value({})".format(n))
        else:
            break
# import sys
# print(sys.argv)
Log = logging.getLogger()
Log.setLevel(logging.INFO)