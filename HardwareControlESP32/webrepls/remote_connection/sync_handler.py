from webrepl_cli import *
import websocket_helper

import os,re

class Handler:
    ws=None
    def __init__(self,esp,File=None,file=None,passwd = "135015"):
        if File is None:
            File="."
        if type(file)==str:File+=file+"/"
        self.passwd=passwd
        self.file=File
        self.esp=esp
        

    def gethost(self):
        place=self.esp
        try:place=int(place)
        except:
            if not re.search("\.",place):place+=".mshome.net"
            try:place=socket.gethostbyname(place)
            except:place=socket.gethostbyname("ESP_"+place)

        if type(place)==int:
            place=str(place)
            place="192.168.1." + place
        return place


    s=None
    def __enter__(self):
        self.s = socket.socket()
        port = 8266
        host = self.gethost()
        ai = socket.getaddrinfo(host, port)
        addr = ai[0][4]
        print(ai)
        # addr="192.168.137.1"
        print(addr)
        self.s.connect(addr)
        # s = s.makefile("rwb")
        websocket_helper.client_handshake(self.s)
        self.ws = websocket(self.s)
        login(self.ws, self.passwd)
        print("Remote WebREPL version:", get_ver(self.ws))
        return self

    dry_run=False
    def handler(self,local_file,remote_file,op = "put"):
        if self.dry_run:
            print(local_file,remote_file,op)
            return
        # self.ws.ioctl(9, 2)
        if op == "get":
            get_file(self.ws, local_file, remote_file)
        elif op == "put":
            put_file(self.ws, local_file, remote_file)
            
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.s.close()

        
        
import datetime
from pathlib import Path

class File:
    def __init__(self,local="",remote="",time_offset=0):
        self.local_path=Path(local)
        self.remote_path=Path(remote)
        self.time_offset=time_offset
        
    def check(self,):
        self.local_path.exists()
        
    def __str__(self,):
        return str(self.local_path)
    
    def listdir(self,with_remote=True):
        for i in os.listdir(self.local_path):
            # TODO:remote path phrase
            if with_remote:yield os.path.join(self.local_path,i),os.path.join(self.remote_path,i)
            else:yield os.path.join(self.local_path,i)
            
    last_modified_time=datetime.datetime.fromtimestamp(0)
    def listdir_ex(self,with_remote=True,pyonly=True,modified_only=True,log=False):
        def to_ret(local,remote):
            # local,remote=str(local),str(remote)
            if with_remote:
                return local,remote
            return local
        
        # TODO:log
        t=datetime.datetime.now()
        #when running in docker, timezone get quite tricky
        #I do not know why jupyter notebook have different timezone, I don't get it.
        t+=datetime.timedelta(seconds=self.time_offset)
        
        if log and modified_only:print(t)
        
        for local,remote in self.listdir(with_remote=True):
            if pyonly:
                if not os.path.splitext(local)[-1]==".py":
                    continue
            if local.endswith(".ipynb_checkpoints"):
                    continue
            if not modified_only:
                yield to_ret(local,remote)
                continue
            
            local_file_last_modified_time=datetime.datetime.fromtimestamp(os.path.getmtime(local))
            if log:print(local,local_file_last_modified_time)
            if local_file_last_modified_time>self.last_modified_time:
                yield to_ret(local,remote)
            
        self.last_modified_time=t
            
            
            
class MakeHandler(Handler):
    def __init__(self,esp,path="",file=None,passwd = "135015"):
        self.passwd=passwd
        if file is None:
            self.file=File(path)
        else:
            self.file=file
        self.esp=esp

    def sync(self,pyonly=True,log=True):
        for local,remote in self.file.listdir_ex(pyonly=pyonly):
            #TODO:log
            if log:print(local)
            remote=remote.strip(".\\")
            self.handler(local,remote)
        print("done")
        
        
def sync_files(ESP,ff,**kwargs):
    with MakeHandler(ESP, file=ff) as A:
        print(A.file)
        A.sync(**kwargs)
def sync_files_dry(ESP,ff):
    with MakeHandler(ESP, file=ff) as A:
        A.dry_run=True
        print(A.file)
        A.sync()

def hard_turnoff(ESP):
    from repl import espweber
    E=espweber(ESP)
    E.turnoff(1)
    
from repl import espweber
def turnoff(ESP):
    E=espweber(ESP)
    E.turnoff(4)

def syncturnoff(ESP,ff):
    sync_files(ESP,ff)
    turnoff(ESP)
    
# TODO: auto_remove
# same mpy name