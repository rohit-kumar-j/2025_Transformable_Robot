#-----------------------------------------------------------------------
# this file contains:
# 1. hexadecimal address of the different registers
# 2. bitposition and bitmasks of the different values of each register
#
# Example:
# the register IOIN has the address 0x06 and the first bit shows
# whether the ENABLE (EN/ENN) Pin is currently HIGH or LOW
#-----------------------------------------------------------------------

#addresses
GCONF           =   0x00
GSTAT           =   0x01
IFCNT           =   0x02
IOIN            =   0x06
IHOLD_IRUN      =   0x10
TSTEP           =   0x12
TCOOLTHRS       =   0x14
SGTHRS          =   0x40
SG_RESULT       =   0x41
MSCNT           =   0x6A
CHOPCONF        =   0x6C
DRVSTATUS       =   0x6F

#GCONF
i_scale_analog      = 1<<0
internal_rsense     = 1<<1
en_spreadcycle      = 1<<2
shaft               = 1<<3
index_otpw          = 1<<4
index_step          = 1<<5
mstep_reg_select    = 1<<7

#GSTAT
reset               = 1<<0
drv_err             = 1<<1
uv_cp               = 1<<2

#CHOPCONF
vsense              = 1<<17
msres0              = 1<<24
msres1              = 1<<25
msres2              = 1<<26
msres3              = 1<<27
intpol              = 1<<28

#IOIN
io_enn              = 1<<0
io_step             = 1<<7
io_spread           = 1<<8
io_dir              = 1<<9

#DRVSTATUS
stst                = 1<<31
stealth             = 1<<30
cs_actual           = 31<<16
t157                = 1<<11
t150                = 1<<10
t143                = 1<<9
t120                = 1<<8
olb                 = 1<<7
ola                 = 1<<6
s2vsb               = 1<<5
s2vsa               = 1<<4
s2gb                = 1<<3
s2ga                = 1<<2
ot                  = 1<<1
otpw                = 1<<0

#IHOLD_IRUN
ihold               = 31<<0
irun                = 31<<8
iholddelay          = 15<<16

#SGTHRS
sgthrs              = 255<<0

#others
mres_256 = 0
mres_128 = 1
mres_64 = 2
mres_32 = 3
mres_16 = 4
mres_8 = 5
mres_4 = 6
mres_2 = 7
mres_1 = 8
#from gpiozero import LED
#from bitstring import BitArray
import time
import sys
import binascii
import struct
import machine
from machine import UART

#-----------------------------------------------------------------------
# TMC_UART
#
# this class is used to communicate with the TMC via UART
# it can be used to change the settings of the TMC.
# like the current or the microsteppingmode
#-----------------------------------------------------------------------
class TMC_UART:

    mtr_id=0
    ser = None
    rFrame  = [0x55, 0, 0, 0  ]
    wFrame  = [0x55, 0, 0, 0 , 0, 0, 0, 0 ]
    communication_pause = 0
    
#-----------------------------------------------------------------------
# constructor
#-----------------------------------------------------------------------
    def __init__(self, uart):
        self.ser = uart
        self.mtr_id=0
        self.ser.init(115200 , bits=8, parity=None, stop=1)
        #self.ser.timeout = 20000/baudrate            # adjust per baud and hardware. Sequential reads without some delay fail.
        self.communication_pause = 500/115200     # adjust per baud and hardware. Sequential reads without some delay fail.
        self.communication_pause = 0.1 

        #self.ser.reset_output_buffer()
        #self.ser.reset_input_buffer()
        
#-----------------------------------------------------------------------
# destructor
#-----------------------------------------------------------------------
    def __del__(self):
        self.ser.close()

#-----------------------------------------------------------------------
# this function calculates the crc8 parity bit
#-----------------------------------------------------------------------
    def compute_crc8_atm(self, datagram, initial_value=0):
        crc = initial_value
        # Iterate bytes in data
        for byte in datagram:
            # Iterate bits in byte
            for _ in range(0, 8):
                if (crc >> 7) ^ (byte & 0x01):
                    crc = ((crc << 1) ^ 0x07) & 0xFF
                else:
                    crc = (crc << 1) & 0xFF
                # Shift to next bit
                byte = byte >> 1
        return crc
    
#-----------------------------------------------------------------------
# reads the registry on the TMC with a given address.
# returns the binary value of that register
#-----------------------------------------------------------------------
    def read_reg(self, reg):
        
        rtn = ""
        #self.ser.reset_output_buffer()
        #self.ser.reset_input_buffer()
        
        self.rFrame[1] = self.mtr_id
        self.rFrame[2] = reg
        self.rFrame[3] = self.compute_crc8_atm(self.rFrame[:-1])

        rt = self.ser.write(bytes(self.rFrame))
        if rt != len(self.rFrame):
            print("TMC2209: Err in write {}".format(__), file=sys.stderr)
            return False
        time.sleep(self.communication_pause)  # adjust per baud and hardware. Sequential reads without some delay fail.
        if self.ser.any():
            rtn = self.ser.read()#read what it self 
        time.sleep(self.communication_pause)  # adjust per baud and hardware. Sequential reads without some delay fail.
        if rtn == None:
            print("TMC2209: Err in read")
            return ""
#         print("received "+str(len(rtn))+" bytes; "+str(len(rtn)*8)+" bits")
        return(rtn[7:11])
#-----------------------------------------------------------------------
# this function tries to read the registry of the TMC 10 times
# if a valid answer is returned, this function returns it as an integer
#-----------------------------------------------------------------------
    def read_int(self, reg):
        tries = 0
        while(True):
            rtn = self.read_reg(reg)
            tries += 1
            if(len(rtn)>=4):
                break
            else:
                print("TMC2209: did not get the expected 4 data bytes. Instead got "+str(len(rtn))+" Bytes")
            if(tries>=10):
                print("TMC2209: after 10 tries not valid answer. exiting")
                print("TMC2209: is Stepper Powersupply switched on ?")
                raise SystemExit
        val = struct.unpack(">i",rtn)[0]
        return(val)

#-----------------------------------------------------------------------
# this function can write a value to the register of the tmc
# 1. use read_int to get the current setting of the TMC
# 2. then modify the settings as wished
# 3. write them back to the driver with this function
#-----------------------------------------------------------------------
    def write_reg(self, reg, val):
        
        #self.ser.reset_output_buffer()
        #self.ser.reset_input_buffer()
        
        self.wFrame[1] = self.mtr_id
        self.wFrame[2] =  reg | 0x80;  # set write bit
        
        self.wFrame[3] = 0xFF & (val>>24)
        self.wFrame[4] = 0xFF & (val>>16)
        self.wFrame[5] = 0xFF & (val>>8)
        self.wFrame[6] = 0xFF & val
        
        self.wFrame[7] = self.compute_crc8_atm(self.wFrame[:-1])

        rtn = self.ser.write(bytes(self.wFrame))
        if rtn != len(self.wFrame):
            print("TMC2209: Err in write {}".format(__), file=sys.stderr)
            return False
        time.sleep(self.communication_pause)

        return(True)

#-----------------------------------------------------------------------
# this function als writes a value to the register of the TMC
# but it also checks if the writing process was successfully by checking
# the InterfaceTransmissionCounter before and after writing
#-----------------------------------------------------------------------
    def write_reg_check(self, reg, val):
        IFCNT           =   0x02

        ifcnt1 = self.read_int(IFCNT)
        self.write_reg(reg, val)
        ifcnt2 = self.read_int(IFCNT)
        ifcnt2 = self.read_int(IFCNT)
        
        if(ifcnt1 >= ifcnt2):
            print("TMC2209: writing not successful!")
            print("reg:{} val:{}", reg, val)
            print("ifcnt:",ifcnt1,ifcnt2)
            return False
        else:
            return True

#-----------------------------------------------------------------------
# this function clear the communication buffers of the Raspberry Pi
#-----------------------------------------------------------------------
    def flushSerialBuffer(self):
        #self.ser.reset_output_buffer()
        #self.ser.reset_input_buffer()
        return

#-----------------------------------------------------------------------
# this sets a specific bit to 1
#-----------------------------------------------------------------------
    def set_bit(self, value, bit):
        return value | (bit)

#-----------------------------------------------------------------------
# this sets a specific bit to 0
#-----------------------------------------------------------------------
    def clear_bit(self, value, bit):
        return value & ~(bit)
