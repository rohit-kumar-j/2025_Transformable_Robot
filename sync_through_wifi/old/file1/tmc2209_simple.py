from machine import Pin as GPIO
import machine
import time
import math

class Direction():
    CCW = 0
    CW = 1

class Loglevel():
    none = 0
    error = 10
    info = 20
    debug = 30
    movement = 40
    all = 100

class MovementAbsRel():
    absolute = 0
    relative = 1

#-----------------------------------------------------------------------
# TMC_2209
#
# this class has two different functions:
# 1. change setting in the TMC-driver via UART
# 2. move the motor via STEP/DIR pins
#-----------------------------------------------------------------------
class TMC_2209:
    
    tmc_uart = None
    _pin_step = -1
    _pin_dir = -1
    _pin_en = -1
    p_pin_step = -1
    p_pin_dir = -1
    p_pin_en = -1
    
    _direction = True

    _stop = False

    _msres = -1
    _stepsPerRevolution = 0
    
    _loglevel = Loglevel.none

    _currentPos = 0                 # current position of stepper in steps
    _targetPos = 0                  # the target position in steps
    _speed = 0.0                    # the current speed in steps per second
    _maxSpeed = 1.0                 # the maximum speed in steps per second
    _maxSpeedHoming = 500           # the maximum speed in steps per second for homing
    _acceleration = 1.0             # the acceleration in steps per second per second
    _accelerationHoming = 10000     # the acceleration in steps per second per second for homing
    _sqrt_twoa = 1.0                # Precomputed sqrt(2*_acceleration)
    _stepInterval = 0               # the current interval between two steps
    _minPulseWidth = 0.001              # minimum allowed pulse with in microseconds
    _lastStepTime = 0               # The last step time in microseconds
    _n = 0                          # step counter
    _c0 = 0                         # Initial step size in microseconds
    _cn = 0                         # Last step size in microseconds
    _cmin = 0                       # Min step size in microseconds based on maxSpeed
    _sg_threshold = 100             # threshold for stallguard
    _movement_abs_rel = MovementAbsRel.absolute
    
    def mean(obj, x):
        a = 0
        for v in x:
           a=a+v
        return(a/len(x))

#-----------------------------------------------------------------------
# constructor
#-----------------------------------------------------------------------
    def __init__(self, pin_step, pin_dir, pin_en):
        self._pin_step = pin_step
        self._pin_dir = pin_dir
        self._pin_en = pin_en
        
        self.p_pin_step = GPIO(self._pin_step, GPIO.OUT)
        self.p_pin_dir = GPIO(self._pin_dir, GPIO.OUT)
        self.p_pin_en = GPIO(self._pin_en, GPIO.OUT)
        self.p_pin_dir(self._direction)


#-----------------------------------------------------------------------
# set whether the movment should be relative or absolute by default.
# See the Enum MovementAbsoluteRelative
#-----------------------------------------------------------------------       
    def setMovementAbsRel(self, movement_abs_rel):
        self._movement_abs_rel = movement_abs_rel

#-----------------------------------------------------------------------
# enables or disables the motor current output
#-----------------------------------------------------------------------
    def setMotorEnabled(self, en):
        if en:
            self.p_pin_en.off()
        else:
            self.p_pin_en.on()
        if(self._loglevel >= Loglevel.info):
            print("TMC2209: Motor output active: {}".format(en))   

    def getCurrentPosition(self):
        return self._currentPos

#-----------------------------------------------------------------------
# overwrites the current motor position in microsteps
#-----------------------------------------------------------------------
    def setCurrentPosition(self, newPos):
        self._currentPos = newPos

#-----------------------------------------------------------------------
# reverses the motor shaft direction
#-----------------------------------------------------------------------
    def reverseDirection_pin(self):
        self._direction = not self._direction
        if self._direction:
            self.p_pin_dir.on()
        else:
            self.p_pin_dir.off()        

#-----------------------------------------------------------------------
# sets the motor shaft direction to the given value: 0 = CCW; 1 = CW
#-----------------------------------------------------------------------
    def setDirection_pin(self, direction):
        self._direction = direction
        if direction:
            self.p_pin_dir.on()
        else:
            self.p_pin_dir.off()
#-----------------------------------------------------------------------
# sets the maximum motor speed in steps per second
#-----------------------------------------------------------------------
    def setMaxSpeed(self, speed):
        if (speed < 0.0):
           speed = -speed
        if (self._maxSpeed != speed):
            self._maxSpeed = speed
            self._cmin = 1000000.0 / speed
            # Recompute _n from current speed and adjust speed if accelerating or cruising
            if (self._n > 0):
                self._n = (self._speed * self._speed) / (2.0 * self._acceleration) # Equation 16
                self.computeNewSpeed()

#-----------------------------------------------------------------------
# returns the maximum motor speed in steps per second
#-----------------------------------------------------------------------
    def getMaxSpeed(self):
        return self._maxSpeed

#-----------------------------------------------------------------------
# sets the motor acceleration/decceleration in steps per sec per sec
#-----------------------------------------------------------------------
    def setAcceleration(self, acceleration):
        if (acceleration == 0.0):
            return
        if (acceleration < 0.0):
          acceleration = -acceleration
        if (self._acceleration != acceleration):
            # Recompute _n per Equation 17
            self._n = self._n * (self._acceleration / acceleration)
            # New c0 per Equation 7, with correction per Equation 15
            self._c0 = 0.676 * math.sqrt(2.0 / acceleration) * 1000000.0 # Equation 15
            self._acceleration = acceleration
            self.computeNewSpeed()

#-----------------------------------------------------------------------
# returns the motor acceleration/decceleration in steps per sec per sec
#-----------------------------------------------------------------------
    def getAcceleration(self):
        return self._acceleration

#-----------------------------------------------------------------------
# stop the current movement
#-----------------------------------------------------------------------
    def stop(self):
        self._stop = True

#-----------------------------------------------------------------------
# runs the motor to the given position.
# with acceleration and deceleration
# blocks the code until finished or stopped from a different thread!
# returns true when the movement if finshed normally and false,
# when the movement was stopped
#-----------------------------------------------------------------------
    def runToPositionSteps(self, steps, movement_abs_rel = None):
        if(movement_abs_rel is not None):
            this_movement_abs_rel = movement_abs_rel
        else:
            this_movement_abs_rel = self._movement_abs_rel

        if(this_movement_abs_rel == MovementAbsRel.relative):
            self._targetPos = self._currentPos + steps
        else:
            self._targetPos = steps

        self._stop = False
        self._stepInterval = 0
        self._speed = 0.0
        self._n = 0
        self.computeNewSpeed()
        #print("speed:", self.computeNewSpeed())
        while (self.run() and not self._stop): #returns false, when target position is reached
            pass
        return not self._stop

#-----------------------------------------------------------------------
# runs the motor to the given position.
# with acceleration and deceleration
# blocks the code until finished!
#-----------------------------------------------------------------------
    def runToPositionRevolutions(self, revolutions, movement_absolute_relative = None):
        return self.runToPositionSteps(round(revolutions * self._stepsPerRevolution), movement_absolute_relative)

#-----------------------------------------------------------------------
# calculates a new speed if a speed was made
# returns true if the target position is reached
# should not be called from outside!
#-----------------------------------------------------------------------
    def run(self):
        if (self.runSpeed()): #returns true, when a step is made
            self.computeNewSpeed()
            #print(self.getStallguard_Result())
            #print(self.getTStep())
        return (self._speed != 0.0 and self.distanceToGo() != 0)

#-----------------------------------------------------------------------
# returns the remaining distance the motor should run
#-----------------------------------------------------------------------
    def distanceToGo(self):
        #print("pos", self._targetPos - self._currentPos)
        return self._targetPos - self._currentPos

#-----------------------------------------------------------------------
# returns the calculated current speed depending on the acceleration
# this code is based on: 
# "Generate stepper-motor speed profiles in real time" by David Austin
#
# https://www.embedded.com/generate-stepper-motor-speed-profiles-in-real-time/
# https://web.archive.org/web/20140705143928/http://fab.cba.mit.edu/classes/MIT/961.09/projects/i0/Stepper_Motor_Speed_Profile.pdf
#-----------------------------------------------------------------------
    def computeNewSpeed(self):
        distanceTo = self.distanceToGo() # +ve is clockwise from curent location     
        stepsToStop = (self._speed * self._speed) / (2.0 * self._acceleration) # Equation 16
        if(self._loglevel >= Loglevel.movement):
            print("TMC2209: distanceTo", distanceTo)
            #print("TMC2209: stepsToStop", stepsToStop)       
        if (distanceTo == 0 and stepsToStop <= 1):
            # We are at the target and its time to stop
            self._stepInterval = 0
            self._speed = 0.0
            self._n = 0
            if(self._loglevel >= Loglevel.movement):
                print("TMC2209: time to stop")
            return
        
        if (distanceTo > 0):
            # We are anticlockwise from the target
            # Need to go clockwise from here, maybe decelerate now
            if (self._n > 0):
                # Currently accelerating, need to decel now? Or maybe going the wrong way?
                if ((stepsToStop >= distanceTo) or self._direction == Direction.CCW):
                    self._n = -stepsToStop # Start deceleration
            elif (self._n < 0):
                # Currently decelerating, need to accel again?
                if ((stepsToStop < distanceTo) and self._direction == Direction.CW):
                    self._n = -self._n # Start accceleration
        elif (distanceTo < 0):
            # We are clockwise from the target
            # Need to go anticlockwise from here, maybe decelerate
            if (self._n > 0):
                # Currently accelerating, need to decel now? Or maybe going the wrong way?
                if ((stepsToStop >= -distanceTo) or self._direction == Direction.CW):
                    self._n = -stepsToStop # Start deceleration
            elif (self._n < 0):
                # Currently decelerating, need to accel again?
                if ((stepsToStop < -distanceTo) and self._direction == Direction.CCW):
                    self._n = -self._n # Start accceleration
        # Need to accelerate or decelerate
        if (self._n == 0):
            # First step from stopped
            self._cn = self._c0
            self.p_pin_step.off()
            #print("TMC2209: distance to: " + str(distanceTo))
            if(distanceTo > 0):
                self.setDirection_pin(1)
                if(self._loglevel >= Loglevel.movement):
                    print("TMC2209: going CW")
            else:
                self.setDirection_pin(0)
                if(self._loglevel >= Loglevel.movement):
                    print("TMC2209: going CCW")
        else:
            # Subsequent step. Works for accel (n is +_ve) and decel (n is -ve).
            self._cn = self._cn - ((2.0 * self._cn) / ((4.0 * self._n) + 1)) # Equation 13
            self._cn = max(self._cn, self._cmin)
        self._n += 1
        self._stepInterval = self._cn
        self._speed = 1000000.0 / self._cn
        if (self._direction == 0):
            self._speed = -self._speed

            
### TODO to asyncio     
            
    def makeAStep(self):
        self.p_pin_step.on()
        # time.sleep_us(1)
        self.p_pin_step.off()
        # time.sleep_us(1)

#-----------------------------------------------------------------------
# this methods does the actual steps with the current speed
#-----------------------------------------------------------------------
    def runSpeed(self):
        # Dont do anything unless we actually have a step interval
        if (not self._stepInterval):
            return False
        
        curtime = time.ticks_us()
        #print("TMC2209: current time: " + str(curtime))
        #print("TMC2209: last st time: " + str(self._lastStepTime))
        #print("TMC2209: _stepInterval: " + str(self._stepInterval))
        
        if (curtime - self._lastStepTime >= self._stepInterval):
            if not self._stop:
                if (self._direction == 1): # Clockwise
                    self._currentPos += 1
                else: # Anticlockwise 
                    self._currentPos -= 1
                self.makeAStep()
                
                self._lastStepTime = curtime # Caution: does not account for costs in step()
                return True
        else:
            return False
            