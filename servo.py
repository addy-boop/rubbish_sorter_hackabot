from machine import Pin,PWM
import sys
import time

led = Pin(25, Pin.OUT)
servo = PWM(Pin(10))
servo.freq(50)  # Servos run at 50Hz

def set_angle(angle):
    # Convert 0-180 degrees to duty cycle
    min_duty = 1638   # 0 degrees
    max_duty = 8192   # 180 degrees
    duty = int(min_duty + (max_duty - min_duty) * angle / 180)
    servo.duty_u16(duty)
    
while True:
     cmd = sys.stdin.read(1)
     if cmd == 'C':
        set_angle(180)
     elif cmd == 'P':
        set_angle(0)

        
  
 