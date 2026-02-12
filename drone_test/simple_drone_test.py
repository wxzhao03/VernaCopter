# -*- coding: utf-8 -*-
"""
simple_drone_test.py
"""

import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

DRONE_URI = 'radio://0/80/2M/E7E7E7E709'  

def simple_flight_test(scf):
    cf = scf.cf
    commander = cf.high_level_commander

    cf.param.set_value('commander.enHighLevel', '1')
    time.sleep(0.5)
    
    print("test start")
    
    # takeoff
    print("z = 0.3 in 2s")
    commander.takeoff(0.3, 2.0) 
    time.sleep(2.5)
    
    # goto x+
    print("goto 0.3, 0.0, 0.3 in 2s")
    commander.go_to(0.3, 0.0, 0.3, yaw=0.0, duration_s=2.0)
    time.sleep(2.5)
    
    # goto Y+
    print("goto 0.3, 0.3, 0.3 in 2s")
    commander.go_to(0.3, 0.3, 0.3, yaw=0.0, duration_s=2.0)
    time.sleep(2.5)
    
    # back x-0
    print("goto 0.0, 0.3, 0.3 in 2s")
    commander.go_to(0.0, 0.3, 0.3, yaw=0.0, duration_s=2.0)
    time.sleep(2.5)
    
    # back Y-0
    print("goto 0.0, 0.0, 0.3 in 2s")
    commander.go_to(0.0, 0.0, 0.5, yaw=0.0, duration_s=2.0)
    time.sleep(2.5)
    
    # diagonal
    print("goto 0.3, 0.3, 0.3 in 2s")
    commander.go_to(0.3, 0.3, 0.3, yaw=0.0, duration_s=2.0)
    time.sleep(2.5)
    
    # land
    print("land")
    commander.land(0.0, 2.0)
    time.sleep(2.5)
    
    commander.stop()
    print("test finish")


if __name__ == '__main__':
    cflib.crtp.init_drivers()
    
    try:
        with SyncCrazyflie(DRONE_URI, cf=Crazyflie(rw_cache='./cache')) as scf:
            
            time.sleep(3)
            
            simple_flight_test(scf)

    except Exception as e:
        print(f"error: {e}")