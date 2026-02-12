# -*- coding: utf-8 -*-

import time
from threading import Thread
import motioncapture
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils.reset_estimator import reset_estimator

DRONE_URI = 'radio://0/80/2M/E7E7E7E709'  
DRONE_NAME = 'cf9'  
VICON_HOST = '131.155.34.241'  


class MocapWrapper(Thread):
    def __init__(self, body_name: str, host_name: str, system_type: str = "vicon"):
        super().__init__(daemon=True)
        self.body_name = body_name
        self.host_name = host_name
        self.system_type = system_type
        self.on_pose = None  
        self._stay_open = True
        self._mc = None
        self.start()

    def close(self):
        self._stay_open = False
        self.join(timeout=1.0)
        try:
            if self._mc is not None:
                self._mc.disconnect()
        except Exception:
            pass

    def run(self):
        self._mc = motioncapture.connect(self.system_type, {'hostname': self.host_name})
        while self._stay_open:
            self._mc.waitForNextFrame()
            for name, obj in self._mc.rigidBodies.items():
                if name == self.body_name and self.on_pose:
                    pos = obj.position
                    self.on_pose([pos[0], pos[1], pos[2], obj.rotation])


def setup_vicon_positioning(scf, mocap_wrapper):
    cf = scf.cf
    
    def send_pose(pose):
        cf.extpos.send_extpose(
            pose[0], pose[1], pose[2],
            pose[3].x, pose[3].y, pose[3].z, pose[3].w
        )
    
    mocap_wrapper.on_pose = send_pose
    
    cf.param.set_value('locSrv.extQuatStdDev', 8.0e-3)
    cf.param.set_value('stabilizer.estimator', '2')
    cf.param.set_value('commander.enHighLevel', '1')
    time.sleep(0.5)
    
    reset_estimator(cf)
    time.sleep(2.0)


def simple_flight_test(scf):
    cf = scf.cf
    commander = cf.high_level_commander
    
    print("\ntest start")
    
    print("Takeoff to 0.3m")
    commander.takeoff(0.3, 2.0) 
    time.sleep(2.5)
    
    print("Move to (0.3, 0.0, 0.3)")
    commander.go_to(0.3, 0.0, 0.3, yaw=0.0, duration_s=2.0)
    time.sleep(2.5)
    
    print("Move to (0.3, 0.3, 0.3)")
    commander.go_to(0.3, 0.3, 0.3, yaw=0.0, duration_s=2.0)
    time.sleep(2.5)
    
    print("Move to (0.0, 0.3, 0.3)")
    commander.go_to(0.0, 0.3, 0.3, yaw=0.0, duration_s=2.0)
    time.sleep(2.5)
    
    print("Move to (0.0, 0.0, 0.5)")
    commander.go_to(0.0, 0.0, 0.5, yaw=0.0, duration_s=2.0)
    time.sleep(2.5)
    
    print("Move to (0.3, 0.3, 0.3)")
    commander.go_to(0.3, 0.3, 0.3, yaw=0.0, duration_s=2.0)
    time.sleep(2.5)
    
    print("Return to start (0.0, 0.0, 0.3)")
    commander.go_to(0.0, 0.0, 0.3, yaw=0.0, duration_s=2.0)
    time.sleep(2.5)
    
    print("Landing")
    commander.land(0.0, 2.0)
    time.sleep(2.5)
    
    commander.stop()
    print("test finish\n")


def main():
    cflib.crtp.init_drivers()
    
    mocap_wrapper = MocapWrapper(DRONE_NAME, VICON_HOST)
    time.sleep(2.0)
    
    try:
        with SyncCrazyflie(DRONE_URI, cf=Crazyflie(rw_cache='./cache')) as scf:
            print("Connected to drone\n")
            setup_vicon_positioning(scf, mocap_wrapper)
            
            try:
                scf.cf.platform.send_arming_request(True)
                time.sleep(1.0)
            except Exception as e:
                print(f"Arming: {e}\n")
            
            simple_flight_test(scf)
            
            try:
                scf.cf.platform.send_arming_request(False)
            except Exception:
                pass
            
            print("\ntest exit")
    
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        mocap_wrapper.close()
        print("Cdisconnect with motioncapture")


if __name__ == '__main__':
    main()