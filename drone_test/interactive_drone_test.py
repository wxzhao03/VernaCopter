# -*- coding: utf-8 -*-

import time
from threading import Thread
import motioncapture
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils.reset_estimator import reset_estimator

DRONE_URI = 'radio://0/80/2M/E7E7E7E708'  
DRONE_NAME = 'cf8'  
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


class InteractiveDroneController:
    def __init__(self, scf, mocap_wrapper):
        self.cf = scf.cf
        self.commander = self.cf.high_level_commander
        self.mocap_wrapper = mocap_wrapper
        
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        
        self.is_flying = False
        self.step_size = 0.15
        
        self._setup_mocap()
        self._setup_estimator()
        
        print("Waiting for Vicon position")
        time.sleep(2.0)
        
        try:
            for name, obj in self.mocap_wrapper._mc.rigidBodies.items():
                if name == DRONE_NAME:
                    pos = obj.position
                    self.x = pos[0]
                    self.y = pos[1]
                    self.z = pos[2]
                    print(f"Initial position: ({self.x:.3f}, {self.y:.3f}, {self.z:.3f})")
                    break
        except Exception as e:
            print(f"Warning: Cannot get initial position: {e}")
            print("Using (0, 0, 0) as default")
    
    def _setup_mocap(self):
        def send_pose(pose):
            self.cf.extpos.send_extpose(
                pose[0], pose[1], pose[2],
                pose[3].x, pose[3].y, pose[3].z, pose[3].w
            )
        
        self.mocap_wrapper.on_pose = send_pose
    
    def _setup_estimator(self):
        self.cf.param.set_value('locSrv.extQuatStdDev', 8.0e-3)
        self.cf.param.set_value('stabilizer.estimator', '2')
        self.cf.param.set_value('commander.enHighLevel', '1')
        time.sleep(0.5)
        reset_estimator(self.cf)
        
    
    def arm(self):
        try:
            self.cf.platform.send_arming_request(True)
        except Exception:
            pass
        time.sleep(1.0)
    
    def disarm(self):
        try:
            self.cf.platform.send_arming_request(False)
        except Exception:
            pass
    
    def takeoff(self, height=0.3):
        if self.is_flying:
            print("Already flying")
            return
        
        print(f"Taking off to {height}m")
        self.commander.takeoff(height, 2.0)
        time.sleep(2.5)
        self.z = height
        self.is_flying = True
    
    def land(self):
        if not self.is_flying:
            print("Not flying")
            return
        print("Landing")
        self.commander.land(0.0, 2.0)
        time.sleep(2.5)
        self.commander.stop()
        self.is_flying = False
        self.x, self.y, self.z = 0.0, 0.0, 0.0
    
    def emergency_stop(self):
        print("EMERGENCY STOP")
        self.commander.send_stop_setpoint()
        self.cf.commander.send_stop_setpoint()
        self.is_flying = False
    
    def move(self, dx=0, dy=0, dz=0, duration=2):
        if not self.is_flying:
            print("Please takeoff first")
            return
        
        self.x += dx
        self.y += dy
        self.z += dz
        
        print(f"Moving to: ({self.x:.2f}, {self.y:.2f}, {self.z:.2f})")
        self.commander.go_to(self.x, self.y, self.z, yaw=0.0, duration_s=duration)
        time.sleep(duration + 0.5)
    
    def goto_position(self, x, y, z, duration=2.0):
        if not self.is_flying:
            print("Please takeoff first")
            return
        
        self.x = x
        self.y = y
        self.z = max(0.02, min(z, 2.0))
        
        print(f"Going to position: ({self.x:.2f}, {self.y:.2f}, {self.z:.2f})")
        self.commander.go_to(self.x, self.y, self.z, yaw=0.0, duration_s=duration)
        time.sleep(duration + 0.5)
    
    def get_position(self):
        return self.x, self.y, self.z
    
    def show_help(self):
        print("waiting command")
        print("  t  - Takeoff")
        print("  l  - Land")
        print("  e  - EMERGENCY STOP")
        print("  q  - Quit program")
        print("  w  - Move forward  (X+)")
        print("  s  - Move backward (X-)")
        print("  a  - Move left     (Y+)")
        print("  d  - Move right    (Y-)")
        print("  r  - Move up       (Z+)")
        print("  f  - Move down     (Z-)")
        print("  g  - Go to (x,y,z)")
        print("  pt  - Get current position")
        print("  1  - simple test")
        print(f"step size for each movement: {self.step_size}m")
    
    def fly_square(self, size=0.3):

        if not self.is_flying:
            print("Please takeoff first")
            return
        
        start_x, start_y, start_z = self.x, self.y, self.z
        
        waypoints = [
            (start_x + size, start_y, start_z),
            (start_x + size, start_y + size, start_z),
            (start_x, start_y + size, start_z),
            (start_x, start_y, start_z),
        ]
        
        for x, y, z in waypoints:
            self.goto_position(x, y, z, duration=2.0)
        
        print("simple test finish")
    

def main():
    
    cflib.crtp.init_drivers()

    print("test start")
    mocap_wrapper = MocapWrapper(DRONE_NAME, VICON_HOST)
    time.sleep(1.0)
    
    try:
        with SyncCrazyflie(DRONE_URI, cf=Crazyflie(rw_cache='./cache')) as scf:
            
            controller = InteractiveDroneController(scf, mocap_wrapper)
            controller.arm()
            controller.show_help()
            while True:
                try:
                    cmd = input("Enter command: ").strip().lower()
                    
                    if cmd == 'q':
                        if controller.is_flying:
                            print("Land and exit")
                            controller.land()
                        controller.disarm()
                        print("test exit")
                        break
                    
                    elif cmd == 't':
                        controller.takeoff()
                    
                    elif cmd == 'l':
                        controller.land()
                    
                    elif cmd == 'e':
                        controller.emergency_stop()
                    
                    elif cmd == 'w':
                        controller.move(dx=controller.step_size)
                    
                    elif cmd == 's':
                        controller.move(dx=-controller.step_size)
                    
                    elif cmd == 'a':
                        controller.move(dy=controller.step_size)
                    
                    elif cmd == 'd':
                        controller.move(dy=-controller.step_size)
                    
                    elif cmd == 'r':
                        controller.move(dz=0.1)
                    
                    elif cmd == 'f':
                        controller.move(dz=-0.1)
                    
                    elif cmd == 'g':
                        try:
                            x = float(input("  X : "))
                            y = float(input("  Y : "))
                            z = float(input("  Z : "))
                            controller.goto_position(x, y, z)
                        except ValueError:
                            print("wrong position value")
                    
                    elif cmd == 'p':
                        x, y, z = controller.get_position()
                        print(f"Current position: ({x:.3f}, {y:.3f}, {z:.3f})")
                    
                    elif cmd == '1':
                        controller.fly_square()
                    
                    elif cmd == '':
                        continue
                    
                    else:
                        print(f"wrong command: '{cmd}'.")
                
                except KeyboardInterrupt:
                    if controller.is_flying:
                        print("flying be interrupted")
                        controller.emergency_stop()
                    break
                
                except Exception as e:
                    print(f"Error executing command: {e}")
    
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
    
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\ndisconnect with motioncapture")
        mocap_wrapper.close()
        print("\ntest finish")


if __name__ == '__main__':
    main()