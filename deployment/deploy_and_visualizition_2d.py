# -*- coding: utf-8 -*-
import time
from threading import Thread

import motioncapture
import numpy as np

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper
from cflib.utils.reset_estimator import reset_estimator


class MocapWrapper(Thread):
    def __init__(self, body_name: str, host_name: str, system_type: str = "vicon"):
        super().__init__(daemon=True)
        self.body_name = body_name
        self.host_name = host_name
        self.system_type = system_type
        self.on_pose = None
        self._stay_open = True
        self._mc = None
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.position_history = [] # Store trajectory for visualization
        self.frame_count = 0              
        self.plot_update_interval = 50
        self.save_interval = 10  
        self.save_counter = 0
        self._collect_data = True
        self.start()

    def close(self):
        self._stay_open = False
        # Give run() loop a chance to exit
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
                    # Update current position from motion capture
                    self.current_position = np.array([pos[0], pos[1], pos[2]])
                    # Collect position data at specified intervals
                    if self._collect_data:
                        self.save_counter += 1
                        if self.save_counter >= self.save_interval:
                            self.position_history.append(self.current_position.copy())
                            self.save_counter = 0
                            # print(f"Vicon [{len(self.position_history):3d}]: " f"x={self.current_position[0]:7.3f}, " f"y={self.current_position[1]:7.3f}, "f"z={self.current_position[2]:7.3f}")
                    
                    if self.on_pose:    
                        self.on_pose([pos[0], pos[1], pos[2], obj.rotation])


def send_extpose_quat(cf, x, y, z, quat, send_full_pose: bool):
    if send_full_pose:
        cf.extpos.send_extpose(x, y, z, quat.x, quat.y, quat.z, quat.w)
    else:
        cf.extpos.send_extpos(x, y, z)


def adjust_orientation_sensitivity(cf, orientation_std_dev: float):
    cf.param.set_value('locSrv.extQuatStdDev', orientation_std_dev)


def activate_kalman_estimator(cf):
    cf.param.set_value('stabilizer.estimator', '2')  # Kalman
    # Enable high-level commander (needed for takeoff/go_to/land)
    cf.param.set_value('commander.enHighLevel', '1')


def activate_mellinger_controller(cf):
    cf.param.set_value('stabilizer.controller', '2')  # Mellinger

# Updated default scale factor
def transform_wp_to_projector(x_old, y_old, scale_xy=0.3):
    return x_old * scale_xy, y_old * scale_xy

# Execute waypoint sequence
def run_sequence(cf, waypoints: str, waypoint_duration: float = 0.75):
    commander = cf.high_level_commander

    if waypoints is None:
        waypoints = np.load("/home/amc/crazyflie-lib-python/examples/mocap/4_waypoints.npy")

    try:
        # Step through every other waypoint (matching your original [::2, :])
        for wp in waypoints.T[::2, :]:
            x, y = transform_wp_to_projector(wp[0], wp[1])
            z = 0.0 * wp[2] / 2 + 0.15 
            # z = wp[2]
            commander.go_to(x, y, z, yaw=0.0, duration_s=waypoint_duration)
            time.sleep(waypoint_duration)
    except Exception:
        # Stop current setpoint stream if something goes wrong mid-flight
        commander.send_stop_setpoint()
        raise
    finally:
        # Land and stop either way
        commander.land(0.0, 2.0)
        time.sleep(2.0)
        commander.stop()


def deploy(
    waypoints = None,
    scenario = None,      
    all_rho = None,
    drone_name: str = 'cf8',
    host_name: str = '131.155.34.241',
    send_full_pose: bool = True,
    orientation_std_dev: float = 8.0e-3,
):
    import threading
    """
    Connects to Crazyflie and Vicon, streams external pose, flies a waypoint sequence,
    and shuts everything down cleanly.
    """
    uri = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E708') # TODO: make this into a variable

    # Drivers once per process
    cflib.crtp.init_drivers()

    # Start mocap streaming thread
    mocap_wrapper = MocapWrapper(drone_name, host_name)

    try:
        with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
            cf = scf.cf

            # Stream mocap poses into the estimator
            mocap_wrapper.on_pose = lambda pose: send_extpose_quat(
                cf, pose[0], pose[1], pose[2], pose[3], send_full_pose
            )

            # Estimator/controller setup
            adjust_orientation_sensitivity(cf, orientation_std_dev)
            activate_kalman_estimator(cf)
            # activate_mellinger_controller(cf)  # uncomment if you want Mellinger

            # Reset estimator to align with ext pose input
            reset_estimator(cf)

            # Arm (platform API present on newer firmwares; fall back to legacy if needed)
            try:
                cf.platform.send_arming_request(True)
            except Exception:
                pass
            time.sleep(1.0)
            # Coordinate calibration: align planned trajectory with actual drone position
            if scenario is not None and waypoints is not None:
                scale_xy = 0.3
                commander = cf.high_level_commander
    
                print("take off and moving drone to starting position")
                commander.takeoff(0.15, 2.0)
                time.sleep(1.0)
    
                start_x, start_y = transform_wp_to_projector(waypoints[0, 0], waypoints[1, 0], scale_xy)
                start_z = 0.15
                commander.go_to(start_x, start_y, start_z, yaw=0.0, duration_s=2.0)
                time.sleep(2.0)

                print("\nCalibrating coordinate offset")
                time.sleep(1.0)
                samples = []
                for i in range(10):
                    samples.append(mocap_wrapper.current_position.copy())
                    time.sleep(0.1)
                actual_position = np.mean(samples, axis=0)
                planned_position = waypoints[:3, 0]
    
                offset_x = (actual_position[0] / scale_xy) - planned_position[0]
                offset_y = (actual_position[1] / scale_xy) - planned_position[1]

                print(f"Planned position:  ({planned_position[0]:.3f}, {planned_position[1]:.3f}, {planned_position[2]:.3f})")
                print(f"Actual position(average value from 10 samples):   ({actual_position[0]:.3f}, {actual_position[1]:.3f}, {actual_position[2]:.3f})")
                print(f"Scaled position:   ({actual_position[0]/scale_xy:.3f}, {actual_position[1]/scale_xy:.3f}, {actual_position[2]:.3f})")
                print(f"Calibrated offset: X={offset_x:.3f}, Y={offset_y:.3f}")
    
                mocap_wrapper.position_history = []

            # Real-time trajectory visualization
            if scenario is not None and waypoints is not None:
                from visuals.visualization import Visualizer
                import matplotlib.pyplot as plt
                from matplotlib.animation import FuncAnimation
                visualizer = Visualizer(waypoints, scenario)
    
                if all_rho is not None:
                    fig, ax = visualizer.visualize_trajectory_rho_gradient_2d(all_rho)
                else:
                    fig, ax = visualizer.visualize_trajectory()
                
                plt.show(block=False)
                mng = plt.get_current_fig_manager()

                try:
                    if hasattr(mng, 'window') and hasattr(mng.window, 'state'):
                        mng.window.state('zoomed')
                        print("✓ Window maximized (Tk)")
                    elif hasattr(mng, 'window') and hasattr(mng.window, 'showMaximized'):
                        mng.window.showMaximized()
                        print("✓ Window maximized (Qt)")
                except Exception as e:
                    print(f"⚠ Could not maximize: {e}")

                fig.canvas.draw()
                fig.canvas.flush_events()
                # Initialize real-time position marker and trajectory line
                realtime_marker, = ax.plot([waypoints[0, 0]], 
                                           [waypoints[1, 0]],  
                                           'bo', markersize=10, 
                                           label='Current Position',
                                           zorder=100)
    
    
                realtime_line, = ax.plot([], [], 'b--', 
                                         linewidth=2, 
                                         label='Real Trajectory',
                                         zorder=99)
    
                ax.legend() 
    
                def update_plot(frame):

                    if len(mocap_wrapper.position_history) > 0:
                       history = np.array(mocap_wrapper.position_history)

                       realtime_line.set_data(history[:, 0]/scale_xy - offset_x, history[:, 1]/scale_xy - offset_y)
            
                       curr = mocap_wrapper.current_position
                       realtime_marker.set_data([curr[0]/scale_xy - offset_x], [curr[1]/scale_xy - offset_y])
                    return realtime_marker, realtime_line
                
                # Initialize real-time position marker and trajectory line
                ani = FuncAnimation(
                    fig, 
                    update_plot, 
                    interval=200,
                    blit=False, 
                    cache_frame_data=False,
                    repeat=False
                )       
                fig.canvas.draw()       
                fig.canvas.flush_events()  
                time.sleep(2)
                
                flight_error = None  
                
                def flight_mission():
                    nonlocal flight_error
                    try:
                        run_sequence(cf, waypoints)
                    except Exception as e:
                        flight_error = e
                        print(f"Flight error: {e}")
                
                flight_thread = threading.Thread(target=flight_mission)
                flight_thread.start()

                while flight_thread.is_alive():
                    plt.pause(0.05)
                
                flight_thread.join()
                
                if flight_error:
                    raise flight_error
                
                mocap_wrapper._collect_data = False
                ani.event_source.stop()
                print(f"Flight complete! Collected {len(mocap_wrapper.position_history)} points")
                
                if len(mocap_wrapper.position_history) > 0:
                    history = np.array(mocap_wrapper.position_history)
                    realtime_line.set_data(history[:, 0]/scale_xy - offset_x, history[:, 1]/scale_xy - offset_y)
                    
                    curr = mocap_wrapper.current_position
                    realtime_marker.set_data([curr[0]/scale_xy - offset_x], [curr[1]/scale_xy - offset_y])
                    
                    fig.canvas.draw()
                    fig.canvas.flush_events()
            else:
                run_sequence(cf, waypoints)

            # Disarm
            try:
                cf.platform.send_arming_request(False)
            except Exception:
                pass

            # Make sure we stop sending setpoints
            cf.commander.send_stop_setpoint()

            if scenario is not None and waypoints is not None:
                input("Press Enter to close the plot")

    finally:
        # Always close mocap thread
        mocap_wrapper.close()
