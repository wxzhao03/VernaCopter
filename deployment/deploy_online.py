## version 1: high level
# # -*- coding: utf-8 -*-
# import time, threading, re
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from threading import Thread

# import motioncapture, cflib.crtp
# from cflib.crazyflie import Crazyflie
# from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
# from cflib.utils import uri_helper
# from cflib.utils.reset_estimator import reset_estimator
# from STL.STL_to_path import STLSolver, drone_dynamics

# REPLAN_INTERVAL = 5
# REPLAN_BUFFER   = 5
# SCALE_XY        = 1.0


# # ── helpers ───────────────────────────────────────────────────────────────────
# def shift_spec(spec_str, offset):
#     def _sub(m):
#         a = max(0, int(m.group(2)) - offset)
#         b = max(0, int(m.group(3)) - offset)
#         return f".{m.group(1)}({a}, {b})"
#     return re.sub(r'\.(eventually|always)\((\d+),\s*(\d+)\)', _sub, spec_str)

# def vicon_to_plan(pos_v, ox, oy):
#     return np.array([pos_v[0]/SCALE_XY - ox, pos_v[1]/SCALE_XY - oy, pos_v[2]])

# def plan_to_vicon(pos_p):
#     return pos_p[0]*SCALE_XY, pos_p[1]*SCALE_XY


# # ── velocity estimator (Vicon finite diff + low-pass) ────────────────────────
# class VelocityEstimator:
#     def __init__(self, alpha=0.7):
#         self.alpha = alpha
#         self._pp, self._pt = None, None
#         self._vf = np.zeros(3)

#     def update(self, pos, t):
#         if self._pp is None:
#             self._pp, self._pt = pos.copy(), t
#             return self._vf.copy()
#         dt = t - self._pt
#         if dt < 1e-6: return self._vf.copy()
#         self._vf = self.alpha*(pos - self._pp)/dt + (1-self.alpha)*self._vf
#         self._pp, self._pt = pos.copy(), t
#         return self._vf.copy()


# # ── mocap thread ──────────────────────────────────────────────────────────────
# class MocapWrapper(Thread):
#     def __init__(self, body_name, host_name):
#         super().__init__(daemon=True)
#         self.body_name = body_name
#         self.host_name = host_name
#         self.on_pose   = None
#         self._open     = True
#         self.pos       = np.zeros(3)
#         self.history   = []
#         self._cnt      = 0
#         self.start()

#     def run(self):
#         mc = motioncapture.connect("vicon", {'hostname': self.host_name})
#         while self._open:
#             mc.waitForNextFrame()
#             for name, obj in mc.rigidBodies.items():
#                 if name == self.body_name:
#                     p = obj.position
#                     self.pos = np.array([p[0], p[1], p[2]])
#                     self._cnt += 1
#                     if self._cnt % 10 == 0:
#                         self.history.append(self.pos.copy())
#                     if self.on_pose:
#                         self.on_pose([p[0], p[1], p[2], obj.rotation])

#     def close(self):
#         self._open = False
#         self.join(timeout=1.0)


# # ── online MPC loop ───────────────────────────────────────────────────────────
# def run_online(cf, wp_v, waypoints, mocap, vel_est, ox, oy,
#                spec_str, objects, dt, max_acc, max_speed,
#                scenario=None, all_rho=None):
#     """
#     waypoints : original planning-coord waypoints (6, N), used for visualiser only
#     wp_v      : vicon-coord waypoints (3, N), used for execution
#     """
#     dyn    = drone_dynamics(dt=dt, max_acc=max_acc)
#     A, B   = dyn.A_tilde, dyn.B_tilde
#     N      = wp_v.shape[1]
#     cur_wp = wp_v.copy()
#     cmd    = cf.high_level_commander

#     lock, pending, running = threading.Lock(), [None], [False]
#     replan_log, rx, ry = [], [], []

#     # visualisation: use original planning-coord waypoints directly
#     if scenario:
#         from visuals.visualization import Visualizer
#         vis = Visualizer(waypoints, scenario)
#         fig, ax = vis.visualize_trajectory_rho_gradient_2d(all_rho) \
#                   if (all_rho is not None and len(all_rho) == waypoints.shape[1]) \
#                   else plt.subplots(figsize=(10,10))
#         ax.set_title('Online MPC Deployment')
#         ln,  = ax.plot([], [], 'b--', lw=2, label='True', zorder=99)
#         dot, = ax.plot([], [], 'bo',  ms=8,               zorder=100)
#         mrk, = ax.plot([], [], 'g^',  ms=10, label='Replan', zorder=101)
#         ax.legend(); plt.show(block=False)
#         fig.canvas.draw(); fig.canvas.flush_events()

#     x_true = np.concatenate([vicon_to_plan(mocap.pos, ox, oy), np.zeros(3)])
#     hist   = [x_true[:3].copy()]

#     def replan(trig, ap, x0):
#         try:
#             T_rem = (N - ap) * dt
#             if T_rem <= dt: return
#             ss  = shift_spec(spec_str, ap)
#             ms  = int(T_rem/dt) - 1
#             ss  = re.sub(r'\.(eventually|always)\((\d+),\s*(\d+)\)',
#                          lambda m: f".{m.group(1)}({min(int(m.group(2)),ms)}, {min(int(m.group(3)),ms)})",
#                          ss)
#             x_new, u_new, rho, _, rt = STLSolver(ss, objects, x0, T_rem)\
#                                         .generate_trajectory(dt, max_acc, max_speed, verbose=False)
#             if x_new is not None and rho > 0:
#                 with lock: pending[0] = (trig, x_new, u_new)
#                 print(f"[Replan t={trig}] rho={rho:.3f} rt={rt:.1f}s")
#             else:
#                 replan_log.append((trig, None, False))
#         except Exception as e:
#             print(f"[Replan t={trig}] {e}")
#             replan_log.append((trig, None, False))
#         finally:
#             with lock: running[0] = False

#     try:
#         for step in range(N):

#             # 1. APPLY
#             with lock: p = pending[0]
#             if p is not None:
#                 trig, x_new, u_new = p
#                 if step == trig + REPLAN_BUFFER:
#                     rem  = N - step
#                     cols = min(x_new.shape[1], rem)
#                     blk  = np.zeros((3, rem))
#                     for i in range(cols):
#                         blk[0,i], blk[1,i] = plan_to_vicon(x_new[:,i])
#                         blk[2,i] = x_new[2,i]
#                     if cols < rem: blk[:,cols:] = blk[:,cols-1:cols]
#                     cur_wp[:,step:] = blk
#                     replan_log.append((trig, step, True))
#                     rx.append(x_true[0]); ry.append(x_true[1])
#                     print(f"[Step {step:3d}] replan applied (trig={trig})")
#                 if step >= trig + REPLAN_BUFFER:
#                     with lock: pending[0] = None

#             # 2. EXECUTE
#             wp = cur_wp[:,step]
#             cmd.go_to(wp[0], wp[1], wp[2], yaw=0.0, duration_s=dt)
#             time.sleep(dt)

#             # 3. SAMPLE
#             pos    = vicon_to_plan(mocap.pos.copy(), ox, oy)
#             vel    = vel_est.update(pos, time.time())
#             x_true = np.concatenate([pos, vel])
#             hist.append(pos.copy())
#             print(f"[Step {step:3d}] pos=({pos[0]:.2f},{pos[1]:.2f}) vel=({vel[0]:.2f},{vel[1]:.2f})")

#             # 4. TRIGGER REPLAN
#             if spec_str and step%REPLAN_INTERVAL==0 and step>0 and step+REPLAN_BUFFER<N:
#                 with lock: busy = running[0]
#                 if not busy:
#                     with lock: running[0] = True
#                     ap = step + REPLAN_BUFFER
#                     xp = x_true.copy()
#                     for i in range(REPLAN_BUFFER - 1):
#                         idx = step + 1 + i
#                         if idx < N:
#                             pc = np.array([cur_wp[0,idx]/SCALE_XY-ox, cur_wp[1,idx]/SCALE_XY-oy, cur_wp[2,idx]])
#                             pn = np.array([cur_wp[0,min(idx+1,N-1)]/SCALE_XY-ox, cur_wp[1,min(idx+1,N-1)]/SCALE_XY-oy, cur_wp[2,min(idx+1,N-1)]])
#                             xp = A@xp + B@((pn-pc)/(dt*dt) - xp[3:6]/dt)
#                     threading.Thread(target=replan, args=(step, ap, xp), daemon=True).start()

#             # 5. VISUALISE
#             if scenario:
#                 ha = np.array(hist)
#                 ln.set_data(ha[:,0], ha[:,1])
#                 dot.set_data([x_true[0]], [x_true[1]])
#                 if rx: mrk.set_data(rx, ry)
#                 fig.canvas.draw(); fig.canvas.flush_events()

#     except Exception as e:
#         cmd.send_stop_setpoint(); raise
#     finally:
#         cmd.land(0.0, 2.0); time.sleep(2.0); cmd.stop()
#         print(f"Replan log: {replan_log}")


# # ── deploy ────────────────────────────────────────────────────────────────────
# def deploy(waypoints, scenario=None, all_rho=None, spec_str=None,
#            dt=0.75, max_acc=10, max_speed=0.5,
#            drone_name='cf11', host_name='10.128.7.250',
#            send_full_pose=True, use_online_mpc=False, vel_alpha=0.7):

#     uri   = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E711')
#     cflib.crtp.init_drivers()
#     mocap = MocapWrapper(drone_name, host_name)

#     try:
#         with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
#             cf  = scf.cf
#             cmd = cf.high_level_commander

#             mocap.on_pose = lambda p: (
#                 cf.extpos.send_extpose(p[0],p[1],p[2],p[3].x,p[3].y,p[3].z,p[3].w)
#                 if send_full_pose else cf.extpos.send_extpos(p[0],p[1],p[2]))

#             cf.param.set_value('locSrv.extQuatStdDev', '8.0e-3')
#             cf.param.set_value('stabilizer.estimator', '2')
#             cf.param.set_value('commander.enHighLevel', '1')
#             reset_estimator(cf)
#             try: cf.platform.send_arming_request(True)
#             except: pass
#             time.sleep(1.0)

#             # takeoff + move to start
#             cmd.takeoff(0.15, 2.0); time.sleep(2.0)
#             cmd.go_to(waypoints[0,0]*SCALE_XY, waypoints[1,0]*SCALE_XY,
#                       0.15, yaw=0.0, duration_s=2.0)
#             time.sleep(2.0)

#             # calibration
#             time.sleep(1.0)
#             samples = []
#             for _ in range(10):
#                 samples.append(mocap.pos.copy()); time.sleep(0.1)
#             actual   = np.mean(samples, axis=0)
#             ox       = actual[0]/SCALE_XY - waypoints[0,0]
#             oy       = actual[1]/SCALE_XY - waypoints[1,0]
#             print(f"Offset: X={ox:.3f}, Y={oy:.3f}")
#             mocap.history = []

#             # vicon waypoints: scale only, no offset
#             wp_v      = np.zeros((3, waypoints.shape[1]))
#             wp_v[0,:] = waypoints[0,:] * SCALE_XY
#             wp_v[1,:] = waypoints[1,:] * SCALE_XY
#             wp_v[2,:] = 0.15

#             if use_online_mpc and spec_str:
#                 vel_est = VelocityEstimator(alpha=vel_alpha)
#                 vel_est.update(vicon_to_plan(mocap.pos, ox, oy), time.time())
#                 run_online(cf, wp_v, waypoints, mocap, vel_est, ox, oy,
#                            spec_str, scenario.objects if scenario else {},
#                            dt, max_acc, max_speed, scenario, all_rho)
#             else:
#                 # non-MPC mode
#                 from visuals.visualization import Visualizer
#                 vis = Visualizer(waypoints, scenario)
#                 fig, ax = vis.visualize_trajectory_rho_gradient_2d(all_rho) \
#                           if all_rho is not None else vis.visualize_trajectory()
#                 plt.show(block=False); fig.canvas.draw(); fig.canvas.flush_events()
#                 mk, = ax.plot([waypoints[0,0]], [waypoints[1,0]], 'bo', ms=10, zorder=100)
#                 ln, = ax.plot([], [], 'b--', lw=2, zorder=99); ax.legend()

#                 def update(_):
#                     if mocap.history:
#                         h = np.array(mocap.history)
#                         ln.set_data(h[:,0]/SCALE_XY-ox, h[:,1]/SCALE_XY-oy)
#                         mk.set_data([mocap.pos[0]/SCALE_XY-ox], [mocap.pos[1]/SCALE_XY-oy])
#                     return mk, ln

#                 ani = FuncAnimation(fig, update, interval=200, blit=False,
#                                     cache_frame_data=False, repeat=False)
#                 fig.canvas.draw(); fig.canvas.flush_events(); time.sleep(2)

#                 err = [None]
#                 def fly():
#                     try:
#                         for i in range(wp_v.shape[1]):
#                             cmd.go_to(wp_v[0,i], wp_v[1,i], wp_v[2,i], yaw=0.0, duration_s=dt)
#                             time.sleep(dt)
#                         cmd.land(0.0, 2.0); time.sleep(2.0); cmd.stop()
#                     except Exception as e: err[0] = e

#                 ft = threading.Thread(target=fly); ft.start()
#                 while ft.is_alive(): plt.pause(0.05)
#                 ft.join()
#                 if err[0]: raise err[0]
#                 ani.event_source.stop()

#             try: cf.platform.send_arming_request(False)
#             except: pass
#             cf.commander.send_stop_setpoint()
#             if scenario: input("Press Enter to close.")

#     finally:
#         mocap.close()










# #version 2: lowlevel
# # -*- coding: utf-8 -*-
# import time, threading, re
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from threading import Thread

# import motioncapture, cflib.crtp
# from cflib.crazyflie import Crazyflie
# from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
# from cflib.utils import uri_helper
# from cflib.utils.reset_estimator import reset_estimator
# from STL.STL_to_path import STLSolver, drone_dynamics

# REPLAN_INTERVAL = 5
# REPLAN_BUFFER   = 5
# SCALE_XY        = 1.0
# SEND_HZ         = 50   # frequency for send_full_state_setpoint


# # ── helpers ───────────────────────────────────────────────────────────────────
# def shift_spec(spec_str, offset):
#     def _sub(m):
#         a = max(0, int(m.group(2)) - offset)
#         b = max(0, int(m.group(3)) - offset)
#         return f".{m.group(1)}({a}, {b})"
#     return re.sub(r'\.(eventually|always)\((\d+),\s*(\d+)\)', _sub, spec_str)

# def vicon_to_plan(pos_v, ox, oy):
#     return np.array([pos_v[0]/SCALE_XY - ox, pos_v[1]/SCALE_XY - oy, pos_v[2]])

# def plan_to_vicon(pos_p):
#     return pos_p[0]*SCALE_XY, pos_p[1]*SCALE_XY


# # ── velocity estimator (Vicon finite diff + low-pass) ────────────────────────
# class VelocityEstimator:
#     def __init__(self, alpha=0.7):
#         self.alpha = alpha
#         self._pp, self._pt = None, None
#         self._vf = np.zeros(3)

#     def update(self, pos, t):
#         if self._pp is None:
#             self._pp, self._pt = pos.copy(), t
#             return self._vf.copy()
#         dt = t - self._pt
#         if dt < 1e-6: return self._vf.copy()
#         self._vf = self.alpha*(pos - self._pp)/dt + (1-self.alpha)*self._vf
#         self._pp, self._pt = pos.copy(), t
#         return self._vf.copy()


# # ── mocap thread ──────────────────────────────────────────────────────────────
# class MocapWrapper(Thread):
#     def __init__(self, body_name, host_name):
#         super().__init__(daemon=True)
#         self.body_name = body_name
#         self.host_name = host_name
#         self.on_pose   = None
#         self._open     = True
#         self.pos       = np.zeros(3)
#         self.history   = []
#         self._cnt      = 0
#         self.start()

#     def run(self):
#         mc = motioncapture.connect("vicon", {'hostname': self.host_name})
#         while self._open:
#             mc.waitForNextFrame()
#             for name, obj in mc.rigidBodies.items():
#                 if name == self.body_name:
#                     p = obj.position
#                     self.pos = np.array([p[0], p[1], p[2]])
#                     self._cnt += 1
#                     if self._cnt % 10 == 0:
#                         self.history.append(self.pos.copy())
#                     if self.on_pose:
#                         self.on_pose([p[0], p[1], p[2], obj.rotation])

#     def close(self):
#         self._open = False
#         self.join(timeout=1.0)


# # ── online MPC loop ───────────────────────────────────────────────────────────
# def run_online(cf, wp_v, waypoints, mocap, vel_est, ox, oy,
#                spec_str, objects, dt, max_acc, max_speed,
#                scenario=None, all_rho=None):
#     """
#     waypoints : original planning-coord waypoints (6, N+1), for velocity reference + visualiser
#     wp_v      : vicon-coord waypoints (3, N+1), for position reference
    
#     Execute step uses send_full_state_setpoint at SEND_HZ for dt seconds,
#     sending planned position + planned velocity at each step.
#     Takeoff and landing remain on high_level_commander.
#     """
#     dyn    = drone_dynamics(dt=dt, max_acc=max_acc)
#     A, B   = dyn.A_tilde, dyn.B_tilde
#     N      = wp_v.shape[1]
#     cur_wp = wp_v.copy()           # (3, N) vicon coords, position only
#     cur_wp_plan = waypoints.copy() # (6, N) planning coords, pos+vel
#     cmd    = cf.high_level_commander

#     lock, pending, running = threading.Lock(), [None], [False]
#     replan_log, rx, ry = [], [], []

#     # visualisation: use original planning-coord waypoints directly
#     if scenario:
#         from visuals.visualization import Visualizer
#         vis = Visualizer(waypoints, scenario)
#         fig, ax = vis.visualize_trajectory_rho_gradient_2d(all_rho) \
#                   if (all_rho is not None and len(all_rho) == waypoints.shape[1]) \
#                   else plt.subplots(figsize=(10,10))
#         ax.set_title('Online MPC Deployment')
#         ln,  = ax.plot([], [], 'b--', lw=2, label='True', zorder=99)
#         dot, = ax.plot([], [], 'bo',  ms=8,               zorder=100)
#         mrk, = ax.plot([], [], 'g^',  ms=10, label='Replan', zorder=101)
#         plan_line, = ax.plot([], [], 'o-', color='orange', ms=4, lw=1, label='Plan steps', zorder=98)
#         ax.legend(); plt.show(block=False)
#         fig.canvas.draw(); fig.canvas.flush_events()

#     x_true = np.concatenate([vicon_to_plan(mocap.pos, ox, oy), np.zeros(3)])
#     hist   = [x_true[:3].copy()]
#     plan_xs, plan_ys = [], []

#     def replan(trig, ap, x0):
#         try:
#             T_rem = (N - ap + 1) * dt
#             if T_rem <= dt: return
#             ss  = shift_spec(spec_str, ap)
#             ms  = int(T_rem/dt) - 1
#             ss  = re.sub(r'\.(eventually|always)\((\d+),\s*(\d+)\)',
#                          lambda m: f".{m.group(1)}({min(int(m.group(2)),ms)}, {min(int(m.group(3)),ms)})",
#                          ss)
#             x_new, u_new, rho, _, rt = STLSolver(ss, objects, x0, T_rem)\
#                                         .generate_trajectory(dt, max_acc, max_speed, verbose=False)
#             if x_new is not None and rho > 0:
#                 with lock: pending[0] = (trig, x_new, u_new)
#                 print(f"[Replan t={trig}] rho={rho:.3f} rt={rt:.1f}s")
#             else:
#                 replan_log.append((trig, None, False))
#         except Exception as e:
#             print(f"[Replan t={trig}] {e}")
#             replan_log.append((trig, None, False))
#         finally:
#             with lock: running[0] = False

#     try:
#         for step in range(N):

#             # 1. APPLY
#             with lock: p = pending[0]
#             if p is not None:
#                 trig, x_new, u_new = p
#                 if step == trig + REPLAN_BUFFER:
#                     rem  = N - step
#                     # 位置更新：从x_new[:,1]开始，跳过x0
#                     cols = min(x_new.shape[1] - 1, rem)
#                     blk  = np.zeros((3, rem))
#                     for i in range(cols):
#                         blk[0,i], blk[1,i] = plan_to_vicon(x_new[:,i+1])
#                         blk[2,i] = x_new[2,i+1]
#                     if cols < rem: blk[:,cols:] = blk[:,cols-1:cols]
#                     cur_wp[:,step:] = blk
#                     # 速度+位置更新：从x_new[:,1]开始，跳过x0
#                     blk_plan = np.zeros((6, rem))
#                     blk_plan[:, :cols] = x_new[:, 1:cols+1]
#                     if cols < rem: blk_plan[:,cols:] = blk_plan[:,cols-1:cols]
#                     blk_plan[3:6, 0] = x_true[3:6]  # 第一步用真实速度过渡
#                     cur_wp_plan[:,step:] = blk_plan
#                     replan_log.append((trig, step, True))
#                     rx.append(x_true[0]); ry.append(x_true[1])
#                     print(f"[Step {step:3d}] replan applied (trig={trig})")
#                 if step >= trig + REPLAN_BUFFER:
#                     with lock: pending[0] = None

#             # 2. EXECUTE: send_full_state_setpoint at SEND_HZ for dt seconds
#             # position from vicon waypoints, velocity from planning waypoints
#             pos_sp  = cur_wp[:, step]           # (x, y, z) in vicon/world coords
#             vel_plan = cur_wp_plan[3:6, step]   # (vx, vy, vz) in planning coords
#             # SCALE_XY=1.0 so planning coords == world coords, no conversion needed

#             send_interval = 1.0 / SEND_HZ
#             t_start = time.time()
#             while time.time() - t_start < dt:
#                 cf.commander.send_full_state_setpoint(
#                     pos_sp,                    # [x, y, z] m
#                     vel_plan,                  # [vx, vy, vz] m/s
#                     np.zeros(3),               # [ax, ay, az] m/s^2
#                     np.array([0., 0., 0., 1.]),# quaternion [qx, qy, qz, qw] (yaw=0)
#                     0., 0., 0.                 # rollrate, pitchrate, yawrate deg/s
#                 )
#                 time.sleep(send_interval)

#             # 3. SAMPLE
#             pos    = vicon_to_plan(mocap.pos.copy(), ox, oy)
#             vel    = vel_est.update(pos, time.time())
#             x_true = np.concatenate([pos, vel])
#             hist.append(pos.copy())
#             print(f"[Step {step:3d}] pos=({pos[0]:.2f},{pos[1]:.2f}) vel=({vel[0]:.2f},{vel[1]:.2f})")

#             # 4. TRIGGER REPLAN
#             if spec_str and step%REPLAN_INTERVAL==0 and step>0 and step+REPLAN_BUFFER<N:
#                 with lock: busy = running[0]
#                 if not busy:
#                     with lock: running[0] = True
#                     ap = step + REPLAN_BUFFER
#                     xp = x_true.copy()
#                     for i in range(REPLAN_BUFFER - 1):
#                         idx = step + 1 + i
#                         if idx < N:
#                             pc = np.array([cur_wp[0,idx]/SCALE_XY-ox, cur_wp[1,idx]/SCALE_XY-oy, cur_wp[2,idx]])
#                             pn = np.array([cur_wp[0,min(idx+1,N-1)]/SCALE_XY-ox, cur_wp[1,min(idx+1,N-1)]/SCALE_XY-oy, cur_wp[2,min(idx+1,N-1)]])
#                             xp = A@xp + B@((pn-pc)/(dt*dt) - xp[3:6]/dt)
#                     threading.Thread(target=replan, args=(step, ap, xp), daemon=True).start()

#             # 5. VISUALISE
#             if scenario:
#                 ha = np.array(hist)
#                 ln.set_data(ha[:,0], ha[:,1])
#                 dot.set_data([x_true[0]], [x_true[1]])
#                 if rx: mrk.set_data(rx, ry)
#                 plan_xs.append(waypoints[0, step])   # 加这行
#                 plan_ys.append(waypoints[1, step])   # 加这行
#                 plan_line.set_data(plan_xs, plan_ys) # 加这行
#                 fig.canvas.draw(); fig.canvas.flush_events()

#     except Exception as e:
#         cf.commander.send_stop_setpoint(); raise
#     finally:
#         # hand back to high level commander before landing
#         cf.param.set_value('commander.enHighLevel', '1')
#         time.sleep(0.1)
#         cmd.land(0.0, 2.0); time.sleep(2.0); cmd.stop()
#         print(f"Replan log: {replan_log}")


# # ── deploy ────────────────────────────────────────────────────────────────────
# def deploy(waypoints, scenario=None, all_rho=None, spec_str=None,
#            dt=0.75, max_acc=10, max_speed=0.5,
#            drone_name='cf11', host_name='10.128.7.250',
#            send_full_pose=True, use_online_mpc=False, vel_alpha=0.7):

#     uri   = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E711')
#     cflib.crtp.init_drivers()
#     mocap = MocapWrapper(drone_name, host_name)

#     try:
#         with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
#             cf  = scf.cf
#             cmd = cf.high_level_commander

#             mocap.on_pose = lambda p: (
#                 cf.extpos.send_extpose(p[0],p[1],p[2],p[3].x,p[3].y,p[3].z,p[3].w)
#                 if send_full_pose else cf.extpos.send_extpos(p[0],p[1],p[2]))

#             cf.param.set_value('locSrv.extQuatStdDev', '8.0e-3')
#             cf.param.set_value('stabilizer.estimator', '2')
#             cf.param.set_value('commander.enHighLevel', '1')
#             reset_estimator(cf)
#             try: cf.platform.send_arming_request(True)
#             except: pass
#             time.sleep(1.0)

#             # takeoff + move to start (high level commander)
#             cmd.takeoff(waypoints[2,0], 2.0); time.sleep(2.0)
#             cmd.go_to(waypoints[0,0]*SCALE_XY, waypoints[1,0]*SCALE_XY,
#                       waypoints[2,0], yaw=0.0, duration_s=2.0)
#             time.sleep(2.0)

#             # calibration
#             time.sleep(1.0)
#             samples = []
#             for _ in range(10):
#                 samples.append(mocap.pos.copy()); time.sleep(0.1)
#             actual   = np.mean(samples, axis=0)
#             ox       = actual[0]/SCALE_XY - waypoints[0,0]
#             oy       = actual[1]/SCALE_XY - waypoints[1,0]
#             print(f"Offset: X={ox:.3f}, Y={oy:.3f}")
#             mocap.history = []

#             # vicon waypoints: scale only, no offset
#             wp_v      = np.zeros((3, waypoints.shape[1]))
#             wp_v[0,:] = waypoints[0,:] * SCALE_XY
#             wp_v[1,:] = waypoints[1,:] * SCALE_XY
#             wp_v[2,:] = waypoints[2,0]  # keep constant z = takeoff height

#             if use_online_mpc and spec_str:
#                 vel_est = VelocityEstimator(alpha=vel_alpha)
#                 vel_est.update(vicon_to_plan(mocap.pos, ox, oy), time.time())
#                 run_online(cf, wp_v, waypoints, mocap, vel_est, ox, oy,
#                            spec_str, scenario.objects if scenario else {},
#                            dt, max_acc, max_speed, scenario, all_rho)
#             else:
#                 # non-MPC mode: mirrors deploy_and_visualization_2d.py
#                 from visuals.visualization import Visualizer
#                 vis = Visualizer(waypoints, scenario)
#                 fig, ax = vis.visualize_trajectory_rho_gradient_2d(all_rho) \
#                           if all_rho is not None else vis.visualize_trajectory()
#                 plt.show(block=False); fig.canvas.draw(); fig.canvas.flush_events()
#                 mk, = ax.plot([waypoints[0,0]], [waypoints[1,0]], 'bo', ms=10, zorder=100)
#                 ln, = ax.plot([], [], 'b--', lw=2, zorder=99); ax.legend()

#                 def update(_):
#                     if mocap.history:
#                         h = np.array(mocap.history)
#                         ln.set_data(h[:,0]/SCALE_XY-ox, h[:,1]/SCALE_XY-oy)
#                         mk.set_data([mocap.pos[0]/SCALE_XY-ox], [mocap.pos[1]/SCALE_XY-oy])
#                     return mk, ln

#                 ani = FuncAnimation(fig, update, interval=200, blit=False,
#                                     cache_frame_data=False, repeat=False)
#                 fig.canvas.draw(); fig.canvas.flush_events(); time.sleep(2)

#                 err = [None]
#                 def fly():
#                     try:
#                         send_interval = 1.0 / SEND_HZ
#                         for i in range(wp_v.shape[1]):
#                             pos_sp  = wp_v[:, i]
#                             vel_sp  = waypoints[3:6, i]
#                             t_start = time.time()
#                             while time.time() - t_start < dt:
#                                 cf.commander.send_full_state_setpoint(
#                                     pos_sp,
#                                     vel_sp,
#                                     np.zeros(3),
#                                     np.array([0., 0., 0., 1.]),
#                                     0., 0., 0.
#                                 )
#                                 time.sleep(send_interval)
#                         cf.param.set_value('commander.enHighLevel', '1')
#                         time.sleep(0.1)
#                         cmd.land(0.0, 2.0); time.sleep(2.0); cmd.stop()
#                     except Exception as e: err[0] = e

#                 ft = threading.Thread(target=fly); ft.start()
#                 while ft.is_alive(): plt.pause(0.05)
#                 ft.join()
#                 if err[0]: raise err[0]
#                 ani.event_source.stop()

#             try: cf.platform.send_arming_request(False)
#             except: pass
#             cf.commander.send_stop_setpoint()
#             if scenario: input("Press Enter to close.")

#     finally:
#         mocap.close()










# ## version 3: low level + no MPC 
# # -*- coding: utf-8 -*-
# import time, threading, re
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from threading import Thread

# import motioncapture, cflib.crtp
# from cflib.crazyflie import Crazyflie
# from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
# from cflib.utils import uri_helper
# from cflib.utils.reset_estimator import reset_estimator
# from STL.STL_to_path import STLSolver, drone_dynamics

# REPLAN_INTERVAL = 5
# REPLAN_BUFFER   = 5
# SCALE_XY        = 1.0
# SEND_HZ         = 50


# # ── helpers ───────────────────────────────────────────────────────────────────
# def shift_spec(spec_str, offset):
#     def _sub(m):
#         a = max(0, int(m.group(2)) - offset)
#         b = max(0, int(m.group(3)) - offset)
#         return f".{m.group(1)}({a}, {b})"
#     return re.sub(r'\.(eventually|always)\((\d+),\s*(\d+)\)', _sub, spec_str)

# def vicon_to_plan(pos_v, ox, oy):
#     return np.array([pos_v[0]/SCALE_XY - ox, pos_v[1]/SCALE_XY - oy, pos_v[2]])

# def plan_to_vicon(pos_p):
#     return pos_p[0]*SCALE_XY, pos_p[1]*SCALE_XY


# # ── velocity estimator (Vicon finite diff + low-pass) ────────────────────────
# class VelocityEstimator:
#     def __init__(self, alpha=0.7):
#         self.alpha = alpha
#         self._pp, self._pt = None, None
#         self._vf = np.zeros(3)

#     def update(self, pos, t):
#         if self._pp is None:
#             self._pp, self._pt = pos.copy(), t
#             return self._vf.copy()
#         dt = t - self._pt
#         if dt < 1e-6: return self._vf.copy()
#         self._vf = self.alpha*(pos - self._pp)/dt + (1-self.alpha)*self._vf
#         self._pp, self._pt = pos.copy(), t
#         return self._vf.copy()


# # ── mocap thread ──────────────────────────────────────────────────────────────
# class MocapWrapper(Thread):
#     def __init__(self, body_name, host_name):
#         super().__init__(daemon=True)
#         self.body_name = body_name
#         self.host_name = host_name
#         self.on_pose   = None
#         self._open     = True
#         self.pos       = np.zeros(3)
#         self.history   = []
#         self._cnt      = 0
#         self.start()

#     def run(self):
#         mc = motioncapture.connect("vicon", {'hostname': self.host_name})
#         while self._open:
#             mc.waitForNextFrame()
#             for name, obj in mc.rigidBodies.items():
#                 if name == self.body_name:
#                     p = obj.position
#                     self.pos = np.array([p[0], p[1], p[2]])
#                     self._cnt += 1
#                     if self._cnt % 10 == 0:
#                         self.history.append(self.pos.copy())
#                     if self.on_pose:
#                         self.on_pose([p[0], p[1], p[2], obj.rotation])

#     def close(self):
#         self._open = False
#         self.join(timeout=1.0)


# def _send_setpoint(cf, pos_sp, vel_sp):
#     """Send one full state setpoint, yaw=0."""
#     cf.commander.send_full_state_setpoint(
#         pos_sp,
#         vel_sp,
#         np.zeros(3),
#         np.array([0., 0., 0., 1.]),
#         0., 0., 0.
#     )


# # ── tracking test: no MPC, step-by-step with full_state_setpoint ─────────────
# def run_tracking_test(cf, waypoints, mocap, vel_est, ox, oy,
#                       dt, scenario=None, all_rho=None):
#     """
#     Execute planned trajectory step by step using send_full_state_setpoint.
#     No replanning. Visualises planned trajectory step-by-step alongside true trajectory.
#     Prints planned vs true position and velocity at each step.
#     """
#     N   = waypoints.shape[1]
#     cmd = cf.high_level_commander

#     # ── visualisation ─────────────────────────────────────────────────────────
#     if scenario:
#         from visuals.visualization import Visualizer
#         vis = Visualizer(waypoints, scenario)
#         if all_rho is not None and len(all_rho) == N:
#             fig, ax = vis.visualize_trajectory_rho_gradient_2d(all_rho)
#         else:
#             fig, ax = plt.subplots(figsize=(10, 10))
#         ax.set_title('Tracking Test (no MPC)')

#         # planned: step-by-step dots (orange)
#         plan_line, = ax.plot([], [], 'o-', color='orange', lw=2,
#                              ms=6, label='Planned', zorder=99)
#         # true: red dashed line like simulation no-MPC
#         true_line, = ax.plot([], [], 'r--', lw=2,
#                              label='True', zorder=100)
#         true_dot,  = ax.plot([], [], 'ro', ms=8, zorder=101)
#         ax.legend(); plt.show(block=False)
#         fig.canvas.draw(); fig.canvas.flush_events()

#     plan_xs, plan_ys = [], []
#     true_hist = [vicon_to_plan(mocap.pos.copy(), ox, oy)[:2]]

#     send_interval = 1.0 / SEND_HZ

#     try:
#         for step in range(N):
#             pos_sp  = np.array([waypoints[0, step]*SCALE_XY,
#                                  waypoints[1, step]*SCALE_XY,
#                                  waypoints[2, step]])
#             vel_sp  = waypoints[3:6, step]

#             # send at SEND_HZ for dt seconds
#             t_start = time.time()
#             while time.time() - t_start < dt:
#                 _send_setpoint(cf, pos_sp, vel_sp)
#                 time.sleep(send_interval)

#             # sample true state
#             pos_true = vicon_to_plan(mocap.pos.copy(), ox, oy)
#             vel_true = vel_est.update(pos_true, time.time())
#             true_hist.append(pos_true[:2].copy())

#             # planned position for this step (planning coords)
#             plan_xs.append(waypoints[0, step])
#             plan_ys.append(waypoints[1, step])

#             # print comparison
#             print(f"[Step {step:3d}] "
#                   f"plan=({waypoints[0,step]:.3f},{waypoints[1,step]:.3f}) "
#                   f"true=({pos_true[0]:.3f},{pos_true[1]:.3f}) | "
#                   f"plan_vel=({vel_sp[0]:.3f},{vel_sp[1]:.3f}) "
#                   f"true_vel=({vel_true[0]:.3f},{vel_true[1]:.3f})")

#             # update visualisation
#             if scenario:
#                 plan_line.set_data(plan_xs, plan_ys)
#                 th = np.array(true_hist)
#                 true_line.set_data(th[:, 0], th[:, 1])
#                 true_dot.set_data([pos_true[0]], [pos_true[1]])
#                 fig.canvas.draw(); fig.canvas.flush_events()

#     except Exception as e:
#         cf.commander.send_stop_setpoint(); raise
#     finally:
#         cf.param.set_value('commander.enHighLevel', '1')
#         time.sleep(0.1)
#         cmd.land(0.0, 2.0); time.sleep(2.0); cmd.stop()


# # ── online MPC loop ───────────────────────────────────────────────────────────
# def run_online(cf, wp_v, waypoints, mocap, vel_est, ox, oy,
#                spec_str, objects, dt, max_acc, max_speed,
#                scenario=None, all_rho=None):

#     dyn    = drone_dynamics(dt=dt, max_acc=max_acc)
#     A, B   = dyn.A_tilde, dyn.B_tilde
#     N      = wp_v.shape[1]
#     cur_wp = wp_v.copy()
#     cur_wp_plan = waypoints.copy()
#     cmd    = cf.high_level_commander

#     lock, pending, running = threading.Lock(), [None], [False]
#     replan_log, rx, ry = [], [], []

#     if scenario:
#         from visuals.visualization import Visualizer
#         vis = Visualizer(waypoints, scenario)
#         fig, ax = vis.visualize_trajectory_rho_gradient_2d(all_rho) \
#                   if (all_rho is not None and len(all_rho) == waypoints.shape[1]) \
#                   else plt.subplots(figsize=(10,10))
#         ax.set_title('Online MPC Deployment')
#         ln,  = ax.plot([], [], 'b--', lw=2, label='True', zorder=99)
#         dot, = ax.plot([], [], 'bo',  ms=8,               zorder=100)
#         mrk, = ax.plot([], [], 'g^',  ms=10, label='Replan', zorder=101)
#         ax.legend(); plt.show(block=False)
#         fig.canvas.draw(); fig.canvas.flush_events()

#     x_true = np.concatenate([vicon_to_plan(mocap.pos, ox, oy), np.zeros(3)])
#     hist   = [x_true[:3].copy()]

#     def replan(trig, ap, x0):
#         try:
#             T_rem = (N - ap + 1) * dt
#             if T_rem <= dt: return
#             ss  = shift_spec(spec_str, ap)
#             ms  = int(T_rem/dt) - 1
#             ss  = re.sub(r'\.(eventually|always)\((\d+),\s*(\d+)\)',
#                          lambda m: f".{m.group(1)}({min(int(m.group(2)),ms)}, {min(int(m.group(3)),ms)})",
#                          ss)
#             x_new, u_new, rho, _, rt = STLSolver(ss, objects, x0, T_rem)\
#                                         .generate_trajectory(dt, max_acc, max_speed, verbose=False)
#             if x_new is not None and rho > 0:
#                 with lock: pending[0] = (trig, x_new, u_new)
#                 print(f"[Replan t={trig}] rho={rho:.3f} rt={rt:.1f}s")
#             else:
#                 replan_log.append((trig, None, False))
#         except Exception as e:
#             print(f"[Replan t={trig}] {e}")
#             replan_log.append((trig, None, False))
#         finally:
#             with lock: running[0] = False

#     try:
#         for step in range(N):

#             # 1. APPLY
#             with lock: p = pending[0]
#             if p is not None:
#                 trig, x_new, u_new = p
#                 if step == trig + REPLAN_BUFFER:
#                     rem  = N - step
#                     cols = min(x_new.shape[1], rem)
#                     blk  = np.zeros((3, rem))
#                     for i in range(cols):
#                         blk[0,i], blk[1,i] = plan_to_vicon(x_new[:,i])
#                         blk[2,i] = x_new[2,i]
#                     if cols < rem: blk[:,cols:] = blk[:,cols-1:cols]
#                     cur_wp[:,step:] = blk
#                     blk_plan = np.zeros((6, rem))
#                     cols_p = min(x_new.shape[1], rem)
#                     blk_plan[:, :cols_p] = x_new[:, :cols_p]
#                     if cols_p < rem: blk_plan[:,cols_p:] = blk_plan[:,cols_p-1:cols_p]
#                     cur_wp_plan[:,step:] = blk_plan
#                     replan_log.append((trig, step, True))
#                     rx.append(x_true[0]); ry.append(x_true[1])
#                     print(f"[Step {step:3d}] replan applied (trig={trig})")
#                 if step >= trig + REPLAN_BUFFER:
#                     with lock: pending[0] = None

#             # 2. EXECUTE
#             pos_sp  = cur_wp[:, step]
#             vel_sp  = cur_wp_plan[3:6, step]

#             send_interval = 1.0 / SEND_HZ
#             t_start = time.time()
#             while time.time() - t_start < dt:
#                 _send_setpoint(cf, pos_sp, vel_sp)
#                 time.sleep(send_interval)

#             # 3. SAMPLE
#             pos    = vicon_to_plan(mocap.pos.copy(), ox, oy)
#             vel    = vel_est.update(pos, time.time())
#             x_true = np.concatenate([pos, vel])
#             hist.append(pos.copy())
#             print(f"[Step {step:3d}] pos=({pos[0]:.2f},{pos[1]:.2f}) vel=({vel[0]:.2f},{vel[1]:.2f})")

#             # 4. TRIGGER REPLAN
#             if spec_str and step%REPLAN_INTERVAL==0 and step>0 and step+REPLAN_BUFFER<N:
#                 with lock: busy = running[0]
#                 if not busy:
#                     with lock: running[0] = True
#                     ap = step + REPLAN_BUFFER
#                     xp = x_true.copy()
#                     for i in range(REPLAN_BUFFER - 1):
#                         idx = step + 1 + i
#                         if idx < N:
#                             pc = np.array([cur_wp[0,idx]/SCALE_XY-ox, cur_wp[1,idx]/SCALE_XY-oy, cur_wp[2,idx]])
#                             pn = np.array([cur_wp[0,min(idx+1,N-1)]/SCALE_XY-ox, cur_wp[1,min(idx+1,N-1)]/SCALE_XY-oy, cur_wp[2,min(idx+1,N-1)]])
#                             xp = A@xp + B@((pn-pc)/(dt*dt) - xp[3:6]/dt)
#                     threading.Thread(target=replan, args=(step, ap, xp), daemon=True).start()

#             # 5. VISUALISE
#             if scenario:
#                 ha = np.array(hist)
#                 ln.set_data(ha[:,0], ha[:,1])
#                 dot.set_data([x_true[0]], [x_true[1]])
#                 if rx: mrk.set_data(rx, ry)
#                 fig.canvas.draw(); fig.canvas.flush_events()

#     except Exception as e:
#         cf.commander.send_stop_setpoint(); raise
#     finally:
#         cf.param.set_value('commander.enHighLevel', '1')
#         time.sleep(0.1)
#         cmd.land(0.0, 2.0); time.sleep(2.0); cmd.stop()
#         print(f"Replan log: {replan_log}")


# # ── deploy ────────────────────────────────────────────────────────────────────
# def deploy(waypoints, scenario=None, all_rho=None, spec_str=None,
#            dt=0.75, max_acc=10, max_speed=0.5,
#            drone_name='cf11', host_name='10.128.7.250',
#            send_full_pose=True, use_online_mpc=False,
#            tracking_test=False, vel_alpha=0.7):
#     """
#     tracking_test=True : run_tracking_test, no MPC, step-by-step full_state_setpoint
#                          with planned vs true comparison print and visualisation
#     use_online_mpc=True : run_online with replanning
#     both False          : original non-MPC mode with FuncAnimation
#     """
#     uri   = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E711')
#     cflib.crtp.init_drivers()
#     mocap = MocapWrapper(drone_name, host_name)

#     try:
#         with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
#             cf  = scf.cf
#             cmd = cf.high_level_commander

#             mocap.on_pose = lambda p: (
#                 cf.extpos.send_extpose(p[0],p[1],p[2],p[3].x,p[3].y,p[3].z,p[3].w)
#                 if send_full_pose else cf.extpos.send_extpos(p[0],p[1],p[2]))

#             cf.param.set_value('locSrv.extQuatStdDev', '8.0e-3')
#             cf.param.set_value('stabilizer.estimator', '2')
#             cf.param.set_value('commander.enHighLevel', '1')
#             reset_estimator(cf)
#             try: cf.platform.send_arming_request(True)
#             except: pass
#             time.sleep(1.0)

#             # takeoff + move to start
#             cmd.takeoff(waypoints[2,0], 2.0); time.sleep(2.0)
#             cmd.go_to(waypoints[0,0]*SCALE_XY, waypoints[1,0]*SCALE_XY,
#                       waypoints[2,0], yaw=0.0, duration_s=2.0)
#             time.sleep(2.0)

#             # calibration
#             time.sleep(1.0)
#             samples = []
#             for _ in range(10):
#                 samples.append(mocap.pos.copy()); time.sleep(0.1)
#             actual   = np.mean(samples, axis=0)
#             ox       = actual[0]/SCALE_XY - waypoints[0,0]
#             oy       = actual[1]/SCALE_XY - waypoints[1,0]
#             print(f"Offset: X={ox:.3f}, Y={oy:.3f}")
#             mocap.history = []

#             # vicon waypoints: scale only, no offset
#             wp_v      = np.zeros((3, waypoints.shape[1]))
#             wp_v[0,:] = waypoints[0,:] * SCALE_XY
#             wp_v[1,:] = waypoints[1,:] * SCALE_XY
#             wp_v[2,:] = waypoints[2,0]

#             vel_est = VelocityEstimator(alpha=vel_alpha)
#             vel_est.update(vicon_to_plan(mocap.pos, ox, oy), time.time())

#             if tracking_test:
#                 run_tracking_test(cf, waypoints, mocap, vel_est, ox, oy,
#                                   dt, scenario, all_rho)

#             elif use_online_mpc and spec_str:
#                 run_online(cf, wp_v, waypoints, mocap, vel_est, ox, oy,
#                            spec_str, scenario.objects if scenario else {},
#                            dt, max_acc, max_speed, scenario, all_rho)

#             else:
#                 # original non-MPC mode
#                 from visuals.visualization import Visualizer
#                 vis = Visualizer(waypoints, scenario)
#                 fig, ax = vis.visualize_trajectory_rho_gradient_2d(all_rho) \
#                           if all_rho is not None else vis.visualize_trajectory()
#                 plt.show(block=False); fig.canvas.draw(); fig.canvas.flush_events()
#                 mk, = ax.plot([waypoints[0,0]], [waypoints[1,0]], 'bo', ms=10, zorder=100)
#                 ln, = ax.plot([], [], 'b--', lw=2, zorder=99); ax.legend()

#                 def update(_):
#                     if mocap.history:
#                         h = np.array(mocap.history)
#                         ln.set_data(h[:,0]/SCALE_XY-ox, h[:,1]/SCALE_XY-oy)
#                         mk.set_data([mocap.pos[0]/SCALE_XY-ox], [mocap.pos[1]/SCALE_XY-oy])
#                     return mk, ln

#                 ani = FuncAnimation(fig, update, interval=200, blit=False,
#                                     cache_frame_data=False, repeat=False)
#                 fig.canvas.draw(); fig.canvas.flush_events(); time.sleep(2)

#                 err = [None]
#                 def fly():
#                     try:
#                         send_interval = 1.0 / SEND_HZ
#                         for i in range(wp_v.shape[1]):
#                             pos_sp = wp_v[:, i]
#                             vel_sp = waypoints[3:6, i]
#                             t_start = time.time()
#                             while time.time() - t_start < dt:
#                                 _send_setpoint(cf, pos_sp, vel_sp)
#                                 time.sleep(send_interval)
#                         cf.param.set_value('commander.enHighLevel', '1')
#                         time.sleep(0.1)
#                         cmd.land(0.0, 2.0); time.sleep(2.0); cmd.stop()
#                     except Exception as e: err[0] = e

#                 ft = threading.Thread(target=fly); ft.start()
#                 while ft.is_alive(): plt.pause(0.05)
#                 ft.join()
#                 if err[0]: raise err[0]
#                 ani.event_source.stop()

#             try: cf.platform.send_arming_request(False)
#             except: pass
#             cf.commander.send_stop_setpoint()
#             if scenario: input("Press Enter to close.")

#     finally:
#         mocap.close()
































#version:new same with simulation two mode
# -*- coding: utf-8 -*-
import time, threading, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from threading import Thread

import motioncapture, cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper
from cflib.utils.reset_estimator import reset_estimator
from STL.STL_to_path import STLSolver, drone_dynamics

REPLAN_INTERVAL = 3
REPLAN_BUFFER   = 3
SCALE_XY        = 1.0
SEND_HZ         = 50


# ── helpers ───────────────────────────────────────────────────────────────────
def shift_spec(spec_str, step_offset):
    def replace_window(match):
        method = match.group(1)
        a = max(0, int(match.group(2)) - step_offset)
        b = max(0, int(match.group(3)) - step_offset)
        return f".{method}({a}, {b})"
    return re.sub(r'\.(eventually|always)\((\d+),\s*(\d+)\)', replace_window, spec_str)

def vicon_to_plan(pos_v, ox, oy):
    return np.array([pos_v[0]/SCALE_XY - ox, pos_v[1]/SCALE_XY - oy, pos_v[2]])

def plan_to_vicon(pos_p):
    return pos_p[0]*SCALE_XY, pos_p[1]*SCALE_XY


# ── velocity estimator ────────────────────────────────────────────────────────
class VelocityEstimator:
    def __init__(self, alpha=0.7):
        self.alpha = alpha
        self._pp, self._pt = None, None
        self._vf = np.zeros(3)

    def update(self, pos, t):
        if self._pp is None:
            self._pp, self._pt = pos.copy(), t
            return self._vf.copy()
        dt = t - self._pt
        if dt < 1e-6:
            return self._vf.copy()
        self._vf = self.alpha*(pos - self._pp)/dt + (1-self.alpha)*self._vf
        self._pp, self._pt = pos.copy(), t
        return self._vf.copy()


# ── mocap thread ──────────────────────────────────────────────────────────────
class MocapWrapper(Thread):
    def __init__(self, body_name, host_name):
        super().__init__(daemon=True)
        self.body_name = body_name
        self.host_name = host_name
        self.on_pose   = None
        self._open     = True
        self.pos       = np.zeros(3)
        self.history   = []
        self._cnt      = 0
        self.start()

    def run(self):
        mc = motioncapture.connect("vicon", {'hostname': self.host_name})
        while self._open:
            mc.waitForNextFrame()
            for name, obj in mc.rigidBodies.items():
                if name == self.body_name:
                    p = obj.position
                    self.pos = np.array([p[0], p[1], p[2]])
                    self._cnt += 1
                    if self._cnt % 10 == 0:
                        self.history.append(self.pos.copy())
                    if self.on_pose:
                        self.on_pose([p[0], p[1], p[2], obj.rotation])

    def close(self):
        self._open = False
        self.join(timeout=1.0)


# ── online MPC loop ───────────────────────────────────────────────────────────
def run_online(cf, waypoints, all_u, mocap, vel_est, ox, oy,
               spec_str, scenario, dt, max_acc, max_speed,
               all_rho=None,
               spec_str_phase2=None,
               switch_step=None,
               use_voice=False):
    """
    waypoints   : (6, N) planning-coord, pos+vel
    all_u       : (3, N) planning-coord controls
    """
    dyn             = drone_dynamics(dt=dt, max_acc=max_acc)
    A_tilde         = dyn.A_tilde
    B_tilde         = dyn.B_tilde
    N_total         = all_u.shape[1]
    scenario_objects= scenario.objects if scenario is not None else {}

    current_u    = all_u.copy()
    spec_current = [spec_str]

    # vicon-coord position waypoints (3, N): x_v = x_plan * SCALE_XY + offset
    # built once here, updated on replan accept
    cur_wp      = np.zeros((3, N_total))
    cur_wp[0,:] = waypoints[0,:] * SCALE_XY   # offset added at send time via ox/oy
    cur_wp[1,:] = waypoints[1,:] * SCALE_XY
    cur_wp[2,:] = waypoints[2,0]
    # planning-coord pos+vel (6, N): for velocity feedforward
    cur_wp_plan = waypoints.copy()

    lock           = threading.Lock()
    new_trajectory = [None]
    replan_running = [False]
    replan_log     = []
    user_decision  = [None]   # None / 'waiting' / 'accepted' / 'rejected'

    active_plan_data  = [waypoints]
    active_plan_start = [0]
    is_offline        = [True]
    history_lines     = []
    accept_pts        = []
    plan_tracking     = [True]

    # ── voice ─────────────────────────────────────────────────────────────────
    if use_voice:
        try:
            from LLM.voice_openai import VoiceOpenAI
            _voice_available = True
            print("[Voice] Voice interaction enabled.")
        except Exception as e:
            _voice_available = False
            print(f"[Voice] Could not load voice module ({e}), keyboard only.")
    else:
        _voice_available = False

    # ── visualisation ─────────────────────────────────────────────────────────
    from visuals.visualization import Visualizer
    visualizer = Visualizer(waypoints, scenario)

    if all_rho is not None and len(all_rho) == waypoints.shape[1]:
        fig, ax = visualizer.visualize_trajectory_rho_gradient_2d(all_rho)
    else:
        fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_title('Online MPC Deployment')

    plan_line,      = ax.plot([], [], 'o-', color='orange', ms=4, lw=1.5,
                              label='Planned (offline)', zorder=98)
    real_line,      = ax.plot([], [], 'b--', lw=2, label='True trajectory', zorder=99)
    real_dot,       = ax.plot([], [], 'bo',  ms=8,                          zorder=100)
    active_line,    = ax.plot([], [], '-',   color='purple', lw=2,
                              label='Current plan', zorder=101)
    candidate_line, = ax.plot([], [], '--',  color='purple', lw=2,
                              label='Candidate new path', zorder=102)
    branch_marker,  = ax.plot([], [], 'o',   color='purple', ms=10,
                              label='Branch point', zorder=103)
    accept_marker,  = ax.plot([], [], '^',   color='purple', ms=12,
                              label='Accepted', zorder=104)

    plan_xs = [waypoints[0, 0]]
    plan_ys = [waypoints[1, 0]]
    ax.legend(loc='upper left', fontsize=8)
    active_line.set_data(waypoints[0, :], waypoints[1, :])

    # prompt / timer bar
    prompt_ax = fig.add_axes([0.05, 0.01, 0.9, 0.05])
    prompt_ax.set_xlim(0, 1)
    prompt_ax.set_ylim(0, 1)
    prompt_ax.axis('off')
    prompt_text_obj = prompt_ax.text(
        0.5, 0.95, '',
        ha='center', va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightyellow',
                  alpha=0.9, edgecolor='gray'),
    )
    timer_bg_patch  = Rectangle((0.0, 0.05), 1.0, 0.25,
                                 transform=prompt_ax.transAxes,
                                 facecolor='lightgray', edgecolor='none')
    timer_bar_patch = Rectangle((0.0, 0.05), 1.0, 0.25,
                                 transform=prompt_ax.transAxes,
                                 facecolor='orange', edgecolor='none')
    prompt_ax.add_patch(timer_bg_patch)
    prompt_ax.add_patch(timer_bar_patch)
    prompt_ax.set_visible(False)

    def on_key(event):
        if user_decision[0] == 'waiting':
            if event.key == 'enter':
                user_decision[0] = 'accepted'
                print("\n[Keyboard] Accepted.")
            elif event.key == ' ':
                user_decision[0] = 'rejected'
                print("\n[Keyboard] Rejected.")

    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show(block=False)
    try:
        plt.get_current_fig_manager().window.showMaximized()
    except Exception:
        pass
    fig.canvas.draw()
    fig.canvas.flush_events()

    # ── background replanning ─────────────────────────────────────────────────
    def replan(trigger_step, apply_step, x0_replan, spec_switched):
        try:
            T_remaining = (N_total - apply_step - 1) * dt
            if T_remaining <= dt:
                print(f"[Replan t={trigger_step}] No time remaining, skipping.")
                return

            shifted_spec = shift_spec(spec_current[0], apply_step)
            max_step = int(T_remaining / dt) - 1

            def clamp_window(match):
                method = match.group(1)
                a = min(int(match.group(2)), max_step)
                b = min(int(match.group(3)), max_step)
                return f".{method}({a}, {b})"
            shifted_spec = re.sub(
                r'\.(eventually|always)\((\d+),\s*(\d+)\)',
                clamp_window, shifted_spec
            )

            solver = STLSolver(shifted_spec, scenario_objects, x0_replan, T_remaining)
            x_new, u_new, rho, _, runtime = solver.generate_trajectory(
                dt, max_acc, max_speed, verbose=False
            )

            if x_new is not None and rho > 0:
                # store result only — NO canvas ops here, main thread handles display
                path_msg = "New path available" if spec_switched else "Safer path available"
                with lock:
                    new_trajectory[0] = (trigger_step, x_new, u_new, rho,
                                         x0_replan.copy(), apply_step, path_msg)
                    user_decision[0] = 'accepted'  # AUTO-ACCEPT for testing
                print(f"[Replan t={trigger_step}] rho={rho:.3f} rt={runtime:.2f}s "
                      f"→ waiting for user (apply at step {apply_step})")

                # optional voice
                if _voice_available:
                    def voice_listen():
                        try:
                            from LLM.voice_openai import VoiceOpenAI
                            voice = VoiceOpenAI()
                            print("[Voice] 🎤 Say YES or NO...")
                            while user_decision[0] == 'waiting':
                                frames = voice.record_audio(duration=2, silence_threshold=2000,
                                                            silence_duration=0.5)
                                audio_np = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32)
                                if np.sqrt(np.mean(audio_np**2)) < 500:
                                    print("[Voice] No speech, listening again...")
                                    continue
                                import tempfile, os
                                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                                voice.save_audio(frames, tmp.name); tmp.close()
                                text = voice.transcribe(tmp.name).lower(); os.unlink(tmp.name)
                                print(f"[Voice] You said: '{text}'")
                                if any(w in text for w in ['yes','yeah','sure','okay','yep','accept']):
                                    if user_decision[0] == 'waiting':
                                        user_decision[0] = 'accepted'
                                        print("[Voice] Accepted.")
                                    break
                                elif any(w in text for w in ['no','nope','reject','cancel']):
                                    if user_decision[0] == 'waiting':
                                        user_decision[0] = 'rejected'
                                        print("[Voice] Rejected.")
                                    break
                                else:
                                    print("[Voice] Please say YES or NO.")
                            voice.close()
                        except Exception as e:
                            print(f"[Voice] Error: {e}, use keyboard.")
                    threading.Thread(target=voice_listen, daemon=True).start()

            else:
                print(f"[Replan t={trigger_step}] Failed (rho={rho}), keeping current.")
                replan_log.append((trigger_step, None, False))

        except Exception as e:
            print(f"[Replan t={trigger_step}] Exception: {e}")
            replan_log.append((trigger_step, None, False))
        finally:
            with lock:
                replan_running[0] = False

    # ── state ─────────────────────────────────────────────────────────────────
    x_true = np.concatenate([vicon_to_plan(mocap.pos, ox, oy), np.zeros(3)])
    position_history = [x_true[:3].copy()]

    send_interval = 1.0 / SEND_HZ

    try:
        for step in range(N_total):

            # ── 1. APPLY ──────────────────────────────────────────────────────
            with lock:
                pending = new_trajectory[0]

            if pending is not None:
                trigger_step, x_new, u_new, rho, x0_rp, apply_step_stored, path_msg = pending
                expected = trigger_step + REPLAN_BUFFER

                if step < expected:
                    if user_decision[0] == 'waiting':
                        fraction = (expected - step) / REPLAN_BUFFER
                        timer_bar_patch.set_width(fraction)

                elif step == expected:
                    decision = user_decision[0]
                    if decision == 'accepted':
                        # apply u_new (consistent with simulation)
                        steps_remaining = N_total - step
                        cols = min(u_new.shape[1], steps_remaining)
                        new_u_block = np.zeros((3, steps_remaining))
                        new_u_block[:, :cols] = u_new[:, :cols]
                        if cols < steps_remaining:
                            new_u_block[:, cols:] = u_new[:, -1:]
                        current_u[:, step:] = new_u_block

                        # update active plan visuals
                        prev_data  = active_plan_data[0]
                        prev_start = active_plan_start[0]
                        seg_len    = step - prev_start
                        if seg_len > 0:
                            if is_offline[0]:
                                seg_end = min(seg_len, prev_data.shape[1])
                                seg_xs, seg_ys = prev_data[0, :seg_end], prev_data[1, :seg_end]
                            else:
                                seg_end = min(seg_len + 1, prev_data.shape[1])
                                seg_xs, seg_ys = prev_data[0, 0:seg_end], prev_data[1, 0:seg_end]
                            if len(seg_xs) > 0:
                                hl, = ax.plot(seg_xs, seg_ys, '-', color='purple', lw=2, zorder=101)
                                history_lines.append(hl)

                        active_plan_data[0]  = x_new
                        active_plan_start[0] = step
                        is_offline[0]        = False
                        plan_tracking[0]     = False
                        active_line.set_data(x_new[0, 0:-1], x_new[1, 0:-1])
                        accept_pts.append((x_new[0, 0], x_new[1, 0]))
                        replan_log.append((trigger_step, step, True))
                        print(f"[Step {step:3d}] Replan APPLIED (trig={trigger_step})")

                        # update cur_wp (vicon coords) and cur_wp_plan (planning coords)
                        # x_new[:,0] is x0 (current state), positions start from x_new[:,1]
                        rem = N_total - step
                        blk_v    = np.zeros((3, rem))
                        blk_plan = np.zeros((6, rem))
                        for i in range(cols):
                            xi = i + 1  # skip x0
                            src = x_new[:, xi] if xi < x_new.shape[1] else x_new[:, -1]
                            blk_v[0, i], blk_v[1, i] = plan_to_vicon(src[:2])
                            blk_v[2, i] = src[2]
                            blk_plan[:, i] = src
                        if cols < rem:
                            blk_v[:, cols:]    = blk_v[:, cols-1:cols]
                            blk_plan[:, cols:] = blk_plan[:, cols-1:cols]
                        blk_plan[3:6, 0] = x_true[3:6]  # smooth velocity transition
                        cur_wp[:, step:]      = blk_v
                        cur_wp_plan[:, step:] = blk_plan
                    else:
                        reason = 'timeout' if decision == 'waiting' else 'rejected'
                        replan_log.append((trigger_step, step, False))
                        print(f"[Step {step:3d}] Replan {reason} (trig={trigger_step}), keeping current.")

                    candidate_line.set_data([], [])
                    branch_marker.set_data([], [])
                    prompt_text_obj.set_text('')
                    timer_bar_patch.set_width(0)
                    prompt_ax.set_visible(False)
                    user_decision[0] = None
                    with lock:
                        new_trajectory[0] = None

                else:
                    # expired
                    candidate_line.set_data([], [])
                    branch_marker.set_data([], [])
                    prompt_text_obj.set_text('')
                    timer_bar_patch.set_width(0)
                    prompt_ax.set_visible(False)
                    user_decision[0] = None
                    with lock:
                        new_trajectory[0] = None
                    print(f"[Step {step:3d}] Replan expired (trig={trigger_step}), discarding.")

            # ── 2. EXECUTE (version 2 logic) ─────────────────────────────────
            # position from cur_wp (vicon/world coords, no offset needed)
            # velocity feedforward from cur_wp_plan (planning coords, same as world when SCALE_XY=1)
            pos_sp = cur_wp[:, step]
            vel_sp = cur_wp_plan[3:6, step]

            t_start = time.time()
            while time.time() - t_start < dt:
                cf.commander.send_full_state_setpoint(
                    pos_sp,
                    vel_sp,
                    np.zeros(3),
                    np.array([0., 0., 0., 1.]),
                    0., 0., 0.
                )
                time.sleep(send_interval)

            # ── 3. SAMPLE ─────────────────────────────────────────────────────
            pos  = vicon_to_plan(mocap.pos.copy(), ox, oy)
            vel  = vel_est.update(pos, time.time())
            x_true = np.concatenate([pos, vel])
            position_history.append(pos.copy())
            print(f"[Step {step:3d}] pos=({pos[0]:.3f},{pos[1]:.3f}) vel=({vel[0]:.3f},{vel[1]:.3f})")

            # ── 4. SPEC SWITCH ────────────────────────────────────────────────
            if switch_step is not None and step == switch_step and spec_str_phase2 is not None:
                spec_current[0] = spec_str_phase2
                print(f"[Step {step:3d}] *** Spec switched to phase 2 ***")

            # ── 5. TRIGGER REPLAN ─────────────────────────────────────────────
            if (spec_current[0] is not None
                    and step % REPLAN_INTERVAL == 0
                    and step > 0
                    and step + REPLAN_BUFFER <= N_total - 2):
                with lock:
                    already_running = replan_running[0]
                if not already_running:
                    with lock:
                        replan_running[0] = True
                    # clear old candidate
                    candidate_line.set_data([], [])
                    branch_marker.set_data([], [])
                    prompt_text_obj.set_text('')
                    timer_bar_patch.set_width(0)
                    prompt_ax.set_visible(False)
                    user_decision[0] = None

                    # predict state at apply_step using current_u (matches simulation)
                    apply_step  = step + REPLAN_BUFFER
                    x_predicted = x_true.copy()
                    for i in range(REPLAN_BUFFER - 1):
                        idx = step + 1 + i
                        if idx < N_total:
                            x_predicted = A_tilde @ x_predicted + B_tilde @ current_u[:, idx]

                    x0_replan = x_predicted.copy()
                    print(f"[Step {step:3d}] Replanning triggered. "
                          f"Predicted apply pos: ({x0_replan[0]:.3f}, {x0_replan[1]:.3f})")

                    threading.Thread(
                        target=replan,
                        args=(step, apply_step, x0_replan, spec_current[0] != spec_str),
                        daemon=True
                    ).start()

            # ── 6. VISUALISE ──────────────────────────────────────────────────
            if step < N_total - 1:
                plan_xs.append(waypoints[0, step + 1])
                plan_ys.append(waypoints[1, step + 1])
                plan_line.set_data(plan_xs, plan_ys)

            if plan_tracking[0]:
                history_arr = np.array(position_history)
                real_line.set_data(history_arr[:, 0], history_arr[:, 1])
                real_dot.set_data([x_true[0]], [x_true[1]])
            else:
                history_arr = np.array(position_history[:-1]) if len(position_history) > 1 else np.array(position_history)
                real_line.set_data(history_arr[:, 0], history_arr[:, 1])
                cur_pos = np.array(position_history[-2]) if len(position_history) > 1 else np.array(position_history[-1])
                real_dot.set_data([cur_pos[0]], [cur_pos[1]])

            if accept_pts:
                accept_marker.set_data([p[0] for p in accept_pts], [p[1] for p in accept_pts])

            # show candidate path + prompt when replan result is ready (main thread only)
            with lock:
                disp = new_trajectory[0]
            if disp is not None and user_decision[0] == 'waiting':
                _, x_disp, _, rho_disp, x0_disp, ap_disp, msg_disp = disp
                candidate_line.set_data(x_disp[0, 0:-1], x_disp[1, 0:-1])
                branch_marker.set_data([x0_disp[0]], [x0_disp[1]])
                prompt_ax.set_visible(True)
                prompt_text_obj.set_text(
                    f"{msg_disp} (rho={rho_disp:.3f})    "
                    f"[Enter] Accept   [Space] Reject    "
                    f"(auto-reject at step {ap_disp})"
                )
                expected_here = disp[0] + REPLAN_BUFFER
                fraction = max(0.0, (expected_here - step) / REPLAN_BUFFER)
                timer_bar_patch.set_width(fraction)

            fig.canvas.draw()
            fig.canvas.flush_events()

    except Exception as e:
        cf.commander.send_stop_setpoint()
        raise
    finally:
        cf.param.set_value('commander.enHighLevel', '1')
        time.sleep(0.1)
        cf.high_level_commander.land(0.0, 2.0)
        time.sleep(2.0)
        cf.high_level_commander.stop()
        print(f"Replan log: {replan_log}")

    print(f"\n{'='*60}\nFlight complete.\nReplan log: {replan_log}\n{'='*60}\n")
    input("Press Enter to close.")
    plt.close('all')
    return position_history, replan_log


# ── deploy ────────────────────────────────────────────────────────────────────
def deploy(waypoints, all_u, scenario=None, all_rho=None, spec_str=None,
           spec_str_phase2=None, switch_step=None,
           dt=0.75, max_acc=10, max_speed=0.5,
           drone_name='cf17', host_name='10.128.7.250',
           send_full_pose=True, use_online_mpc=False,
           use_voice=False, vel_alpha=0.7):
    """
    waypoints     : (6, N) planning-coord pos+vel
    all_u         : (3, N) planning-coord controls
    use_online_mpc: if True, run online MPC with accept/reject UI
    use_voice     : if True, also listen for voice YES/NO
    spec_str_phase2 / switch_step: optional mid-flight spec switch
    """
    uri   = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E711')
    cflib.crtp.init_drivers()
    mocap = MocapWrapper(drone_name, host_name)

    try:
        with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
            cf  = scf.cf
            cmd = cf.high_level_commander

            mocap.on_pose = lambda p: (
                cf.extpos.send_extpose(p[0], p[1], p[2], p[3].x, p[3].y, p[3].z, p[3].w)
                if send_full_pose else cf.extpos.send_extpos(p[0], p[1], p[2]))

            cf.param.set_value('locSrv.extQuatStdDev', '8.0e-3')
            cf.param.set_value('stabilizer.estimator', '2')
            cf.param.set_value('commander.enHighLevel', '1')
            reset_estimator(cf)
            try:
                cf.platform.send_arming_request(True)
            except Exception:
                pass
            time.sleep(1.0)

            # takeoff + move to start
            cmd.takeoff(waypoints[2, 0], 2.0)
            time.sleep(2.0)
            cmd.go_to(waypoints[0, 0]*SCALE_XY, waypoints[1, 0]*SCALE_XY,
                      waypoints[2, 0], yaw=0.0, duration_s=2.0)
            time.sleep(2.0)

            # calibration: compute vicon→plan offset
            time.sleep(1.0)
            samples = []
            for _ in range(10):
                samples.append(mocap.pos.copy())
                time.sleep(0.1)
            actual = np.mean(samples, axis=0)
            ox = actual[0]/SCALE_XY - waypoints[0, 0]
            oy = actual[1]/SCALE_XY - waypoints[1, 0]
            print(f"Offset: X={ox:.3f}, Y={oy:.3f}")
            mocap.history = []

            vel_est = VelocityEstimator(alpha=vel_alpha)
            vel_est.update(vicon_to_plan(mocap.pos, ox, oy), time.time())

            if use_online_mpc and spec_str:
                run_online(cf, waypoints, all_u, mocap, vel_est, ox, oy,
                           spec_str, scenario,
                           dt, max_acc, max_speed,
                           all_rho=all_rho,
                           spec_str_phase2=spec_str_phase2,
                           switch_step=switch_step,
                           use_voice=use_voice)
            else:
                # non-MPC: just fly the planned trajectory with send_full_state_setpoint
                from visuals.visualization import Visualizer
                vis = Visualizer(waypoints, scenario)
                fig, ax = vis.visualize_trajectory_rho_gradient_2d(all_rho) \
                          if all_rho is not None else vis.visualize_trajectory()
                plt.show(block=False)
                fig.canvas.draw()
                fig.canvas.flush_events()
                mk, = ax.plot([waypoints[0, 0]], [waypoints[1, 0]], 'bo', ms=10, zorder=100)
                ln, = ax.plot([], [], 'b--', lw=2, zorder=99)
                ax.legend()

                def update(_):
                    if mocap.history:
                        h = np.array(mocap.history)
                        ln.set_data(h[:, 0]/SCALE_XY - ox, h[:, 1]/SCALE_XY - oy)
                        mk.set_data([mocap.pos[0]/SCALE_XY - ox], [mocap.pos[1]/SCALE_XY - oy])
                    return mk, ln

                ani = FuncAnimation(fig, update, interval=200, blit=False,
                                    cache_frame_data=False, repeat=False)
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(2)

                send_interval = 1.0 / SEND_HZ
                err = [None]

                def fly():
                    try:
                        for i in range(waypoints.shape[1]):
                            pos_sp = np.array([waypoints[0, i]*SCALE_XY,
                                               waypoints[1, i]*SCALE_XY,
                                               waypoints[2, i]])
                            vel_sp = waypoints[3:6, i]
                            t_start = time.time()
                            while time.time() - t_start < dt:
                                cf.commander.send_full_state_setpoint(
                                    pos_sp, vel_sp, np.zeros(3),
                                    np.array([0., 0., 0., 1.]),
                                    0., 0., 0.
                                )
                                time.sleep(send_interval)
                        cf.param.set_value('commander.enHighLevel', '1')
                        time.sleep(0.1)
                        cmd.land(0.0, 2.0)
                        time.sleep(2.0)
                        cmd.stop()
                    except Exception as e:
                        err[0] = e

                ft = threading.Thread(target=fly)
                ft.start()
                while ft.is_alive():
                    plt.pause(0.05)
                ft.join()
                if err[0]:
                    raise err[0]
                ani.event_source.stop()

            try:
                cf.platform.send_arming_request(False)
            except Exception:
                pass
            cf.commander.send_stop_setpoint()
            if scenario:
                input("Press Enter to close.")

    finally:
        mocap.close()