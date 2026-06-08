# -*- coding: utf-8 -*-
"""
deploy_online2.py

Online MPC deployment with voice-driven goal switching and auto time extension.
Synced from simulation_two_mode.py logic.
Use this instead of deploy_online.py when voice switching is needed.
"""
import time
import multiprocessing as mp
import re
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from threading import Thread

import motioncapture, cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.mem import MemoryElement
from cflib.utils import uri_helper
from cflib.utils.reset_estimator import reset_estimator

REPLAN_INTERVAL        = 2
REPLAN_BUFFER_AUTO     = 2
REPLAN_BUFFER_INTERACT = 4
EXTENSION_STEPS        = 10
SCALE_XY               = 0.45
SEND_HZ                = 50


# ── helpers ───────────────────────────────────────────────────────────────────
def shift_spec(spec_str, step_offset):
    def replace_window(match):
        method = match.group(1)
        a = max(0, int(match.group(2)) - step_offset)
        b = max(0, int(match.group(3)) - step_offset)
        return f".{method}({a}, {b})"
    return re.sub(r'\.(eventually|always)\((\d+),\s*(\d+)\)', replace_window, spec_str)


def vicon_to_plan(pos_v, ox, oy):
    return np.array([pos_v[0] / SCALE_XY - ox, pos_v[1] / SCALE_XY - oy, pos_v[2]])


def plan_to_vicon(pos_p):
    return pos_p[0] * SCALE_XY, pos_p[1] * SCALE_XY


def rho_to_wrgb8888(rho_val):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        'rho_colormap', ['#8B0000', '#FF0000', '#FFFF00', '#00FF00', '#006400'])
    norm_val = 0.0 if rho_val <= 0.5 else min(1.0, (rho_val - 0.5) / 1.0)
    r, g, b, _ = cmap(norm_val)
    return (int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255)


def set_color_led_ring(mem, rho_val):
    wrgb = rho_to_wrgb8888(rho_val)
    R = (wrgb >> 16) & 0xFF
    G = (wrgb >> 8) & 0xFF
    B = wrgb & 0xFF
    for i in range(12):
        mem[0].leds[i].set(r=R, g=G, b=B)
    mem[0].write_data(None)


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
        self._vf = self.alpha * (pos - self._pp) / dt + (1 - self.alpha) * self._vf
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


# ── persistent replan worker ──────────────────────────────────────────────────
def _persistent_worker(task_queue, result_queue):
    from STL.STL_to_path import STLSolver, drone_dynamics
    import gurobipy

    while True:
        task = task_queue.get()
        if task is None:
            break

        (trigger_step, apply_step, x0_replan,
         spec_str, scenario_objects, N_total, dt, max_acc, max_speed,
         spec_switched, spec_switch_announced, apply_buffer) = task

        try:
            T_remaining = (N_total - apply_step - 2) * dt
            if T_remaining < dt:
                result_queue.put(('skip', trigger_step))
                continue

            shifted_spec = shift_spec(spec_str, apply_step + 1)
            solver = STLSolver(shifted_spec, scenario_objects, x0_replan, T_remaining)
            x_new, u_new, rho, rho_series, runtime = solver.generate_trajectory(
                dt, max_acc, max_speed, verbose=False)

            if x_new is not None and rho > 0:
                needs_interaction = spec_switched and not spec_switch_announced
                result_queue.put(('success', trigger_step, x_new, u_new, rho,
                                  rho_series, runtime, needs_interaction, x0_replan, apply_buffer))
            else:
                result_queue.put(('failed', trigger_step, rho))

        except Exception as e:
            result_queue.put(('error', trigger_step, str(e)))


# ── online MPC loop with voice switching + auto time extension ────────────────
def run_online(cf, cmd, waypoints, all_u, mocap,
               spec_str, scenario, dt, max_acc, max_speed,
               all_rho=None, spec_str_phase2=None, switch_step=None,
               use_voice=False, vel_alpha=0.7):

    _N_total         = [all_u.shape[1]]
    orig_N           = _N_total[0]
    scenario_objects = scenario.objects if scenario is not None else {}

    current_u    = all_u.copy()
    spec_current = [spec_str]

    cur_wp      = np.zeros((3, orig_N))
    cur_wp[0,:] = waypoints[0,:] * SCALE_XY
    cur_wp[1,:] = waypoints[1,:] * SCALE_XY
    cur_wp[2,:] = waypoints[2, 0]
    cur_wp_plan = waypoints.copy()
    cur_wp_plan[5, :] = 0.0

    task_queue          = mp.Queue()
    result_queue        = mp.Queue()
    new_trajectory      = [None]
    replan_running      = [False]
    replan_log          = []
    user_decision       = [None]
    _current_step       = [0]
    pending_voice_goal  = [None]
    spec_switch_announced = [False]
    spec_switch_accepted  = [False]

    worker = mp.Process(target=_persistent_worker,
                        args=(task_queue, result_queue), daemon=True)
    worker.start()
    print("[Worker] Persistent replan worker started.")

    # ── spec builder from goal name ───────────────────────────────────────────
    def _build_spec_for_goal(goal_name, objects, N_total):
        N = N_total - 1
        obs_parts = ' & '.join(
            f'STL_formulas.outside_cuboid(objects["{k}"], name="!{k}")'
            for k in objects if 'obstacle' in k.lower()
        )
        spec = (f'STL_formulas.inside_cuboid(objects["{goal_name}"], name="{goal_name}")'
                f'.always({N}, {N})')
        if obs_parts:
            spec += f' & ({obs_parts}).always(0, {N})'
        return spec

    # ── voice ─────────────────────────────────────────────────────────────────
    voice_listener = None
    if use_voice:
        try:
            from LLM.voice_vad import VoiceListener
            from LLM.voice_interpreter import VoiceInterpreter

            goal_names  = [k for k in scenario_objects if 'goal' in k.lower()]
            interpreter = VoiceInterpreter(goal_names, GPT_model="gpt-4o")

            def on_utterance(text):
                if user_decision[0] == 'waiting':
                    def _classify_decision():
                        result = interpreter.interpret(text)
                        if result == 'yes':
                            user_decision[0] = 'accepted'
                            print("[Voice] Accepted.")
                        elif result == 'no':
                            user_decision[0] = 'rejected'
                            print("[Voice] Rejected.")
                    threading.Thread(target=_classify_decision, daemon=True).start()
                    return

                def _classify_instruction():
                    result = interpreter.interpret(text)
                    if result in (None, 'yes', 'no'):
                        return
                    new_spec = _build_spec_for_goal(result, scenario_objects, _N_total[0])
                    print(f"[Voice] New spec for '{result}': {new_spec}")
                    spec_switch_announced[0] = False   # reset first
                    spec_current[0] = new_spec         # spec set before pending flag
                    pending_voice_goal[0] = result     # last: triggers is_spec_switched
                    print("[Voice] spec_current updated — takes effect on next replan.")

                threading.Thread(target=_classify_instruction, daemon=True).start()

            voice_listener = VoiceListener(on_utterance=on_utterance)
            print("[Voice] Voice interaction enabled.")
        except Exception as e:
            print(f"[Voice] Could not load voice module ({e}), keyboard only.")

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
    real_dot,       = ax.plot([], [], 'bo',  ms=8, zorder=100)
    candidate_line, = ax.plot([], [], '--',  color='purple', lw=2,
                              label='Candidate new path', zorder=102)
    branch_marker,  = ax.plot([], [], 'o',   color='purple', ms=10,
                              label='Branch point', zorder=103)

    plan_xs = [waypoints[0, 0]]
    plan_ys = [waypoints[1, 0]]
    ax.legend(loc='lower right', fontsize=8)

    prompt_ax = fig.add_axes([0.05, 0.01, 0.9, 0.05])
    prompt_ax.set_xlim(0, 1); prompt_ax.set_ylim(0, 1); prompt_ax.axis('off')
    prompt_text_obj = prompt_ax.text(
        0.5, 0.95, '', ha='center', va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='gray'))
    timer_bg_patch  = Rectangle((0.0, 0.05), 1.0, 0.25, transform=prompt_ax.transAxes,
                                 facecolor='lightgray', edgecolor='none')
    timer_bar_patch = Rectangle((0.0, 0.05), 1.0, 0.25, transform=prompt_ax.transAxes,
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
    fig.canvas.draw()
    fig.canvas.flush_events()

    # ── takeoff + calibrate ───────────────────────────────────────────────────
    print("[Deploy] Taking off...")
    cmd.takeoff(waypoints[2, 0], 2.0)
    time.sleep(2.0)
    cmd.go_to(waypoints[0, 0] * SCALE_XY, waypoints[1, 0] * SCALE_XY,
              waypoints[2, 0], yaw=0.0, duration_s=2.0)
    time.sleep(2.0)

    time.sleep(1.0)
    samples = []
    for _ in range(10):
        samples.append(mocap.pos.copy())
        time.sleep(0.1)
    actual = np.mean(samples, axis=0)
    ox = actual[0] / SCALE_XY - waypoints[0, 0]
    oy = actual[1] / SCALE_XY - waypoints[1, 0]
    print(f"Offset: X={ox:.3f}, Y={oy:.3f}")
    mocap.history = []

    vel_est = VelocityEstimator(alpha=vel_alpha)
    vel_est.update(vicon_to_plan(mocap.pos, ox, oy), time.time())

    # ── LED ring ──────────────────────────────────────────────────────────────
    led_mem = None
    try:
        cf.param.set_value('ring.effect', '13')
        time.sleep(0.3)
        led_mem = cf.mem.get_mems(MemoryElement.TYPE_DRIVER_LED)
        if not led_mem:
            led_mem = None
            print("[LED] No LED memory found.")
    except Exception as e:
        print(f"[LED] Could not initialize LED ring: {e}")
        led_mem = None

    cur_rho = all_rho.copy() if all_rho is not None else np.zeros(_N_total[0])

    pos    = vicon_to_plan(mocap.pos.copy(), ox, oy)
    vel    = vel_est.update(pos, time.time())
    x_true = np.concatenate([pos, vel])
    position_history = [pos.copy()]
    send_interval    = 1.0 / SEND_HZ

    print(f"\n{'='*60}\nDeployment: {_N_total[0]} steps, dt={dt}s\n{'='*60}\n")

    step = 0
    try:
        while step < _N_total[0]:
            _current_step[0] = step

            # ── 0. CHECK REPLAN RESULT ────────────────────────────────────────
            if not result_queue.empty():
                msg = result_queue.get_nowait()
                if msg[0] == 'success':
                    _, tsr, xnr, unr, rhor, rhoseqr, rtr, nir, x0r, abr = msg
                    if not nir:
                        new_trajectory[0] = (tsr, xnr, unr, rhor, rhoseqr, abr)
                        user_decision[0]  = 'accepted'
                        print(f"[Replan t={tsr}] rho={rhor:.3f} rt={rtr:.2f}s → auto-accepted")
                    else:
                        spec_switch_announced[0] = True
                        new_trajectory[0] = (tsr, xnr, unr, rhor, rhoseqr, abr)
                        candidate_line.set_data(xnr[0, :], xnr[1, :])
                        branch_marker.set_data([x0r[0]], [x0r[1]])
                        user_decision[0] = 'waiting'
                        prompt_ax.set_visible(True)
                        prompt_text_obj.set_text(
                            f"New path available (rho={rhor:.3f})    "
                            f"[Enter/Yes] Accept   [Space/No] Reject    "
                            f"(auto-reject at step {tsr + abr})")
                        timer_bar_patch.set_width(1.0)
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                        print(f"[Replan t={tsr}] rho={rhor:.3f} rt={rtr:.2f}s "
                              f"→ waiting for user (apply at step {tsr + abr})")
                    replan_running[0] = False

                elif msg[0] in ('failed', 'skip', 'error'):
                    lbl = {'failed': f"Failed (rho={msg[2]})",
                           'skip':   "No time remaining",
                           'error':  f"Exception: {msg[2]}"}[msg[0]]
                    print(f"[Replan t={msg[1]}] {lbl}, keeping current.")
                    replan_log.append((msg[1], None, False))
                    replan_running[0] = False

                    # auto-extend when voice goal unreachable in remaining time
                    if pending_voice_goal[0] is not None and spec_current[0] != spec_str:
                        extra = EXTENSION_STEPS
                        _N_total[0] += extra
                        current_u   = np.hstack([current_u,   np.zeros((3, extra))])
                        cur_wp      = np.hstack([cur_wp,      np.tile(cur_wp[:, -1:],      (1, extra))])
                        cur_wp_plan = np.hstack([cur_wp_plan, np.tile(cur_wp_plan[:, -1:], (1, extra))])
                        cur_rho     = np.concatenate([cur_rho, np.zeros(extra)])
                        new_spec = _build_spec_for_goal(
                            pending_voice_goal[0], scenario_objects, _N_total[0])
                        spec_current[0] = new_spec
                        print(f"[Extend] Extended by {extra} steps → N_total={_N_total[0]}.")

            # ── 1. EXECUTE ────────────────────────────────────────────────────
            if led_mem:
                try:
                    if step < _N_total[0] - 1:
                        set_color_led_ring(led_mem, cur_rho[min(step, len(cur_rho) - 1)])
                    else:
                        cf.param.set_value('ring.effect', '2')
                except Exception:
                    pass

            si     = min(step, cur_wp.shape[1] - 1)
            pos_sp = cur_wp[:, si]
            vel_sp = cur_wp_plan[3:6, si]
            # velocity in vicon coords (x,y scaled, z unchanged)
            vel_vicon = np.array([vel_sp[0] * SCALE_XY, vel_sp[1] * SCALE_XY, vel_sp[2]])

            t_start = time.time()
            while time.time() - t_start < dt:
                t_elapsed = time.time() - t_start
                interp_pos = pos_sp + vel_vicon * t_elapsed
                cf.commander.send_full_state_setpoint(
                    interp_pos, vel_vicon, np.zeros(3),
                    np.array([0., 0., 0., 1.]), 0., 0., 0.)
                fig.canvas.flush_events()
                time.sleep(send_interval)

            # ── 2. SAMPLE from vicon ──────────────────────────────────────────
            pos    = vicon_to_plan(mocap.pos.copy(), ox, oy)
            vel    = vel_est.update(pos, time.time())
            x_true = np.concatenate([pos, vel])
            position_history.append(pos.copy())
            plan_pos = cur_wp_plan[:3, si]
            print(f"[Step {step:3d}] "
                  f"plan=({plan_pos[0]:.3f},{plan_pos[1]:.3f},{plan_pos[2]:.3f}) "
                  f"true=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) "
                  f"vel=({vel[0]:.3f},{vel[1]:.3f},{vel[2]:.3f})")

            # ── 3. APPLY ──────────────────────────────────────────────────────
            pending = new_trajectory[0]
            if pending is not None:
                trigger_step, x_new, u_new, rho, rho_series, replan_buffer = pending
                expected = trigger_step + replan_buffer

                if step < expected:
                    if user_decision[0] == 'waiting':
                        timer_bar_patch.set_width((expected - step) / replan_buffer)

                elif step == expected:
                    decision = user_decision[0]
                    if decision == 'accepted':
                        rem  = _N_total[0] - step - 1
                        cols = min(u_new.shape[1], rem)
                        new_u_block = np.zeros((3, rem))
                        new_u_block[:, :cols] = u_new[:, :cols]
                        if cols < rem:
                            new_u_block[:, cols:] = u_new[:, -1:]
                        current_u[:, step+1:step+1+rem] = new_u_block

                        blk_v    = np.zeros((3, rem))
                        blk_plan = np.zeros((6, rem))
                        for i in range(min(cols, rem)):
                            xi  = i + 1
                            src = x_new[:, xi] if xi < x_new.shape[1] else x_new[:, -1]
                            blk_v[0, i], blk_v[1, i] = plan_to_vicon(src[:2])
                            blk_v[2, i]    = src[2]
                            blk_plan[:, i] = src
                        if cols < rem:
                            blk_v[:, cols:]    = blk_v[:, cols-1:cols]
                            blk_plan[:, cols:] = blk_plan[:, cols-1:cols]
                        blk_plan[3:6, 0] = x_true[3:6]
                        cur_wp[:, step+1:step+1+rem]      = blk_v
                        cur_wp[2, step+1:step+1+rem]      = waypoints[2, 0]
                        cur_wp_plan[:, step+1:step+1+rem] = blk_plan
                        cur_wp_plan[5, step+1:step+1+rem] = 0.0

                        rc = min(len(rho_series), rem)
                        cur_rho[step+1:step+1+rc] = rho_series[:rc]
                        if rc < rem:
                            cur_rho[step+1+rc:step+1+rem] = rho_series[-1] if rc > 0 else 0.0

                        if not spec_switch_accepted[0] and spec_switch_announced[0]:
                            spec_switch_accepted[0] = True
                            pts  = np.array([x_new[0, :], x_new[1, :]]).T.reshape(-1, 1, 2)
                            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                            norm = plt.Normalize(
                                vmin=0, vmax=max(rho_series) if max(rho_series) > 0 else 1)
                            lc = LineCollection(segs, cmap='RdYlGn', norm=norm,
                                                linewidth=2, zorder=101)
                            lc.set_array(np.array(rho_series[:len(segs)]))
                            ax.add_collection(lc)

                        pending_voice_goal[0] = None
                        replan_log.append((trigger_step, step, True))
                        print(f"[Step {step:3d}] Replan APPLIED (trig={trigger_step})")
                    else:
                        reason = 'timeout' if decision == 'waiting' else 'rejected'
                        if reason in ('rejected', 'timeout'):
                            spec_current[0] = spec_str
                            spec_switch_announced[0] = False
                            pending_voice_goal[0] = None
                        replan_log.append((trigger_step, step, False))
                        print(f"[Step {step:3d}] Replan {reason} (trig={trigger_step}).")

                    candidate_line.set_data([], [])
                    branch_marker.set_data([], [])
                    prompt_text_obj.set_text('')
                    timer_bar_patch.set_width(0)
                    prompt_ax.set_visible(False)
                    user_decision[0]  = None
                    new_trajectory[0] = None

                else:
                    candidate_line.set_data([], [])
                    branch_marker.set_data([], [])
                    prompt_text_obj.set_text('')
                    timer_bar_patch.set_width(0)
                    prompt_ax.set_visible(False)
                    user_decision[0]  = None
                    new_trajectory[0] = None
                    print(f"[Step {step:3d}] Replan expired (trig={trigger_step}), discarding.")

            # ── 4. SPEC SWITCH (predefined, optional) ─────────────────────────
            if switch_step is not None and step == switch_step and spec_str_phase2 is not None:
                spec_current[0] = spec_str_phase2
                print(f"[Step {step:3d}] *** Spec switched to phase 2 ***")

            # ── 5. TRIGGER REPLAN ─────────────────────────────────────────────
            if (spec_current[0] is not None
                    and step % REPLAN_INTERVAL == 0
                    and step > 0
                    and user_decision[0] != 'waiting'):

                is_spec_switched = ((spec_current[0] != spec_str)
                                    or (pending_voice_goal[0] is not None))
                needs_interact   = is_spec_switched and not spec_switch_announced[0]
                apply_buffer     = REPLAN_BUFFER_INTERACT if needs_interact else REPLAN_BUFFER_AUTO

                if step + apply_buffer <= _N_total[0] - 2 and not replan_running[0]:
                    replan_running[0] = True
                    candidate_line.set_data([], [])
                    branch_marker.set_data([], [])
                    prompt_text_obj.set_text('')
                    timer_bar_patch.set_width(0)
                    prompt_ax.set_visible(False)
                    user_decision[0] = None

                    apply_step = step + apply_buffer
                    pi         = min(apply_step + 1, cur_wp_plan.shape[1] - 1)
                    x0_replan  = cur_wp_plan[:, pi].copy()
                    print(f"[Step {step:3d}] Replanning triggered (buffer={apply_buffer}). "
                          f"Predicted apply pos: ({x0_replan[0]:.3f}, {x0_replan[1]:.3f})")

                    task_queue.put((
                        step, apply_step, x0_replan,
                        spec_current[0], scenario_objects, _N_total[0],
                        dt, max_acc, max_speed,
                        is_spec_switched, spec_switch_announced[0], apply_buffer
                    ))

            # ── 6. VISUALISE ──────────────────────────────────────────────────
            if step < orig_N - 1:
                plan_xs.append(waypoints[0, step + 1])
                plan_ys.append(waypoints[1, step + 1])
                plan_line.set_data(plan_xs, plan_ys)

            history_arr = np.array(position_history)
            real_line.set_data(history_arr[:, 0], history_arr[:, 1])
            real_dot.set_data([x_true[0]], [x_true[1]])
            fig.canvas.draw()
            fig.canvas.flush_events()

            step += 1

        # hover at final position until user confirms landing
        final_sp = cur_wp[:, min(_N_total[0] - 1, cur_wp.shape[1] - 1)]
        try:
            cf.param.set_value('ring.effect', '2')
        except Exception:
            pass
        print("[Deploy] Trajectory complete. Hovering — press Enter to land.")
        _land_requested = [False]

        def _wait_for_enter():
            input()
            _land_requested[0] = True

        import threading as _t
        _t.Thread(target=_wait_for_enter, daemon=True).start()

        while not _land_requested[0]:
            cf.commander.send_full_state_setpoint(
                final_sp, np.zeros(3), np.zeros(3),
                np.array([0., 0., 0., 1.]), 0., 0., 0.)
            fig.canvas.flush_events()
            time.sleep(send_interval)

    except Exception as e:
        cf.commander.send_stop_setpoint()
        raise
    finally:
        cf.param.set_value('commander.enHighLevel', '1')
        time.sleep(0.8)
        cf.high_level_commander.land(0.0, 2.0)
        time.sleep(3.0)
        cf.high_level_commander.stop()

        if led_mem:
            try:
                for i in range(12):
                    led_mem[0].leds[i].set(r=0, g=0, b=0)
                led_mem[0].write_data(None)
            except Exception:
                pass

        if voice_listener is not None:
            voice_listener.close()
        task_queue.put(None)
        worker.join(timeout=3.0)
        if worker.is_alive():
            worker.terminate()
        print(f"Replan log: {replan_log}")

    print(f"\n{'='*60}\nFlight complete.\nReplan log: {replan_log}\n{'='*60}\n")
    input("Press Enter to close.")
    plt.close('all')
    return position_history, replan_log


# ── deploy entry point ────────────────────────────────────────────────────────
def deploy(waypoints, all_u, scenario=None, all_rho=None, spec_str=None,
           spec_str_phase2=None, switch_step=None,
           dt=0.75, max_acc=10, max_speed=0.5,
           drone_name='cf11', host_name='10.128.7.250',
           send_full_pose=False, use_online_mpc=False,
           use_voice=False, vel_alpha=0.7):

    uri = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E701')
    cflib.crtp.init_drivers()
    mocap = MocapWrapper(drone_name, host_name)

    try:
        with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
            cf  = scf.cf
            cmd = cf.high_level_commander

            mocap.on_pose = lambda p: (
                cf.extpos.send_extpose(p[0], p[1], p[2], p[3].x, p[3].y, p[3].z, p[3].w)
                if send_full_pose else cf.extpos.send_extpos(p[0], p[1], p[2]))

            cf.param.set_value('locSrv.extQuatStdDev', '0.06')
            cf.param.set_value('stabilizer.estimator', '2')
            cf.param.set_value('commander.enHighLevel', '1')
            reset_estimator(cf)
            try:
                cf.supervisor.send_arming_request(True)
            except Exception:
                pass
            time.sleep(1.0)

            if use_online_mpc and spec_str:
                run_online(cf, cmd, waypoints, all_u, mocap,
                           spec_str, scenario, dt, max_acc, max_speed,
                           all_rho=all_rho,
                           spec_str_phase2=spec_str_phase2,
                           switch_step=switch_step,
                           use_voice=use_voice,
                           vel_alpha=vel_alpha)
            else:
                # non-MPC: takeoff + fly planned trajectory
                cmd.takeoff(waypoints[2, 0], 2.0)
                time.sleep(2.0)
                cmd.go_to(waypoints[0, 0] * SCALE_XY, waypoints[1, 0] * SCALE_XY,
                          waypoints[2, 0], yaw=0.0, duration_s=2.0)
                time.sleep(2.0)

                time.sleep(1.0)
                samples = []
                for _ in range(10):
                    samples.append(mocap.pos.copy())
                    time.sleep(0.1)
                actual = np.mean(samples, axis=0)
                ox = actual[0] / SCALE_XY - waypoints[0, 0]
                oy = actual[1] / SCALE_XY - waypoints[1, 0]
                print(f"Offset: X={ox:.3f}, Y={oy:.3f}")
                mocap.history = []

                from visuals.visualization import Visualizer
                from matplotlib.animation import FuncAnimation
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
                        ln.set_data(h[:, 0] / SCALE_XY - ox, h[:, 1] / SCALE_XY - oy)
                        mk.set_data([mocap.pos[0] / SCALE_XY - ox],
                                    [mocap.pos[1] / SCALE_XY - oy])
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
                            pos_sp = np.array([waypoints[0, i] * SCALE_XY,
                                               waypoints[1, i] * SCALE_XY,
                                               waypoints[2, i]])
                            vel_sp  = waypoints[3:6, i]
                            t_start = time.time()
                            while time.time() - t_start < dt:
                                cf.commander.send_full_state_setpoint(
                                    pos_sp, vel_sp, np.zeros(3),
                                    np.array([0., 0., 0., 1.]), 0., 0., 0.)
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
                cf.supervisor.send_arming_request(False)
            except Exception:
                pass
            cf.commander.send_stop_setpoint()
            if scenario:
                input("Press Enter to close.")

    finally:
        mocap.close()
