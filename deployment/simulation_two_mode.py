# # -*- coding: utf-8 -*-
# import threading
# import re
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle

# from STL.STL_to_path import STLSolver, drone_dynamics

# REPLAN_INTERVAL = 3
# REPLAN_BUFFER   = 3


# def shift_spec(spec_str, step_offset):
#     def replace_window(match):
#         method = match.group(1)
#         a = max(0, int(match.group(2)) - step_offset)
#         b = max(0, int(match.group(3)) - step_offset)
#         return f".{method}({a}, {b})"
#     return re.sub(r'\.(eventually|always)\((\d+),\s*(\d+)\)', replace_window, spec_str)


# def simulate_deployment(
#     waypoints,
#     all_u,
#     scenario=None,
#     all_rho=None,
#     noise_std=0.05,
#     spec_str=None,
#     spec_str_phase2=None,   # optional: spec to switch to at switch_step
#     switch_step=None,        # step at which to switch spec (None = no switch)
#     dt=0.7,
#     max_acc=10,
#     max_speed=0.5,
#     use_voice=False,
#     show_no_mpc=False,
# ):
#     dyn     = drone_dynamics(dt=dt, max_acc=max_acc)
#     A_tilde = dyn.A_tilde
#     B_tilde = dyn.B_tilde

#     N_total          = all_u.shape[1]
#     scenario_objects = scenario.objects if scenario is not None else {}

#     current_u    = all_u.copy()
#     x_true       = waypoints[:6, 0].copy()
#     spec_current = [spec_str]   # mutable so replan closure sees updates

#     position_history = [x_true[:3].copy()]

#     # no-mpc comparison
#     nompc_x_true  = waypoints[:6, 0].copy()
#     nompc_history = [nompc_x_true[:3].copy()]

#     lock           = threading.Lock()
#     new_trajectory = [None]
#     replan_running = [False]
#     replan_log     = []

#     user_decision = [None]  # None / 'accepted' / 'rejected'

#     active_plan_data  = [waypoints]
#     active_plan_start = [0]
#     is_offline        = [True]
#     history_lines     = []
#     accept_pts        = []
#     plan_tracking = [True]

#     # voice
#     listen_for_yes_no = None
#     if use_voice:
#         try:
#             from LLM.voice_openai import listen_for_yes_no_voice
#             listen_for_yes_no = listen_for_yes_no_voice
#             print("[Voice] Voice interaction enabled.")
#         except Exception as e:
#             print(f"[Voice] Could not load voice module ({e}), keyboard only.")

#     # ── Visualisation ──────────────────────────────────────────────────────────
#     from visuals.visualization import Visualizer
#     vis_x = waypoints[:6, :]
#     visualizer = Visualizer(vis_x, scenario)

#     if all_rho is not None and len(all_rho) == vis_x.shape[1]:
#         fig, ax = visualizer.visualize_trajectory_rho_gradient_2d(all_rho)
#     else:
#         fig, ax = plt.subplots(figsize=(10, 10))

#     ax.set_title('Online MPC Simulation')

#     plan_line,      = ax.plot([], [], 'o-', color='orange', ms=4, lw=1.5,
#                               label='Planned (offline)', zorder=98)
#     real_line,      = ax.plot([], [], 'b--', lw=2, label='True trajectory', zorder=99)
#     real_dot,       = ax.plot([], [], 'bo',  ms=8,                          zorder=100)
#     active_line,    = ax.plot([], [], '-',   color='purple', lw=2,
#                               label='Current plan', zorder=101)
#     candidate_line, = ax.plot([], [], '--',  color='purple', lw=2,
#                               label='Candidate new path', zorder=102)
#     branch_marker,  = ax.plot([], [], 'o',   color='purple', ms=10,
#                               label='Branch point', zorder=103)
#     accept_marker,  = ax.plot([], [], '^',   color='purple', ms=12,
#                               label='Accepted', zorder=104)
#     nompc_line,     = ax.plot([], [], 'r--', lw=1.5, label='No MPC',
#                               zorder=97, visible=show_no_mpc)

#     plan_xs = [waypoints[0, 0]]
#     plan_ys = [waypoints[1, 0]]
#     ax.legend(loc='upper left', fontsize=8)

#     prompt_ax = fig.add_axes([0.05, 0.01, 0.9, 0.05])
#     prompt_ax.set_xlim(0, 1)
#     prompt_ax.set_ylim(0, 1)
#     prompt_ax.axis('off')

#     prompt_text_obj = prompt_ax.text(
#         0.5, 0.95, '',
#         ha='center', va='top', fontsize=10,
#         bbox=dict(boxstyle='round', facecolor='lightyellow', 
#                 alpha=0.9, edgecolor='gray'),
#     )
#     timer_bg_patch = Rectangle((0.0, 0.05), 1.0, 0.25,
#                                 transform=prompt_ax.transAxes,
#                                 facecolor='lightgray', edgecolor='none')
#     timer_bar_patch = Rectangle((0.0, 0.05), 1.0, 0.25,
#                                 transform=prompt_ax.transAxes,
#                                 facecolor='orange', edgecolor='none')
#     prompt_ax.add_patch(timer_bg_patch)
#     prompt_ax.add_patch(timer_bar_patch)
#     prompt_ax.set_visible(False)

#     def on_key(event):
#         if user_decision[0] == 'waiting':
#             if event.key == 'enter':
#                 user_decision[0] = 'accepted'
#                 print("\n[Keyboard] Accepted.")
#             elif event.key == ' ':
#                 user_decision[0] = 'rejected'
#                 print("\n[Keyboard] Rejected.")

#     fig.canvas.mpl_connect('key_press_event', on_key)
#     active_line.set_data(waypoints[0, :], waypoints[1, :])

#     plt.show(block=False)
#     mng = plt.get_current_fig_manager()
#     try:
#         mng.window.showMaximized()
#     except Exception as e:
#         print(f"Could not maximize: {e}")
#     fig.canvas.draw()
#     fig.canvas.flush_events()

#     # ── Background replanning ──────────────────────────────────────────────────
#     def replan(trigger_step, apply_step, x0_replan, spec_switched):
#         try:
#             T_remaining = (N_total - apply_step -1) * dt
#             if T_remaining <= dt:
#                 print(f"[Replan t={trigger_step}] No time remaining, skipping.")
#                 return

#             shifted_spec = shift_spec(spec_current[0], apply_step)
#             max_step = int(T_remaining / dt) - 1

#             def clamp_window(match):
#                 method = match.group(1)
#                 a = min(int(match.group(2)), max_step)
#                 b = min(int(match.group(3)), max_step)
#                 return f".{method}({a}, {b})"
#             shifted_spec = re.sub(
#                 r'\.(eventually|always)\((\d+),\s*(\d+)\)',
#                 clamp_window, shifted_spec
#             )

#             solver = STLSolver(shifted_spec, scenario_objects, x0_replan, T_remaining)
#             x_new, u_new, rho, _, runtime = solver.generate_trajectory(
#                 dt, max_acc, max_speed, verbose=False
#             )

#             if x_new is not None and rho > 0:
#                 with lock:
#                     new_trajectory[0] = (trigger_step, x_new, u_new, rho)
#                 candidate_line.set_data(x_new[0, 0:-1], x_new[1, 0:-1])
#                 branch_marker.set_data([x0_replan[0]], [x0_replan[1]])
#                 user_decision[0] = 'waiting'
#                 prompt_ax.set_visible(True)
#                 path_msg = "New path available" if spec_switched else "Safer path available"
#                 prompt_text_obj.set_text(
#                     f"{path_msg} (rho={rho:.3f})    "
#                     f"[Enter/Yes] Accept   [Space/No] Reject    "
#                     f"(auto-reject at step {apply_step})"
#                     )
#                 timer_bar_patch.set_width(1.0)
#                 fig.canvas.draw()
#                 fig.canvas.flush_events()
#                 print(f"[Replan t={trigger_step}] rho={rho:.3f} rt={runtime:.2f}s "
#                       f"→ waiting for user (apply at step {apply_step})")

#                 if listen_for_yes_no is not None:
#                     def voice_listen():
#                         try:
#                             from LLM.voice_openai import VoiceOpenAI
#                             voice = VoiceOpenAI()
#                             print(f"\n[Voice] 🎤 Say YES or NO...")
#                             while user_decision[0] == 'waiting':
#                                 frames = voice.record_audio(duration=2, silence_threshold=2000, silence_duration=0.5)
#                                 audio_np = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32)
#                                 if np.sqrt(np.mean(audio_np**2)) < 500:
#                                     print("[Voice] No speech, listening again...")
#                                     continue
#                                 import tempfile, os
#                                 tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
#                                 voice.save_audio(frames, tmp.name); tmp.close()
#                                 text = voice.transcribe(tmp.name).lower(); os.unlink(tmp.name)
#                                 print(f"[Voice] You said: '{text}'")
#                                 if any(w in text for w in ['yes','yeah','sure','okay','yep','accept']):
#                                     if user_decision[0] == 'waiting':
#                                         user_decision[0] = 'accepted'; print("[Voice] Accepted.")
#                                     break
#                                 elif any(w in text for w in ['no','nope','reject','cancel']):
#                                     if user_decision[0] == 'waiting':
#                                         user_decision[0] = 'rejected'; print("[Voice] Rejected.")
#                                     break
#                                 else:
#                                     print("[Voice] Please say YES or NO.")
#                             voice.close()
#                         except Exception as e:
#                             print(f"[Voice] Error: {e}, use keyboard.")
#                     threading.Thread(target=voice_listen, daemon=True).start()
#             else:
#                 print(f"[Replan t={trigger_step}] Failed (rho={rho}), keeping current.")
#                 replan_log.append((trigger_step, None, False))

#         except Exception as e:
#             print(f"[Replan t={trigger_step}] Exception: {e}")
#             replan_log.append((trigger_step, None, False))
#         finally:
#             with lock:
#                 replan_running[0] = False

#     # ── Main loop ──────────────────────────────────────────────────────────────
#     print(f"\n{'='*60}")
#     print(f"Simulation: {N_total} steps, dt={dt}s, noise={noise_std}")
#     if switch_step is not None:
#         print(f"Spec switch at step {switch_step}")
#     print(f"{'='*60}\n")

#     for step in range(N_total):

#         # ── 1. APPLY ──────────────────────────────────────────────────────────
#         with lock:
#             pending = new_trajectory[0]

#         if pending is not None:
#             trigger_step, x_new, u_new, rho = pending
#             expected = trigger_step + REPLAN_BUFFER

#             if step < expected:
#                 if user_decision[0] == 'waiting':
#                     fraction = (expected - step) / REPLAN_BUFFER
#                     timer_bar_patch.set_width(fraction)

#             elif step == expected:
#                 decision = user_decision[0]
#                 if decision == 'accepted':
#                     steps_remaining = N_total - step
#                     cols = min(u_new.shape[1], steps_remaining)
#                     new_u_block = np.zeros((3, steps_remaining))
#                     new_u_block[:, :cols] = u_new[:, :cols]
#                     if cols < steps_remaining:
#                         new_u_block[:, cols:] = u_new[:, -1:]
#                     current_u[:, step:] = new_u_block

#                     prev_data  = active_plan_data[0]
#                     prev_start = active_plan_start[0]
#                     seg_len    = step - prev_start
#                     if seg_len > 0:
#                         if is_offline[0]:
#                             seg_end = min(seg_len, prev_data.shape[1])
#                             seg_xs, seg_ys = prev_data[0, :seg_end], prev_data[1, :seg_end]
#                         else:
#                             seg_end = min(seg_len + 1, prev_data.shape[1])
#                             seg_xs, seg_ys = prev_data[0, 0:seg_end], prev_data[1, 0:seg_end]
#                         if len(seg_xs) > 0:
#                             hl, = ax.plot(seg_xs, seg_ys, '-', color='purple', lw=2, zorder=101)
#                             history_lines.append(hl)

#                     active_plan_data[0]  = x_new
#                     active_plan_start[0] = step
#                     is_offline[0]        = False
#                     plan_tracking[0] = False
#                     active_line.set_data(x_new[0, 0:-1], x_new[1, 0:-1])
#                     accept_pts.append((x_new[0, 0], x_new[1, 0]))
#                     replan_log.append((trigger_step, step, True))
#                     print(f"[Step {step:3d}] Replan APPLIED (trig={trigger_step})")
#                 else:
#                     reason = 'timeout' if decision == 'waiting' else 'rejected'
#                     replan_log.append((trigger_step, step, False))
#                     print(f"[Step {step:3d}] Replan {reason} (trig={trigger_step}), keeping current.")

#                 candidate_line.set_data([], [])
#                 branch_marker.set_data([], [])
#                 prompt_text_obj.set_text('')
#                 timer_bar_patch.set_width(0)
#                 prompt_ax.set_visible(False)
#                 user_decision[0] = None
#                 with lock: new_trajectory[0] = None

#             else:
#                 candidate_line.set_data([], [])
#                 branch_marker.set_data([], [])
#                 prompt_text_obj.set_text('')
#                 timer_bar_patch.set_width(0)
#                 prompt_ax.set_visible(False)
#                 user_decision[0] = None
#                 with lock: new_trajectory[0] = None
#                 print(f"[Step {step:3d}] Replan expired (trig={trigger_step}), discarding.")

#         # ── 2. PROPAGATE ──────────────────────────────────────────────────────
#         if step < N_total-1:
#             noise = np.zeros(6)
#             noise[:3] = np.random.normal(0, noise_std, 3)
#             x_true = A_tilde @ x_true + B_tilde @ current_u[:, step] + noise
#             position_history.append(x_true[:3].copy())
#             if show_no_mpc:
#                 nompc_x_true = A_tilde @ nompc_x_true + B_tilde @ all_u[:, step] + noise
#                 nompc_history.append(nompc_x_true[:3].copy())
#             print(f"[Step {step:3d}] true pos: ({x_true[0]:.3f}, {x_true[1]:.3f}, {x_true[2]:.3f})")
#         else:
#             print(f"[Step {step:3d}] final step, no propagation.")

#         # ── 3. SPEC SWITCH ────────────────────────────────────────────────────
#         if switch_step is not None and step == switch_step and spec_str_phase2 is not None:
#             spec_current[0] = spec_str_phase2
#             print(f"[Step {step:3d}] *** Spec switched to phase 2 ***")

#         # ── 4. TRIGGER REPLAN ─────────────────────────────────────────────────
#         if (spec_current[0] is not None
#                 and step % REPLAN_INTERVAL == 0
#                 and step > 0
#                 and step + REPLAN_BUFFER <= N_total - 2):
#             with lock:
#                 already_running = replan_running[0]
#             if not already_running:
#                 with lock: replan_running[0] = True
#                 candidate_line.set_data([], [])
#                 branch_marker.set_data([], [])
#                 prompt_text_obj.set_text('')
#                 timer_bar_patch.set_width(0)
#                 prompt_ax.set_visible(False)
#                 user_decision[0] = None

#                 apply_step  = step + REPLAN_BUFFER
#                 x_predicted = x_true.copy()
#                 for i in range(REPLAN_BUFFER - 1):
#                     idx = step + 1 + i
#                     if idx < N_total:
#                         x_predicted = A_tilde @ x_predicted + B_tilde @ current_u[:, idx]

#                 x0_replan = x_predicted.copy()
#                 print(f"[Step {step:3d}] Replanning triggered. "
#                       f"Predicted apply pos: ({x0_replan[0]:.3f}, {x0_replan[1]:.3f})")

#                 threading.Thread(
#                     target=replan, args=(step, apply_step, x0_replan, spec_current[0] != spec_str), daemon=True
#                 ).start()

#         # ── 5. VISUALISE ──────────────────────────────────────────────────────
#         print(f"[Vis step={step}] plan_line points={len(plan_xs)+1}, "
#             f"real_line points={len(position_history)}, "
#             f"plan_end=({waypoints[0,step]:.3f},{waypoints[1,step]:.3f}), "
#             f"real_end=({x_true[0]:.3f},{x_true[1]:.3f})")

#         if step < N_total - 1:
#             plan_xs.append(waypoints[0, step + 1])
#             plan_ys.append(waypoints[1, step + 1])
#             plan_line.set_data(plan_xs, plan_ys)

#         if plan_tracking[0]:
#             history_arr = np.array(position_history)
#             real_line.set_data(history_arr[:, 0], history_arr[:, 1])
#             real_dot.set_data([x_true[0]], [x_true[1]])
#         else:
#             history_arr = np.array(position_history[:-1]) if len(position_history) > 1 else np.array(position_history)
#             real_line.set_data(history_arr[:, 0], history_arr[:, 1])
#             current_pos = np.array(position_history[-2]) if len(position_history) > 1 else np.array(position_history[-1])
#             real_dot.set_data([current_pos[0]], [current_pos[1]])
       

#         if show_no_mpc and len(nompc_history) > 1:
#             na = np.array(nompc_history)
#             nompc_line.set_data(na[:, 0], na[:, 1])

#         if accept_pts:
#             accept_marker.set_data([p[0] for p in accept_pts], [p[1] for p in accept_pts])

#         fig.canvas.draw()
#         fig.canvas.flush_events()
#         plt.pause(dt)

#     # ── Done ──────────────────────────────────────────────────────────────────
#     print(f"\n{'='*60}\nSimulation complete.\nReplan log: {replan_log}\n{'='*60}\n")

#     if plan_tracking[0]:
#         history_arr = np.array(position_history)
#     else:
#         history_arr = np.array(position_history[:-1])
#     real_line.set_data(history_arr[:, 0], history_arr[:, 1])
#     plan_line.set_data(plan_xs, plan_ys)
#     if show_no_mpc:
#         na = np.array(nompc_history)
#         nompc_line.set_data(na[:, 0], na[:, 1])
#     fig.canvas.draw()
#     fig.canvas.flush_events()

#     input("Simulation finished. Press Enter to close.")
#     plt.close('all')
#     return position_history, replan_log






# Version 2： same spec closeloop
# # -*- coding: utf-8 -*-
# import threading
# import re
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle

# from STL.STL_to_path import STLSolver, drone_dynamics

# REPLAN_INTERVAL = 3
# REPLAN_BUFFER   = 3


# def shift_spec(spec_str, step_offset):
#     def replace_window(match):
#         method = match.group(1)
#         a = max(0, int(match.group(2)) - step_offset)
#         b = max(0, int(match.group(3)) - step_offset)
#         return f".{method}({a}, {b})"
#     return re.sub(r'\.(eventually|always)\((\d+),\s*(\d+)\)', replace_window, spec_str)


# def simulate_deployment(
#     waypoints,
#     all_u,
#     scenario=None,
#     all_rho=None,
#     noise_std=0.05,
#     spec_str=None,
#     spec_str_phase2=None,   # optional: spec to switch to at switch_step
#     switch_step=None,        # step at which to switch spec (None = no switch)
#     dt=0.7,
#     max_acc=10,
#     max_speed=0.5,
#     use_voice=False,
#     show_no_mpc=False,
# ):
#     dyn     = drone_dynamics(dt=dt, max_acc=max_acc)
#     A_tilde = dyn.A_tilde
#     B_tilde = dyn.B_tilde

#     N_total          = all_u.shape[1]
#     scenario_objects = scenario.objects if scenario is not None else {}

#     current_u    = all_u.copy()
#     x_true       = waypoints[:6, 0].copy()
#     spec_current = [spec_str]   # mutable so replan closure sees updates

#     position_history = [x_true[:3].copy()]

#     # no-mpc comparison
#     nompc_x_true  = waypoints[:6, 0].copy()
#     nompc_history = [nompc_x_true[:3].copy()]

#     lock           = threading.Lock()
#     new_trajectory = [None]
#     replan_running = [False]
#     replan_log     = []

#     user_decision = [None]  # None / 'accepted' / 'rejected' / 'waiting'

#     spec_switch_announced = [False]  # True after the first interactive replan post-switch

#     active_plan_data  = [waypoints]
#     active_plan_start = [0]
#     is_offline        = [True]
#     history_lines     = []
#     plan_tracking = [True]

#     # voice
#     listen_for_yes_no = None
#     if use_voice:
#         try:
#             from LLM.voice_openai import listen_for_yes_no_voice
#             listen_for_yes_no = listen_for_yes_no_voice
#             print("[Voice] Voice interaction enabled.")
#         except Exception as e:
#             print(f"[Voice] Could not load voice module ({e}), keyboard only.")

#     # ── Visualisation ──────────────────────────────────────────────────────────
#     from visuals.visualization import Visualizer
#     vis_x = waypoints[:6, :]
#     visualizer = Visualizer(vis_x, scenario)

#     if all_rho is not None and len(all_rho) == vis_x.shape[1]:
#         fig, ax = visualizer.visualize_trajectory_rho_gradient_2d(all_rho)
#     else:
#         fig, ax = plt.subplots(figsize=(10, 10))

#     ax.set_title('Online MPC Simulation')

#     plan_line,      = ax.plot([], [], 'o-', color='orange', ms=4, lw=1.5,
#                               label='Planned (offline)', zorder=98)
#     real_line,      = ax.plot([], [], 'b--', lw=2, label='True trajectory', zorder=99)
#     real_dot,       = ax.plot([], [], 'bo',  ms=8,                          zorder=100)
#     active_line,    = ax.plot([], [], '-',   color='purple', lw=2,
#                               label='Current plan', zorder=101)
#     candidate_line, = ax.plot([], [], '--',  color='purple', lw=2,
#                               label='Candidate new path', zorder=102)
#     branch_marker,  = ax.plot([], [], 'o',   color='purple', ms=10,
#                               label='Branch point', zorder=103)
#     nompc_line,     = ax.plot([], [], 'r--', lw=1.5, label='No MPC',
#                               zorder=97, visible=show_no_mpc)

#     plan_xs = [waypoints[0, 0]]
#     plan_ys = [waypoints[1, 0]]
#     ax.legend(loc='upper left', fontsize=8)

#     prompt_ax = fig.add_axes([0.05, 0.01, 0.9, 0.05])
#     prompt_ax.set_xlim(0, 1)
#     prompt_ax.set_ylim(0, 1)
#     prompt_ax.axis('off')

#     prompt_text_obj = prompt_ax.text(
#         0.5, 0.95, '',
#         ha='center', va='top', fontsize=10,
#         bbox=dict(boxstyle='round', facecolor='lightyellow',
#                 alpha=0.9, edgecolor='gray'),
#     )
#     timer_bg_patch = Rectangle((0.0, 0.05), 1.0, 0.25,
#                                 transform=prompt_ax.transAxes,
#                                 facecolor='lightgray', edgecolor='none')
#     timer_bar_patch = Rectangle((0.0, 0.05), 1.0, 0.25,
#                                 transform=prompt_ax.transAxes,
#                                 facecolor='orange', edgecolor='none')
#     prompt_ax.add_patch(timer_bg_patch)
#     prompt_ax.add_patch(timer_bar_patch)
#     prompt_ax.set_visible(False)

#     def on_key(event):
#         # Only handle keyboard input when waiting (spec-switch interactive mode)
#         if user_decision[0] == 'waiting':
#             if event.key == 'enter':
#                 user_decision[0] = 'accepted'
#                 print("\n[Keyboard] Accepted.")
#             elif event.key == ' ':
#                 user_decision[0] = 'rejected'
#                 print("\n[Keyboard] Rejected.")

#     fig.canvas.mpl_connect('key_press_event', on_key)
#     active_line.set_data(waypoints[0, :], waypoints[1, :])

#     plt.show(block=False)
#     mng = plt.get_current_fig_manager()
#     try:
#         mng.window.showMaximized()
#     except Exception as e:
#         print(f"Could not maximize: {e}")
#     fig.canvas.draw()
#     fig.canvas.flush_events()

#     # ── Background replanning ──────────────────────────────────────────────────
#     def replan(trigger_step, apply_step, x0_replan, spec_switched):
#         try:
#             T_remaining = (N_total - apply_step -1) * dt
#             if T_remaining <= dt:
#                 print(f"[Replan t={trigger_step}] No time remaining, skipping.")
#                 return

#             shifted_spec = shift_spec(spec_current[0], apply_step)
#             max_step = int(T_remaining / dt) - 1

#             def clamp_window(match):
#                 method = match.group(1)
#                 a = min(int(match.group(2)), max_step)
#                 b = min(int(match.group(3)), max_step)
#                 return f".{method}({a}, {b})"
#             shifted_spec = re.sub(
#                 r'\.(eventually|always)\((\d+),\s*(\d+)\)',
#                 clamp_window, shifted_spec
#             )

#             solver = STLSolver(shifted_spec, scenario_objects, x0_replan, T_remaining)
#             x_new, u_new, rho, _, runtime = solver.generate_trajectory(
#                 dt, max_acc, max_speed, verbose=False
#             )

#             if x_new is not None and rho > 0:
#                 needs_interaction = spec_switched and not spec_switch_announced[0]
#                 if not needs_interaction:
#                     # ── Silent auto-accept: no UI, no candidate line ───────────
#                     with lock:
#                         new_trajectory[0] = (trigger_step, x_new, u_new, rho)
#                     user_decision[0] = 'accepted'
#                     print(f"[Replan t={trigger_step}] rho={rho:.3f} rt={runtime:.2f}s "
#                           f"→ auto-accepted (no spec switch)")
#                 else:
#                     # ── Spec switched for first time: show candidate + wait ────
#                     spec_switch_announced[0] = True
#                     with lock:
#                         new_trajectory[0] = (trigger_step, x_new, u_new, rho)
#                     candidate_line.set_data(x_new[0, 0:-1], x_new[1, 0:-1])
#                     branch_marker.set_data([x0_replan[0]], [x0_replan[1]])
#                     user_decision[0] = 'waiting'
#                     prompt_ax.set_visible(True)
#                     prompt_text_obj.set_text(
#                         f"New path available (rho={rho:.3f})    "
#                         f"[Enter/Yes] Accept   [Space/No] Reject    "
#                         f"(auto-reject at step {apply_step})"
#                     )
#                     timer_bar_patch.set_width(1.0)
#                     fig.canvas.draw()
#                     fig.canvas.flush_events()
#                     print(f"[Replan t={trigger_step}] rho={rho:.3f} rt={runtime:.2f}s "
#                           f"→ waiting for user (apply at step {apply_step})")

#                     if listen_for_yes_no is not None:
#                         def voice_listen():
#                             try:
#                                 from LLM.voice_openai import VoiceOpenAI
#                                 voice = VoiceOpenAI()
#                                 print(f"\n[Voice] 🎤 Say YES or NO...")
#                                 while user_decision[0] == 'waiting':
#                                     frames = voice.record_audio(duration=2, silence_threshold=2000, silence_duration=0.5)
#                                     audio_np = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32)
#                                     if np.sqrt(np.mean(audio_np**2)) < 500:
#                                         print("[Voice] No speech, listening again...")
#                                         continue
#                                     import tempfile, os
#                                     tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
#                                     voice.save_audio(frames, tmp.name); tmp.close()
#                                     text = voice.transcribe(tmp.name).lower(); os.unlink(tmp.name)
#                                     print(f"[Voice] You said: '{text}'")
#                                     if any(w in text for w in ['yes','yeah','sure','okay','yep','accept']):
#                                         if user_decision[0] == 'waiting':
#                                             user_decision[0] = 'accepted'; print("[Voice] Accepted.")
#                                         break
#                                     elif any(w in text for w in ['no','nope','reject','cancel']):
#                                         if user_decision[0] == 'waiting':
#                                             user_decision[0] = 'rejected'; print("[Voice] Rejected.")
#                                         break
#                                     else:
#                                         print("[Voice] Please say YES or NO.")
#                                 voice.close()
#                             except Exception as e:
#                                 print(f"[Voice] Error: {e}, use keyboard.")
#                         threading.Thread(target=voice_listen, daemon=True).start()
#             else:
#                 print(f"[Replan t={trigger_step}] Failed (rho={rho}), keeping current.")
#                 replan_log.append((trigger_step, None, False))

#         except Exception as e:
#             print(f"[Replan t={trigger_step}] Exception: {e}")
#             replan_log.append((trigger_step, None, False))
#         finally:
#             with lock:
#                 replan_running[0] = False

#     # ── Main loop ──────────────────────────────────────────────────────────────
#     print(f"\n{'='*60}")
#     print(f"Simulation: {N_total} steps, dt={dt}s, noise={noise_std}")
#     if switch_step is not None:
#         print(f"Spec switch at step {switch_step}")
#     print(f"{'='*60}\n")

#     for step in range(N_total):

#         # ── 1. APPLY ──────────────────────────────────────────────────────────
#         with lock:
#             pending = new_trajectory[0]

#         if pending is not None:
#             trigger_step, x_new, u_new, rho = pending
#             expected = trigger_step + REPLAN_BUFFER

#             if step < expected:
#                 # Only show timer bar when in interactive (spec-switch) mode
#                 if user_decision[0] == 'waiting':
#                     fraction = (expected - step) / REPLAN_BUFFER
#                     timer_bar_patch.set_width(fraction)

#             elif step == expected:
#                 decision = user_decision[0]
#                 if decision == 'accepted':
#                     steps_remaining = N_total - step
#                     cols = min(u_new.shape[1], steps_remaining)
#                     new_u_block = np.zeros((3, steps_remaining))
#                     new_u_block[:, :cols] = u_new[:, :cols]
#                     if cols < steps_remaining:
#                         new_u_block[:, cols:] = u_new[:, -1:]
#                     current_u[:, step:] = new_u_block

#                     prev_data  = active_plan_data[0]
#                     prev_start = active_plan_start[0]
#                     seg_len    = step - prev_start
#                     if seg_len > 0:
#                         if is_offline[0]:
#                             seg_end = min(seg_len, prev_data.shape[1])
#                             seg_xs, seg_ys = prev_data[0, :seg_end], prev_data[1, :seg_end]
#                         else:
#                             seg_end = min(seg_len + 1, prev_data.shape[1])
#                             seg_xs, seg_ys = prev_data[0, 0:seg_end], prev_data[1, 0:seg_end]
#                         if len(seg_xs) > 0:
#                             hl, = ax.plot(seg_xs, seg_ys, '-', color='purple', lw=2, zorder=101)
#                             history_lines.append(hl)

#                     active_plan_data[0]  = x_new
#                     active_plan_start[0] = step
#                     is_offline[0]        = False
#                     plan_tracking[0] = False
#                     active_line.set_data(x_new[0, 0:-1], x_new[1, 0:-1])
#                     replan_log.append((trigger_step, step, True))
#                     print(f"[Step {step:3d}] Replan APPLIED (trig={trigger_step})")
#                 else:
#                     reason = 'timeout' if decision == 'waiting' else 'rejected'
#                     replan_log.append((trigger_step, step, False))
#                     print(f"[Step {step:3d}] Replan {reason} (trig={trigger_step}), keeping current.")

#                 # Clean up UI (only matters for interactive mode, harmless otherwise)
#                 candidate_line.set_data([], [])
#                 branch_marker.set_data([], [])
#                 prompt_text_obj.set_text('')
#                 timer_bar_patch.set_width(0)
#                 prompt_ax.set_visible(False)
#                 user_decision[0] = None
#                 with lock: new_trajectory[0] = None

#             else:
#                 candidate_line.set_data([], [])
#                 branch_marker.set_data([], [])
#                 prompt_text_obj.set_text('')
#                 timer_bar_patch.set_width(0)
#                 prompt_ax.set_visible(False)
#                 user_decision[0] = None
#                 with lock: new_trajectory[0] = None
#                 print(f"[Step {step:3d}] Replan expired (trig={trigger_step}), discarding.")

#         # ── 2. PROPAGATE ──────────────────────────────────────────────────────
#         if step < N_total-1:
#             noise = np.zeros(6)
#             noise[:3] = np.random.normal(0, noise_std, 3)
#             x_true = A_tilde @ x_true + B_tilde @ current_u[:, step] + noise
#             position_history.append(x_true[:3].copy())
#             if show_no_mpc:
#                 nompc_x_true = A_tilde @ nompc_x_true + B_tilde @ all_u[:, step] + noise
#                 nompc_history.append(nompc_x_true[:3].copy())
#             print(f"[Step {step:3d}] true pos: ({x_true[0]:.3f}, {x_true[1]:.3f}, {x_true[2]:.3f})")
#         else:
#             print(f"[Step {step:3d}] final step, no propagation.")

#         # ── 3. SPEC SWITCH ────────────────────────────────────────────────────
#         if switch_step is not None and step == switch_step and spec_str_phase2 is not None:
#             spec_current[0] = spec_str_phase2
#             print(f"[Step {step:3d}] *** Spec switched to phase 2 ***")

#         # ── 4. TRIGGER REPLAN ─────────────────────────────────────────────────
#         if (spec_current[0] is not None
#                 and step % REPLAN_INTERVAL == 0
#                 and step > 0
#                 and step + REPLAN_BUFFER <= N_total - 2):
#             with lock:
#                 already_running = replan_running[0]
#             if not already_running:
#                 with lock: replan_running[0] = True
#                 # Only clear candidate UI if we're not already in interactive mode
#                 if user_decision[0] != 'waiting':
#                     candidate_line.set_data([], [])
#                     branch_marker.set_data([], [])
#                     prompt_text_obj.set_text('')
#                     timer_bar_patch.set_width(0)
#                     prompt_ax.set_visible(False)
#                     user_decision[0] = None

#                 apply_step  = step + REPLAN_BUFFER
#                 x_predicted = x_true.copy()
#                 for i in range(REPLAN_BUFFER - 1):
#                     idx = step + 1 + i
#                     if idx < N_total:
#                         x_predicted = A_tilde @ x_predicted + B_tilde @ current_u[:, idx]

#                 x0_replan = x_predicted.copy()
#                 print(f"[Step {step:3d}] Replanning triggered. "
#                       f"Predicted apply pos: ({x0_replan[0]:.3f}, {x0_replan[1]:.3f})")

#                 threading.Thread(
#                     target=replan, args=(step, apply_step, x0_replan, spec_current[0] != spec_str), daemon=True
#                 ).start()

#         # ── 5. VISUALISE ──────────────────────────────────────────────────────
#         print(f"[Vis step={step}] plan_line points={len(plan_xs)+1}, "
#             f"real_line points={len(position_history)}, "
#             f"plan_end=({waypoints[0,step]:.3f},{waypoints[1,step]:.3f}), "
#             f"real_end=({x_true[0]:.3f},{x_true[1]:.3f})")

#         if step < N_total - 1:
#             plan_xs.append(waypoints[0, step + 1])
#             plan_ys.append(waypoints[1, step + 1])
#             plan_line.set_data(plan_xs, plan_ys)

#         if plan_tracking[0]:
#             history_arr = np.array(position_history)
#             real_line.set_data(history_arr[:, 0], history_arr[:, 1])
#             real_dot.set_data([x_true[0]], [x_true[1]])
#         else:
#             history_arr = np.array(position_history[:-1]) if len(position_history) > 1 else np.array(position_history)
#             real_line.set_data(history_arr[:, 0], history_arr[:, 1])
#             current_pos = np.array(position_history[-2]) if len(position_history) > 1 else np.array(position_history[-1])
#             real_dot.set_data([current_pos[0]], [current_pos[1]])

#         if show_no_mpc and len(nompc_history) > 1:
#             na = np.array(nompc_history)
#             nompc_line.set_data(na[:, 0], na[:, 1])

#         fig.canvas.draw()
#         fig.canvas.flush_events()
#         plt.pause(dt)

#     # ── Done ──────────────────────────────────────────────────────────────────
#     print(f"\n{'='*60}\nSimulation complete.\nReplan log: {replan_log}\n{'='*60}\n")

#     if plan_tracking[0]:
#         history_arr = np.array(position_history)
#     else:
#         history_arr = np.array(position_history[:-1])
#     real_line.set_data(history_arr[:, 0], history_arr[:, 1])
#     plan_line.set_data(plan_xs, plan_ys)
#     if show_no_mpc:
#         na = np.array(nompc_history)
#         nompc_line.set_data(na[:, 0], na[:, 1])
#     fig.canvas.draw()
#     fig.canvas.flush_events()

#     input("Simulation finished. Press Enter to close.")
#     plt.close('all')
#     return position_history, replan_log























#version:new obj
# # -*- coding: utf-8 -*-
# import threading
# import re
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle

# from STL.STL_to_path import STLSolver, drone_dynamics

# REPLAN_INTERVAL = 3
# REPLAN_BUFFER   = 3


# def shift_spec(spec_str, step_offset):
#     def replace_window(match):
#         method = match.group(1)
#         a = max(0, int(match.group(2)) - step_offset)
#         b = max(0, int(match.group(3)) - step_offset)
#         return f".{method}({a}, {b})"
#     return re.sub(r'\.(eventually|always)\((\d+),\s*(\d+)\)', replace_window, spec_str)


# def simulate_deployment(
#     waypoints,
#     all_u,
#     scenario=None,
#     all_rho=None,
#     noise_std=0.05,
#     spec_str=None,
#     spec_str_phase2=None,   # optional: spec to switch to at switch_step
#     switch_step=None,        # step at which to switch spec (None = no switch)
#     dt=0.7,
#     max_acc=10,
#     max_speed=0.5,
#     use_voice=False,
#     show_no_mpc=False,
# ):
#     dyn     = drone_dynamics(dt=dt, max_acc=max_acc)
#     A_tilde = dyn.A_tilde
#     B_tilde = dyn.B_tilde

#     N_total          = all_u.shape[1]
#     scenario_objects = scenario.objects if scenario is not None else {}

#     current_u    = all_u.copy()
#     x_true       = waypoints[:6, 0].copy()
#     spec_current = [spec_str]   # mutable so replan closure sees updates

#     position_history = [x_true[:3].copy()]

#     # no-mpc comparison
#     nompc_x_true  = waypoints[:6, 0].copy()
#     nompc_history = [nompc_x_true[:3].copy()]

#     lock           = threading.Lock()
#     new_trajectory = [None]
#     replan_running = [False]
#     replan_log     = []

#     user_decision = [None]  # None / 'accepted' / 'rejected' / 'waiting'

#     spec_switch_announced = [False]  # True after the first interactive replan post-switch

#     active_plan_data  = [waypoints]
#     active_plan_start = [0]
#     is_offline        = [True]
#     history_lines     = []
#     plan_tracking = [True]

#     # voice
#     listen_for_yes_no = None
#     if use_voice:
#         try:
#             from LLM.voice_openai import listen_for_yes_no_voice
#             listen_for_yes_no = listen_for_yes_no_voice
#             print("[Voice] Voice interaction enabled.")
#         except Exception as e:
#             print(f"[Voice] Could not load voice module ({e}), keyboard only.")

#     # ── Visualisation ──────────────────────────────────────────────────────────
#     from visuals.visualization import Visualizer
#     vis_x = waypoints[:6, :]
#     visualizer = Visualizer(vis_x, scenario)

#     if all_rho is not None and len(all_rho) == vis_x.shape[1]:
#         fig, ax = visualizer.visualize_trajectory_rho_gradient_2d(all_rho)
#     else:
#         fig, ax = plt.subplots(figsize=(10, 10))

#     ax.set_title('Online MPC Simulation')

#     plan_line,      = ax.plot([], [], 'o-', color='orange', ms=4, lw=1.5,
#                               label='Planned (offline)', zorder=98)
#     real_line,      = ax.plot([], [], 'b--', lw=2, label='True trajectory', zorder=99)
#     real_dot,       = ax.plot([], [], 'bo',  ms=8,                          zorder=100)
#     active_line,    = ax.plot([], [], '-',   color='purple', lw=2,
#                               label='Current plan', zorder=101)
#     candidate_line, = ax.plot([], [], '--',  color='purple', lw=2,
#                               label='Candidate new path', zorder=102)
#     branch_marker,  = ax.plot([], [], 'o',   color='purple', ms=10,
#                               label='Branch point', zorder=103)
#     nompc_line,     = ax.plot([], [], 'r--', lw=1.5, label='',
#                               zorder=97, visible=show_no_mpc)

#     plan_xs = [waypoints[0, 0]]
#     plan_ys = [waypoints[1, 0]]
#     ax.legend(loc='upper left', fontsize=8)

#     prompt_ax = fig.add_axes([0.05, 0.01, 0.9, 0.05])
#     prompt_ax.set_xlim(0, 1)
#     prompt_ax.set_ylim(0, 1)
#     prompt_ax.axis('off')

#     prompt_text_obj = prompt_ax.text(
#         0.5, 0.95, '',
#         ha='center', va='top', fontsize=10,
#         bbox=dict(boxstyle='round', facecolor='lightyellow',
#                 alpha=0.9, edgecolor='gray'),
#     )
#     timer_bg_patch = Rectangle((0.0, 0.05), 1.0, 0.25,
#                                 transform=prompt_ax.transAxes,
#                                 facecolor='lightgray', edgecolor='none')
#     timer_bar_patch = Rectangle((0.0, 0.05), 1.0, 0.25,
#                                 transform=prompt_ax.transAxes,
#                                 facecolor='orange', edgecolor='none')
#     prompt_ax.add_patch(timer_bg_patch)
#     prompt_ax.add_patch(timer_bar_patch)
#     prompt_ax.set_visible(False)

#     def on_key(event):
#         # Only handle keyboard input when waiting (spec-switch interactive mode)
#         if user_decision[0] == 'waiting':
#             if event.key == 'enter':
#                 user_decision[0] = 'accepted'
#                 print("\n[Keyboard] Accepted.")
#             elif event.key == ' ':
#                 user_decision[0] = 'rejected'
#                 print("\n[Keyboard] Rejected.")

#     fig.canvas.mpl_connect('key_press_event', on_key)
#     active_line.set_data(waypoints[0, :], waypoints[1, :])

#     plt.show(block=False)
#     mng = plt.get_current_fig_manager()
#     try:
#         mng.window.showMaximized()
#     except Exception as e:
#         print(f"Could not maximize: {e}")
#     fig.canvas.draw()
#     fig.canvas.flush_events()

#     # ── Background replanning ──────────────────────────────────────────────────
#     def replan(trigger_step, apply_step, x0_replan, spec_switched):
#         try:
#             T_remaining = (N_total - apply_step -1) * dt
#             if T_remaining <= dt:
#                 print(f"[Replan t={trigger_step}] No time remaining, skipping.")
#                 return

#             shifted_spec = shift_spec(spec_current[0], apply_step)
#             max_step = int(T_remaining / dt) - 1

#             def clamp_window(match):
#                 method = match.group(1)
#                 a = min(int(match.group(2)), max_step)
#                 b = min(int(match.group(3)), max_step)
#                 return f".{method}({a}, {b})"
#             shifted_spec = re.sub(
#                 r'\.(eventually|always)\((\d+),\s*(\d+)\)',
#                 clamp_window, shifted_spec
#             )

#             solver = STLSolver(shifted_spec, scenario_objects, x0_replan, T_remaining)
#             x_new, u_new, rho, _, runtime = solver.generate_trajectory(
#                 dt, max_acc, max_speed, verbose=False
#             )

#             if x_new is not None and rho > 0:
#                 needs_interaction = spec_switched and not spec_switch_announced[0]
#                 if not needs_interaction:
#                     # ── Silent auto-accept: no UI, no candidate line ───────────
#                     with lock:
#                         new_trajectory[0] = (trigger_step, x_new, u_new, rho)
#                     user_decision[0] = 'accepted'
#                     print(f"[Replan t={trigger_step}] rho={rho:.3f} rt={runtime:.2f}s "
#                           f"→ auto-accepted (no spec switch)")
#                 else:
#                     # ── Spec switched for first time: show candidate + wait ────
#                     spec_switch_announced[0] = True
#                     with lock:
#                         new_trajectory[0] = (trigger_step, x_new, u_new, rho)
#                     candidate_line.set_data(x_new[0, 0:-1], x_new[1, 0:-1])
#                     branch_marker.set_data([x0_replan[0]], [x0_replan[1]])
#                     user_decision[0] = 'waiting'
#                     prompt_ax.set_visible(True)
#                     prompt_text_obj.set_text(
#                         f"New path available (rho={rho:.3f})    "
#                         f"[Enter/Yes] Accept   [Space/No] Reject    "
#                         f"(auto-reject at step {apply_step})"
#                     )
#                     timer_bar_patch.set_width(1.0)
#                     fig.canvas.draw()
#                     fig.canvas.flush_events()
#                     print(f"[Replan t={trigger_step}] rho={rho:.3f} rt={runtime:.2f}s "
#                           f"→ waiting for user (apply at step {apply_step})")

#                     if listen_for_yes_no is not None:
#                         def voice_listen():
#                             try:
#                                 from LLM.voice_openai import VoiceOpenAI
#                                 voice = VoiceOpenAI()
#                                 print(f"\n[Voice] 🎤 Say YES or NO...")
#                                 while user_decision[0] == 'waiting':
#                                     frames = voice.record_audio(duration=2, silence_threshold=2000, silence_duration=0.5)
#                                     audio_np = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32)
#                                     if np.sqrt(np.mean(audio_np**2)) < 500:
#                                         print("[Voice] No speech, listening again...")
#                                         continue
#                                     import tempfile, os
#                                     tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
#                                     voice.save_audio(frames, tmp.name); tmp.close()
#                                     text = voice.transcribe(tmp.name).lower(); os.unlink(tmp.name)
#                                     print(f"[Voice] You said: '{text}'")
#                                     if any(w in text for w in ['yes','yeah','sure','okay','yep','accept']):
#                                         if user_decision[0] == 'waiting':
#                                             user_decision[0] = 'accepted'; print("[Voice] Accepted.")
#                                         break
#                                     elif any(w in text for w in ['no','nope','reject','cancel']):
#                                         if user_decision[0] == 'waiting':
#                                             user_decision[0] = 'rejected'; print("[Voice] Rejected.")
#                                         break
#                                     else:
#                                         print("[Voice] Please say YES or NO.")
#                                 voice.close()
#                             except Exception as e:
#                                 print(f"[Voice] Error: {e}, use keyboard.")
#                         threading.Thread(target=voice_listen, daemon=True).start()
#             else:
#                 print(f"[Replan t={trigger_step}] Failed (rho={rho}), keeping current.")
#                 replan_log.append((trigger_step, None, False))

#         except Exception as e:
#             print(f"[Replan t={trigger_step}] Exception: {e}")
#             replan_log.append((trigger_step, None, False))
#         finally:
#             with lock:
#                 replan_running[0] = False

#     # ── Main loop ──────────────────────────────────────────────────────────────
#     print(f"\n{'='*60}")
#     print(f"Simulation: {N_total} steps, dt={dt}s, noise={noise_std}")
#     if switch_step is not None:
#         print(f"Spec switch at step {switch_step}")
#     print(f"{'='*60}\n")

#     for step in range(N_total):

#         # ── 1. APPLY ──────────────────────────────────────────────────────────
#         with lock:
#             pending = new_trajectory[0]

#         if pending is not None:
#             trigger_step, x_new, u_new, rho = pending
#             expected = trigger_step + REPLAN_BUFFER

#             if step < expected:
#                 # Only show timer bar when in interactive (spec-switch) mode
#                 if user_decision[0] == 'waiting':
#                     fraction = (expected - step) / REPLAN_BUFFER
#                     timer_bar_patch.set_width(fraction)

#             elif step == expected:
#                 decision = user_decision[0]
#                 if decision == 'accepted':
#                     steps_remaining = N_total - step
#                     cols = min(u_new.shape[1], steps_remaining)
#                     new_u_block = np.zeros((3, steps_remaining))
#                     new_u_block[:, :cols] = u_new[:, :cols]
#                     if cols < steps_remaining:
#                         new_u_block[:, cols:] = u_new[:, -1:]
#                     current_u[:, step:] = new_u_block

#                     prev_data  = active_plan_data[0]
#                     prev_start = active_plan_start[0]
#                     seg_len    = step - prev_start
#                     if seg_len > 0:
#                         if is_offline[0]:
#                             seg_end = min(seg_len, prev_data.shape[1])
#                             seg_xs, seg_ys = prev_data[0, :seg_end], prev_data[1, :seg_end]
#                         else:
#                             seg_end = min(seg_len + 1, prev_data.shape[1])
#                             seg_xs, seg_ys = prev_data[0, 0:seg_end], prev_data[1, 0:seg_end]
#                         if len(seg_xs) > 0:
#                             hl, = ax.plot(seg_xs, seg_ys, '-', color='purple', lw=2, zorder=101)
#                             history_lines.append(hl)

#                     active_plan_data[0]  = x_new
#                     active_plan_start[0] = step
#                     is_offline[0]        = False
#                     plan_tracking[0] = False
#                     active_line.set_data(x_new[0, 0:-1], x_new[1, 0:-1])
#                     replan_log.append((trigger_step, step, True))
#                     print(f"[Step {step:3d}] Replan APPLIED (trig={trigger_step})")
#                 else:
#                     reason = 'timeout' if decision == 'waiting' else 'rejected'
#                     replan_log.append((trigger_step, step, False))
#                     print(f"[Step {step:3d}] Replan {reason} (trig={trigger_step}), keeping current.")

#                 # Clean up UI (only matters for interactive mode, harmless otherwise)
#                 candidate_line.set_data([], [])
#                 branch_marker.set_data([], [])
#                 prompt_text_obj.set_text('')
#                 timer_bar_patch.set_width(0)
#                 prompt_ax.set_visible(False)
#                 user_decision[0] = None
#                 with lock: new_trajectory[0] = None

#             else:
#                 candidate_line.set_data([], [])
#                 branch_marker.set_data([], [])
#                 prompt_text_obj.set_text('')
#                 timer_bar_patch.set_width(0)
#                 prompt_ax.set_visible(False)
#                 user_decision[0] = None
#                 with lock: new_trajectory[0] = None
#                 print(f"[Step {step:3d}] Replan expired (trig={trigger_step}), discarding.")

#         # ── 2. PROPAGATE ──────────────────────────────────────────────────────
#         if step < N_total-1:
#             noise = np.zeros(6)
#             noise[:3] = np.random.normal(0, noise_std, 3)
#             x_true = A_tilde @ x_true + B_tilde @ current_u[:, step] + noise
#             position_history.append(x_true[:3].copy())
#             if show_no_mpc:
#                 nompc_x_true = A_tilde @ nompc_x_true + B_tilde @ all_u[:, step] + noise
#                 nompc_history.append(nompc_x_true[:3].copy())
#             print(f"[Step {step:3d}] true pos: ({x_true[0]:.3f}, {x_true[1]:.3f}, {x_true[2]:.3f})")
#         else:
#             print(f"[Step {step:3d}] final step, no propagation.")

#         # ── 3. SPEC SWITCH ────────────────────────────────────────────────────
#         if switch_step is not None and step == switch_step and spec_str_phase2 is not None:
#             spec_current[0] = spec_str_phase2
#             print(f"[Step {step:3d}] *** Spec switched to phase 2 ***")

#             # Load new obstacles from scenario.objects_phase2 (keys not in current objects)
#             if scenario is not None and hasattr(scenario, 'objects_phase2'):
#                 new_obs = {k: v for k, v in scenario.objects_phase2.items()
#                            if k not in scenario_objects}
#                 for name, bounds in new_obs.items():
#                     scenario_objects[name] = bounds
#                     xmin, xmax, ymin, ymax = bounds[0], bounds[1], bounds[2], bounds[3]
#                     rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
#                                      linewidth=2, edgecolor='red', facecolor='salmon',
#                                      alpha=0.5, zorder=50, label='')
#                     ax.add_patch(rect)
#                     print(f"[Step {step:3d}] New obstacle added: '{name}' {bounds[:4]}")
#                 if new_obs:
#                     ax.legend(loc='upper left', fontsize=8)
#                     fig.canvas.draw()
#                     fig.canvas.flush_events()

#         # ── 4. TRIGGER REPLAN ─────────────────────────────────────────────────
#         if (spec_current[0] is not None
#                 and step % REPLAN_INTERVAL == 0
#                 and step > 0
#                 and step + REPLAN_BUFFER <= N_total - 2):
#             with lock:
#                 already_running = replan_running[0]
#             if not already_running:
#                 with lock: replan_running[0] = True
#                 # Only clear candidate UI if we're not already in interactive mode
#                 if user_decision[0] != 'waiting':
#                     candidate_line.set_data([], [])
#                     branch_marker.set_data([], [])
#                     prompt_text_obj.set_text('')
#                     timer_bar_patch.set_width(0)
#                     prompt_ax.set_visible(False)
#                     user_decision[0] = None

#                 apply_step  = step + REPLAN_BUFFER
#                 x_predicted = x_true.copy()
#                 for i in range(REPLAN_BUFFER - 1):
#                     idx = step + 1 + i
#                     if idx < N_total:
#                         x_predicted = A_tilde @ x_predicted + B_tilde @ current_u[:, idx]

#                 x0_replan = x_predicted.copy()
#                 print(f"[Step {step:3d}] Replanning triggered. "
#                       f"Predicted apply pos: ({x0_replan[0]:.3f}, {x0_replan[1]:.3f})")

#                 threading.Thread(
#                     target=replan, args=(step, apply_step, x0_replan, spec_current[0] != spec_str), daemon=True
#                 ).start()

#         # ── 5. VISUALISE ──────────────────────────────────────────────────────
#         print(f"[Vis step={step}] plan_line points={len(plan_xs)+1}, "
#             f"real_line points={len(position_history)}, "
#             f"plan_end=({waypoints[0,step]:.3f},{waypoints[1,step]:.3f}), "
#             f"real_end=({x_true[0]:.3f},{x_true[1]:.3f})")

#         if step < N_total - 1:
#             plan_xs.append(waypoints[0, step + 1])
#             plan_ys.append(waypoints[1, step + 1])
#             plan_line.set_data(plan_xs, plan_ys)

#         if plan_tracking[0]:
#             history_arr = np.array(position_history)
#             real_line.set_data(history_arr[:, 0], history_arr[:, 1])
#             real_dot.set_data([x_true[0]], [x_true[1]])
#         else:
#             history_arr = np.array(position_history[:-1]) if len(position_history) > 1 else np.array(position_history)
#             real_line.set_data(history_arr[:, 0], history_arr[:, 1])
#             current_pos = np.array(position_history[-2]) if len(position_history) > 1 else np.array(position_history[-1])
#             real_dot.set_data([current_pos[0]], [current_pos[1]])

#         if show_no_mpc and len(nompc_history) > 1:
#             na = np.array(nompc_history)
#             nompc_line.set_data(na[:, 0], na[:, 1])

#         fig.canvas.draw()
#         fig.canvas.flush_events()
#         plt.pause(dt)

#     # ── Done ──────────────────────────────────────────────────────────────────
#     print(f"\n{'='*60}\nSimulation complete.\nReplan log: {replan_log}\n{'='*60}\n")

#     if plan_tracking[0]:
#         history_arr = np.array(position_history)
#     else:
#         history_arr = np.array(position_history[:-1])
#     real_line.set_data(history_arr[:, 0], history_arr[:, 1])
#     plan_line.set_data(plan_xs, plan_ys)
#     if show_no_mpc:
#         na = np.array(nompc_history)
#         nompc_line.set_data(na[:, 0], na[:, 1])
#     fig.canvas.draw()
#     fig.canvas.flush_events()

#     input("Simulation finished. Press Enter to close.")
#     plt.close('all')
#     return position_history, replan_log








#version: new voice
# -*- coding: utf-8 -*-
import threading
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from STL.STL_to_path import STLSolver, drone_dynamics

REPLAN_INTERVAL = 5
REPLAN_BUFFER   = 5


def shift_spec(spec_str, step_offset):
    def replace_window(match):
        method = match.group(1)
        a = max(0, int(match.group(2)) - step_offset)
        b = max(0, int(match.group(3)) - step_offset)
        return f".{method}({a}, {b})"
    return re.sub(r'\.(eventually|always)\((\d+),\s*(\d+)\)', replace_window, spec_str)


def simulate_deployment(
    waypoints,
    all_u,
    scenario=None,
    all_rho=None,
    noise_std=0.05,
    spec_str=None,
    spec_str_phase2=None,   # optional: spec to switch to at switch_step
    switch_step=None,        # step at which to switch spec (None = no switch)
    dt=0.7,
    max_acc=10,
    max_speed=0.5,
    use_voice=False,
    show_no_mpc=False,
):
    dyn     = drone_dynamics(dt=dt, max_acc=max_acc)
    A_tilde = dyn.A_tilde
    B_tilde = dyn.B_tilde

    N_total          = all_u.shape[1]
    scenario_objects = scenario.objects if scenario is not None else {}

    current_u    = all_u.copy()
    x_true       = waypoints[:6, 0].copy()
    spec_current = [spec_str]   # mutable so replan closure sees updates

    position_history = [x_true[:3].copy()]

    # no-mpc comparison
    nompc_x_true  = waypoints[:6, 0].copy()
    nompc_history = [nompc_x_true[:3].copy()]

    lock           = threading.Lock()
    new_trajectory = [None]
    replan_running = [False]
    replan_log     = []

    user_decision = [None]  # None / 'accepted' / 'rejected' / 'waiting'

    spec_switch_announced = [False]  # True after the first interactive replan post-switch

    active_plan_data  = [waypoints]
    active_plan_start = [0]
    is_offline        = [True]
    history_lines     = []
    plan_tracking     = [True]

    # ── Voice: load model + start background stream BEFORE simulation loop ────
    voice_listener = None
    if use_voice:
        try:
            from LLM.voice_vad import VoiceListener
            voice_listener = VoiceListener(model_size='base')
            print("[Voice] Voice interaction enabled.")
        except Exception as e:
            print(f"[Voice] Could not load voice module ({e}), keyboard only.")

    # ── Visualisation ──────────────────────────────────────────────────────────
    from visuals.visualization import Visualizer
    vis_x = waypoints[:6, :]
    visualizer = Visualizer(vis_x, scenario)

    if all_rho is not None and len(all_rho) == vis_x.shape[1]:
        fig, ax = visualizer.visualize_trajectory_rho_gradient_2d(all_rho)
    else:
        fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_title('Online MPC Simulation')

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
    nompc_line,     = ax.plot([], [], 'r--', lw=1.5, label='_nolegend_',
                              zorder=97, visible=show_no_mpc)

    plan_xs = [waypoints[0, 0]]
    plan_ys = [waypoints[1, 0]]
    ax.legend(loc='upper left', fontsize=8)

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
    timer_bg_patch = Rectangle((0.0, 0.05), 1.0, 0.25,
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
    active_line.set_data(waypoints[0, :], waypoints[1, :])

    plt.show(block=False)
    mng = plt.get_current_fig_manager()
    try:
        mng.window.showMaximized()
    except Exception as e:
        print(f"Could not maximize: {e}")
    fig.canvas.draw()
    fig.canvas.flush_events()

    # ── Background replanning ──────────────────────────────────────────────────
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
                needs_interaction = spec_switched and not spec_switch_announced[0]

                if not needs_interaction:
                    # ── Silent auto-accept ─────────────────────────────────────
                    with lock:
                        new_trajectory[0] = (trigger_step, x_new, u_new, rho)
                    user_decision[0] = 'accepted'
                    print(f"[Replan t={trigger_step}] rho={rho:.3f} rt={runtime:.2f}s "
                          f"→ auto-accepted")
                else:
                    # ── Spec switched: show candidate + wait for user ──────────
                    spec_switch_announced[0] = True
                    with lock:
                        new_trajectory[0] = (trigger_step, x_new, u_new, rho)
                    candidate_line.set_data(x_new[0, 0:-1], x_new[1, 0:-1])
                    branch_marker.set_data([x0_replan[0]], [x0_replan[1]])
                    user_decision[0] = 'waiting'
                    prompt_ax.set_visible(True)
                    prompt_text_obj.set_text(
                        f"New path available (rho={rho:.3f})    "
                        f"[Enter/Yes] Accept   [Space/No] Reject    "
                        f"(auto-reject at step {apply_step})"
                    )
                    timer_bar_patch.set_width(1.0)
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    print(f"[Replan t={trigger_step}] rho={rho:.3f} rt={runtime:.2f}s "
                          f"→ waiting for user (apply at step {apply_step})")

                    # ── Trigger voice listener (already warm, no init delay) ───
                    if voice_listener is not None:
                        voice_listener.start_listening(user_decision)

            else:
                print(f"[Replan t={trigger_step}] Failed (rho={rho}), keeping current.")
                replan_log.append((trigger_step, None, False))

        except Exception as e:
            print(f"[Replan t={trigger_step}] Exception: {e}")
            replan_log.append((trigger_step, None, False))
        finally:
            with lock:
                replan_running[0] = False

    # ── Main loop ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Simulation: {N_total} steps, dt={dt}s, noise={noise_std}")
    if switch_step is not None:
        print(f"Spec switch at step {switch_step}")
    print(f"{'='*60}\n")

    for step in range(N_total):

        # ── 1. APPLY ──────────────────────────────────────────────────────────
        with lock:
            pending = new_trajectory[0]

        if pending is not None:
            trigger_step, x_new, u_new, rho = pending
            expected = trigger_step + REPLAN_BUFFER

            if step < expected:
                if user_decision[0] == 'waiting':
                    fraction = (expected - step) / REPLAN_BUFFER
                    timer_bar_patch.set_width(fraction)

            elif step == expected:
                decision = user_decision[0]

                # Stop voice listener if still running
                if voice_listener is not None:
                    voice_listener.stop_listening()

                if decision == 'accepted':
                    steps_remaining = N_total - step
                    cols = min(u_new.shape[1], steps_remaining)
                    new_u_block = np.zeros((3, steps_remaining))
                    new_u_block[:, :cols] = u_new[:, :cols]
                    if cols < steps_remaining:
                        new_u_block[:, cols:] = u_new[:, -1:]
                    current_u[:, step:] = new_u_block

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
                    replan_log.append((trigger_step, step, True))
                    print(f"[Step {step:3d}] Replan APPLIED (trig={trigger_step})")
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
                with lock: new_trajectory[0] = None

            else:
                if voice_listener is not None:
                    voice_listener.stop_listening()
                candidate_line.set_data([], [])
                branch_marker.set_data([], [])
                prompt_text_obj.set_text('')
                timer_bar_patch.set_width(0)
                prompt_ax.set_visible(False)
                user_decision[0] = None
                with lock: new_trajectory[0] = None
                print(f"[Step {step:3d}] Replan expired (trig={trigger_step}), discarding.")

        # ── 2. PROPAGATE ──────────────────────────────────────────────────────
        if step < N_total - 1:
            noise = np.zeros(6)
            noise[:3] = np.random.normal(0, noise_std, 3)
            x_true = A_tilde @ x_true + B_tilde @ current_u[:, step] + noise
            position_history.append(x_true[:3].copy())
            if show_no_mpc:
                nompc_x_true = A_tilde @ nompc_x_true + B_tilde @ all_u[:, step] + noise
                nompc_history.append(nompc_x_true[:3].copy())
            print(f"[Step {step:3d}] true pos: ({x_true[0]:.3f}, {x_true[1]:.3f}, {x_true[2]:.3f})")
        else:
            print(f"[Step {step:3d}] final step, no propagation.")

        # ── 3. SPEC SWITCH + UPDATE OBJECTS + VISUALISE NEW OBSTACLES ─────────
        if switch_step is not None and step == switch_step and spec_str_phase2 is not None:
            spec_current[0] = spec_str_phase2
            print(f"[Step {step:3d}] *** Spec switched to phase 2 ***")

            if scenario is not None and hasattr(scenario, 'objects_phase2'):
                new_obs = {k: v for k, v in scenario.objects_phase2.items()
                           if k not in scenario_objects}
                for name, bounds in new_obs.items():
                    scenario_objects[name] = bounds
                    xmin, xmax, ymin, ymax = bounds[0], bounds[1], bounds[2], bounds[3]
                    rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=2, edgecolor='red', facecolor='salmon',
                                     alpha=0.5, zorder=50, label='_nolegend_')
                    ax.add_patch(rect)
                    print(f"[Step {step:3d}] New obstacle added: '{name}' {bounds[:4]}")
                if new_obs:
                    fig.canvas.draw()
                    fig.canvas.flush_events()

        # ── 4. TRIGGER REPLAN ─────────────────────────────────────────────────
        if (spec_current[0] is not None
                and step % REPLAN_INTERVAL == 0
                and step > 0
                and step + REPLAN_BUFFER <= N_total - 2):
            with lock:
                already_running = replan_running[0]
            if not already_running:
                with lock: replan_running[0] = True
                if user_decision[0] != 'waiting':
                    candidate_line.set_data([], [])
                    branch_marker.set_data([], [])
                    prompt_text_obj.set_text('')
                    timer_bar_patch.set_width(0)
                    prompt_ax.set_visible(False)
                    user_decision[0] = None

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

        # ── 5. VISUALISE ──────────────────────────────────────────────────────
        print(f"[Vis step={step}] plan_line points={len(plan_xs)+1}, "
              f"real_line points={len(position_history)}, "
              f"plan_end=({waypoints[0,step]:.3f},{waypoints[1,step]:.3f}), "
              f"real_end=({x_true[0]:.3f},{x_true[1]:.3f})")

        if step < N_total - 1:
            plan_xs.append(waypoints[0, step + 1])
            plan_ys.append(waypoints[1, step + 1])
            plan_line.set_data(plan_xs, plan_ys)

        if plan_tracking[0]:
            history_arr = np.array(position_history)
            real_line.set_data(history_arr[:, 0], history_arr[:, 1])
            real_dot.set_data([x_true[0]], [x_true[1]])
        else:
            history_arr = (np.array(position_history[:-1]) if len(position_history) > 1
                           else np.array(position_history))
            real_line.set_data(history_arr[:, 0], history_arr[:, 1])
            current_pos = (np.array(position_history[-2]) if len(position_history) > 1
                           else np.array(position_history[-1]))
            real_dot.set_data([current_pos[0]], [current_pos[1]])

        if show_no_mpc and len(nompc_history) > 1:
            na = np.array(nompc_history)
            nompc_line.set_data(na[:, 0], na[:, 1])

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(dt)

    # ── Done ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nSimulation complete.\nReplan log: {replan_log}\n{'='*60}\n")

    if voice_listener is not None:
        voice_listener.close()

    if plan_tracking[0]:
        history_arr = np.array(position_history)
    else:
        history_arr = np.array(position_history[:-1])
    real_line.set_data(history_arr[:, 0], history_arr[:, 1])
    plan_line.set_data(plan_xs, plan_ys)
    if show_no_mpc:
        na = np.array(nompc_history)
        nompc_line.set_data(na[:, 0], na[:, 1])
    fig.canvas.draw()
    fig.canvas.flush_events()

    input("Simulation finished. Press Enter to close.")
    plt.close('all')
    return position_history, replan_log