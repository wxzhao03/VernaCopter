# -*- coding: utf-8 -*-
import multiprocessing as mp
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection

from STL.STL_to_path import STLSolver, drone_dynamics

REPLAN_INTERVAL        = 2   # how often to trigger replanning (every N steps)
REPLAN_BUFFER_AUTO     = 2   # buffer for auto-correction replans
REPLAN_BUFFER_INTERACT = 6   # buffer for interactive (spec-switch) replans
EXTENSION_STEPS        = 10  # extra steps added when voice goal can't be reached in time


def shift_spec(spec_str, step_offset):
    def replace_window(match):
        method = match.group(1)
        a = max(0, int(match.group(2)) - step_offset)
        b = max(0, int(match.group(3)) - step_offset)
        return f".{method}({a}, {b})"
    return re.sub(r'\.(eventually|always)\((\d+),\s*(\d+)\)', replace_window, spec_str)


def _persistent_worker(task_queue, result_queue):
    """
    Long-lived worker process. Imports and initializes everything once,
    then loops waiting for replan tasks. Avoids per-replan startup cost
    (pydrake warning, pybullet build, Gurobi license check).
    """
    from STL.STL_to_path import STLSolver, drone_dynamics
    import gurobipy

    while True:
        task = task_queue.get()
        if task is None:
            break

        (trigger_step, apply_step, x0_replan,
         spec_str, scenario_objects, N_total, dt, max_acc, max_speed,
         spec_switched, spec_switch_announced, apply_buffer, map_bounds) = task

        try:
            T_remaining = (N_total - apply_step - 2) * dt
            if T_remaining < dt:
                result_queue.put(('skip', trigger_step))
                continue

            shifted_spec = shift_spec(spec_str, apply_step + 1)

            solver = STLSolver(shifted_spec, scenario_objects, x0_replan, T_remaining,
                               map_bounds=map_bounds)
            x_new, u_new, rho, rho_series, runtime = solver.generate_trajectory(
                dt, max_acc, max_speed, verbose=False
            )

            if x_new is not None and rho > 0:
                needs_interaction = spec_switched and not spec_switch_announced
                result_queue.put(('success', trigger_step, x_new, u_new, rho, rho_series,
                                  runtime, needs_interaction, x0_replan, apply_buffer))
            else:
                result_queue.put(('failed', trigger_step, rho))

        except Exception as e:
            result_queue.put(('error', trigger_step, str(e)))


def simulate_deployment(
    waypoints,
    all_u,
    scenario=None,
    all_rho=None,
    noise_std=0.05,
    spec_str=None,
    spec_str_phase2=None,
    switch_step=None,
    dt=0.7,
    max_acc=10,
    max_speed=0.5,
    use_voice=False,
    show_no_mpc=False,
):
    dyn     = drone_dynamics(dt=dt, max_acc=max_acc)
    A_tilde = dyn.A_tilde
    B_tilde = dyn.B_tilde

    _N_total         = [all_u.shape[1]]   # mutable so loop/closures share one value
    scenario_objects = scenario.objects if scenario is not None else {}
    map_bounds       = getattr(scenario, 'map_bounds', None)

    current_u    = all_u.copy()
    x_true       = waypoints[:6, 0].copy()
    spec_current = [spec_str]

    position_history = [x_true[:3].copy()]

    nompc_x_true  = waypoints[:6, 0].copy()
    nompc_history = [nompc_x_true[:3].copy()]

    task_queue     = mp.Queue()
    result_queue   = mp.Queue()
    new_trajectory = [None]
    replan_running = [False]
    replan_log     = []

    user_decision       = [None]
    _current_step       = [0]    # shared with voice on_utterance closure
    pending_voice_goal  = [None] # goal name from latest voice instruction
    plan_end_step       = [None] # global step at which current plan's u_new runs out

    spec_switch_announced = [False]
    spec_switch_accepted  = [False]

    # ── Start persistent worker process ───────────────────────────────────────
    worker = mp.Process(target=_persistent_worker,
                        args=(task_queue, result_queue), daemon=True)
    worker.start()
    # print("[Worker] Persistent replan worker started.")

    # ── Voice ─────────────────────────────────────────────────────────────────
    def _build_spec_for_goal(goal_name, objects, N_total):
        """Build an STL spec string targeting goal_name, avoiding all obstacles.
        Uses N_total-1 as time index: after shift_spec offsets by (apply_step+1),
        the shifted index equals the solver's N exactly (same as predefined specs).
        """
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

    voice_listener = None
    if use_voice:
        try:
            from LLM.voice_vad import VoiceListener
            from LLM.voice_interpreter import VoiceInterpreter
            import threading as _threading

            goal_names = [k for k in scenario_objects if 'goal' in k.lower()]
            interpreter = VoiceInterpreter(goal_names, GPT_model="gpt-4o")

            def on_utterance(text):
                # when waiting for accept/reject: only handle yes/no, ignore new instructions
                if user_decision[0] == 'waiting':
                    def _classify_decision():
                        result = interpreter.interpret(text)
                        if result == 'yes':
                            user_decision[0] = 'accepted'
                            print("[Voice] Accepted.")
                        elif result == 'no':
                            user_decision[0] = 'rejected'
                            print("[Voice] Rejected.")
                    _threading.Thread(target=_classify_decision, daemon=True).start()
                    return

                # otherwise: interpret as potential new instruction
                def _classify_instruction():
                    result = interpreter.interpret(text)
                    if result in (None, 'yes', 'no'):
                        return  # not a goal instruction, ignore
                    # result is a goal name
                    new_spec = _build_spec_for_goal(result, scenario_objects, _N_total[0])
                    # print(f"[Voice] New spec for '{result}': {new_spec}")
                    spec_switch_announced[0] = False
                    spec_switch_accepted[0]  = False
                    spec_current[0] = new_spec
                    pending_voice_goal[0] = result
                    # print("[Voice] spec_current updated — takes effect on next replan.")

                _threading.Thread(target=_classify_instruction, daemon=True).start()

            voice_listener = VoiceListener(on_utterance=on_utterance)
            print("[Voice] Voice interaction enabled.")

            # Pre-generate TTS prompt audio to eliminate API delay during simulation
            _tts_prompt_path = [None]
            def _pregen_tts():
                try:
                    import tempfile
                    from openai import OpenAI
                    import pygame
                    client = OpenAI()
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                    _tts_prompt_path[0] = tmp.name
                    tmp.close()
                    with client.audio.speech.with_streaming_response.create(
                        model="tts-1", voice="nova",
                        input="Accept the trajectory? Please say yes or no."
                    ) as resp:
                        resp.stream_to_file(_tts_prompt_path[0])
                    pygame.mixer.init()
                    # print("[TTS] Prompt audio pre-generated.")
                except Exception as e:
                    # print(f"[TTS] Pre-generation failed: {e}")
                    pass
            _threading.Thread(target=_pregen_tts, daemon=True).start()

        except Exception as e:
            # print(f"[Voice] Could not load voice module ({e}), keyboard only.")
            _tts_prompt_path = [None]

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
                              label='_nolegend_', zorder=98, visible=False)
    real_line,      = ax.plot([], [], 'b--', lw=2, label='True trajectory', zorder=99)
    real_dot,       = ax.plot([], [], 'bo',  ms=8,                          zorder=100)
    active_line,    = ax.plot([], [], '-',   color='purple', lw=2,
                              label='_nolegend_', zorder=101, visible=False)
    candidate_line, = ax.plot([], [], '--',  color='purple', lw=2,
                              label='Candidate new path', zorder=102)
    branch_marker,  = ax.plot([], [], 'o',   color='purple', ms=10,
                              label='Branch point', zorder=103)
    nompc_line,     = ax.plot([], [], 'r--', lw=1.5, label='_nolegend_',
                              zorder=97, visible=show_no_mpc)

    plan_xs = [waypoints[0, 0]]
    plan_ys = [waypoints[1, 0]]
    ax.legend(loc='lower right', fontsize=8)

    prompt_ax = fig.add_axes([0.05, 0.01, 0.9, 0.05])
    prompt_ax.set_xlim(0, 1)
    prompt_ax.set_ylim(0, 1)
    prompt_ax.axis('off')

    prompt_text_obj = prompt_ax.text(
        0.5, 0.95, '',
        ha='center', va='top', fontsize=16, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightyellow',
                alpha=0.9, edgecolor='gray', linewidth=2),
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

    # Countdown text: top-left outside axes, in figure coordinates
    countdown_text = fig.text(
        0.01, 0.97, '',
        ha='left', va='top', fontsize=11, color='darkgreen',
        bbox=dict(boxstyle='round', facecolor='honeydew', alpha=0.85, edgecolor='green'),
    )

    plt.show(block=False)
    # mng = plt.get_current_fig_manager()
    # try:
    #     mng.window.showMaximized()
    # except Exception as e:
    #     print(f"Could not maximize: {e}")
    fig.canvas.draw()
    fig.canvas.flush_events()

    # ── Main loop ──────────────────────────────────────────────────────────────
    step = 0
    while step < _N_total[0]:
        _current_step[0] = step

        # ── 0. CHECK REPLAN RESULT ────────────────────────────────────────────
        if not result_queue.empty():
            msg = result_queue.get_nowait()
            if msg[0] == 'success':
                _, trigger_step_r, x_new_r, u_new_r, rho_r, rho_series_r, \
                    runtime_r, needs_interaction_r, x0_r, apply_buffer_r = msg
                if not needs_interaction_r:
                    new_trajectory[0] = (trigger_step_r, x_new_r, u_new_r,
                                         rho_r, rho_series_r, apply_buffer_r)
                    user_decision[0] = 'accepted'
                    # print(f"[Replan t={trigger_step_r}] rho={rho_r:.3f} "
                    #       f"rt={runtime_r:.2f}s → auto-accepted")
                else:
                    spec_switch_announced[0] = True
                    new_trajectory[0] = (trigger_step_r, x_new_r, u_new_r,
                                         rho_r, rho_series_r, apply_buffer_r)
                    candidate_line.set_data(x_new_r[0, :], x_new_r[1, :])
                    branch_marker.set_data([x0_r[0]], [x0_r[1]])
                    user_decision[0] = 'waiting'
                    prompt_ax.set_visible(True)
                    prompt_text_obj.set_text(
                        "Accept the trajectory?    Please say YES or NO"
                    )
                    timer_bar_patch.set_width(1.0)
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    # print(f"[Replan t={trigger_step_r}] rho={rho_r:.3f} "
                    #       f"rt={runtime_r:.2f}s → waiting for user "
                    #       f"(apply at step {trigger_step_r + apply_buffer_r})")
                    def _play_prompt():
                        try:
                            import pygame
                            if _tts_prompt_path[0]:
                                pygame.mixer.music.load(_tts_prompt_path[0])
                                pygame.mixer.music.play()
                            else:
                                from LLM.voice_openai import VoiceOpenAI as _V
                                v = _V()
                                v.speak("Accept the trajectory? Please say yes or no.")
                                v.close()
                        except Exception as e:
                            print(f"[TTS] Playback error: {e}")
                    _threading.Thread(target=_play_prompt, daemon=True).start()
                replan_running[0] = False
            elif msg[0] in ('failed', 'skip', 'error'):
                if msg[0] == 'failed':
                    print(f"[Replan t={msg[1]}] Failed (rho={msg[2]}), keeping current.")
                elif msg[0] == 'skip':
                    print(f"[Replan t={msg[1]}] No time remaining, skipping.")
                else:
                    print(f"[Replan t={msg[1]}] Exception: {msg[2]}")
                replan_log.append((msg[1], None, False))
                replan_running[0] = False

                # ── Auto-extend when voice goal is unreachable in remaining time ──
                if pending_voice_goal[0] is not None:
                    extra = EXTENSION_STEPS
                    _N_total[0] += extra
                    current_u = np.hstack([current_u, np.zeros((3, extra))])
                    new_spec = _build_spec_for_goal(
                        pending_voice_goal[0], scenario_objects, _N_total[0])
                    spec_current[0] = new_spec
                    print(f"[Extend] Not enough time for '{pending_voice_goal[0]}'. "
                          f"Extended by {extra} steps → N_total={_N_total[0]}. "
                          f"Rebuilding spec.")

        # ── 1. PROPAGATE ──────────────────────────────────────────────────────
        if step < _N_total[0] - 1:
            noise = np.zeros(6)
            noise[:3] = np.random.normal(0, noise_std, 3)
            x_true = A_tilde @ x_true + B_tilde @ current_u[:, step] + noise
            if plan_end_step[0] is not None and step >= plan_end_step[0]:
                x_true[3:6] = np.zeros(3)  # hover: zero velocity after plan ends
            position_history.append(x_true[:3].copy())
            if show_no_mpc:
                nompc_x_true = A_tilde @ nompc_x_true + B_tilde @ all_u[:, step] + noise
                nompc_history.append(nompc_x_true[:3].copy())
            # print(f"[Step {step:3d}] true pos: ({x_true[0]:.3f}, {x_true[1]:.3f}, {x_true[2]:.3f})")
        else:
            # print(f"[Step {step:3d}] final step, no propagation.")
            pass

        # ── 2. APPLY ──────────────────────────────────────────────────────────
        with_pending = new_trajectory[0]

        if with_pending is not None:
            trigger_step, x_new, u_new, rho, rho_series, replan_buffer = with_pending
            expected = trigger_step + replan_buffer

            if step < expected:
                if user_decision[0] == 'waiting':
                    fraction = (expected - step) / replan_buffer
                    timer_bar_patch.set_width(fraction)

            elif step == expected:
                decision = user_decision[0]

                if decision == 'accepted':
                    steps_remaining = _N_total[0] - step - 1
                    cols = min(u_new.shape[1], steps_remaining)
                    new_u_block = np.zeros((3, steps_remaining))
                    new_u_block[:, :cols] = u_new[:, :cols]
                    current_u[:, step+1:] = new_u_block
                    plan_end_step[0] = step + cols  # beyond this step: hover (velocity zeroed)

                    if not spec_switch_accepted[0] and spec_switch_announced[0]:
                        spec_switch_accepted[0] = True
                        points = np.array([x_new[0, :], x_new[1, :]]).T.reshape(-1, 1, 2)
                        segments = np.concatenate([points[:-1], points[1:]], axis=1)
                        norm = plt.Normalize(vmin=0, vmax=max(rho_series) if max(rho_series) > 0 else 1)
                        lc = LineCollection(segments, cmap='RdYlGn', norm=norm, linewidth=2, zorder=101)
                        lc.set_array(np.array(rho_series[:len(segments)]))
                        ax.add_collection(lc)

                    replan_log.append((trigger_step, step, True))
                    if spec_switch_announced[0]:
                        pending_voice_goal[0] = None
                    # print(f"[Step {step:3d}] Replan APPLIED (trig={trigger_step})")
                else:
                    reason = 'timeout' if decision == 'waiting' else 'rejected'
                    if reason == 'rejected':
                        # user explicitly said no — revert to original spec
                        spec_current[0] = spec_str
                        spec_switch_announced[0] = False
                        pending_voice_goal[0] = None
                    elif reason == 'timeout':
                        # user didn't respond in time — keep pursuing the voice goal
                        # (do NOT reset spec_current: it may be built for extended N_total,
                        #  resetting to spec_str would produce degenerate shifted specs)
                        spec_switch_announced[0] = False
                        pending_voice_goal[0] = None
                    replan_log.append((trigger_step, step, False))
                    print(f"[Step {step:3d}] Replan {reason} (trig={trigger_step}), keeping current.")

                candidate_line.set_data([], [])
                branch_marker.set_data([], [])
                prompt_text_obj.set_text('')
                timer_bar_patch.set_width(0)
                prompt_ax.set_visible(False)
                user_decision[0] = None
                new_trajectory[0] = None

            else:
                candidate_line.set_data([], [])
                branch_marker.set_data([], [])
                prompt_text_obj.set_text('')
                timer_bar_patch.set_width(0)
                prompt_ax.set_visible(False)
                user_decision[0] = None
                new_trajectory[0] = None
                print(f"[Step {step:3d}] Replan expired (trig={trigger_step}), discarding.")

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
                and not replan_running[0]
                and user_decision[0] != 'waiting'): 

            # choose buffer based on whether this replan needs interaction.
            # pending_voice_goal covers the case where the new goal matches the
            # original spec_str (e.g. switching back), which would otherwise
            # give is_spec_switched=False and skip the confirmation dialog.
            is_spec_switched = (spec_current[0] != spec_str) or (pending_voice_goal[0] is not None)
            needs_interact   = is_spec_switched and not spec_switch_announced[0]
            apply_buffer     = REPLAN_BUFFER_INTERACT if needs_interact else REPLAN_BUFFER_AUTO

            # Interact buffer may not fit near end of trajectory; fall back to auto buffer
            # so the attempt still fires and the extension logic can add more steps if needed.
            if needs_interact and step + apply_buffer > _N_total[0] - 2:
                apply_buffer = REPLAN_BUFFER_AUTO

            if step + apply_buffer <= _N_total[0] - 2 and not replan_running[0]:
                replan_running[0] = True
                if user_decision[0] != 'waiting':
                    candidate_line.set_data([], [])
                    branch_marker.set_data([], [])
                    prompt_text_obj.set_text('')
                    timer_bar_patch.set_width(0)
                    prompt_ax.set_visible(False)
                    user_decision[0] = None

                apply_step  = step + apply_buffer
                x_predicted = x_true.copy()
                for i in range(apply_buffer):
                    idx = step + 1 + i
                    if idx < _N_total[0]:
                        x_predicted = A_tilde @ x_predicted + B_tilde @ current_u[:, idx]

                x0_replan = x_predicted.copy()
                # print(f"[Step {step:3d}] Replanning triggered (buffer={apply_buffer}). "
                #       f"Predicted apply pos: ({x0_replan[0]:.3f}, {x0_replan[1]:.3f})")

                task_queue.put((
                    step, apply_step, x0_replan,
                    spec_current[0], scenario_objects, _N_total[0], dt, max_acc, max_speed,
                    is_spec_switched, spec_switch_announced[0], apply_buffer, map_bounds
                ))

        # ── 5. VISUALISE ──────────────────────────────────────────────────────
        orig_N = all_u.shape[1]
        if step < orig_N - 1:
            # print(f"[Vis step={step}] real_line points={len(position_history)}, "
            #     f"real_end=({position_history[-1][0]:.3f},{position_history[-1][1]:.3f})")
            pass

        history_arr = np.array(position_history)
        real_line.set_data(history_arr[:, 0], history_arr[:, 1])
        real_dot.set_data([x_true[0]], [x_true[1]])

        if show_no_mpc and len(nompc_history) > 1:
            na = np.array(nompc_history)
            nompc_line.set_data(na[:, 0], na[:, 1])

        # Update countdown: seconds remaining before voice instruction deadline
        deadline_step = _N_total[0] - 2 - REPLAN_BUFFER_INTERACT
        secs_left = max(0.0, (deadline_step - step) * dt)
        display_secs = max(0.0, secs_left - 2.0)
        if secs_left > 0:
            countdown_text.set_text(f"New instruction window open — {display_secs:.1f}s left")
            countdown_text.set_visible(True)
        else:
            countdown_text.set_text("New instruction window closed")
            countdown_text.set_color('gray')

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(dt)

        step += 1

    # ── Done ──────────────────────────────────────────────────────────────────
    # print(f"\n{'='*60}\nSimulation complete.\nReplan log: {replan_log}\n{'='*60}\n")

    if voice_listener is not None:
        voice_listener.close()

    task_queue.put(None)
    worker.join(timeout=3.0)
    if worker.is_alive():
        worker.terminate()

    history_arr = np.array(position_history)
    real_line.set_data(history_arr[:, 0], history_arr[:, 1])
    if show_no_mpc:
        na = np.array(nompc_history)
        nompc_line.set_data(na[:, 0], na[:, 1])
    fig.canvas.draw()
    fig.canvas.flush_events()

    input("Simulation finished. Press Enter to close.")
    plt.close('all')
    return position_history, replan_log