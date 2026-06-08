"""
one_shot_automatic.py

Runs the system in "one-shot" mode using predefined parameters and task descriptions.
Supports two execution paths after trajectory generation:
  - Simulation : use_simulation=True  → simulation_two_mode.simulate_deployment
  - Real drone : deploy_on_drone=True → deploy_online.deploy
    For pre-flight interactive path adjustment (no real-time interruption during flight),
    use deploy_and_visualizition_2d (instead of deploy_online2 ) and set interactive_optimization_enabled=True in config.

NOTE: spec, newspec, switch_step, use_voice are currently hardcoded for debugging.
      In future, spec and newspec will be generated at runtime from voice input.
"""

from basics.config import One_shot_parameters
from main import main
from experiments.save_results import save_results
import numpy as np

try:
    from deployment.deploy_online2 import deploy
    deployment = True
except Exception as e:
    deployment = False
    print("Motion capturing not installed, no real deployment possible")
    print(e)


def run_one_shot(scenario_name="reach_avoid"):
    import matplotlib.pyplot as plt
    plt.close('all')

    pars = One_shot_parameters(scenario_name=scenario_name)

    if pars.multi_drone:
        from multidrone.multidrone_planner import run_multidrone_online
        run_multidrone_online()
        return
    # ─────────────────────────────────────────────────────────────────────────

    try:
        messages, task_accomplished, waypoints, all_rho_np, all_u, spec = main(pars)
        # print("waypoints after main:", waypoints.shape)
        # print("all_u after main:", all_u.shape)
    except Exception as e:
        print(e)


        task_accomplished = False
        messages          = []
        all_rho_np        = None
        waypoints         = None
        all_u             = None
        spec              = None

    if pars.save_results:
        save_results(pars, messages, task_accomplished, waypoints)

    from basics.scenarios import Scenarios
    scenario = Scenarios(pars.scenario_name)

    # TODO: these will come from runtime voice input in the future
    newspec     = scenario.automated_translator_newspec or None  
    switch_step = pars.switch_step
    use_voice   = pars.use_voice

    # ── Simulation path ───────────────────────────────────────────────────────
    if pars.use_simulation and (waypoints is not None):
        from deployment.simulation_two_mode import simulate_deployment

        simulate_deployment(
            waypoints, all_u,
            scenario=scenario,
            all_rho=all_rho_np,
            noise_std=pars.noise_std,
            spec_str=spec,
            spec_str_phase2=None,   # voice-driven: no predefined newspec
            switch_step=None,       # voice-driven: no predefined switch step
            dt=pars.dt,
            max_acc=pars.max_acc,
            max_speed=pars.max_speed,
            use_voice=use_voice,
        )

    # ── Real drone path ───────────────────────────────────────────────────────
    elif pars.deploy_on_drone and deployment and (waypoints is not None):
        import matplotlib.pyplot as plt
        plt.close('all')

        deploy(
            waypoints, all_u,
            scenario=scenario,
            all_rho=all_rho_np,
            spec_str=spec,
            spec_str_phase2=newspec,
            switch_step=switch_step,
            dt=pars.dt,
            max_acc=pars.max_acc,
            max_speed=pars.max_speed,
            use_online_mpc=True,
            use_voice=use_voice,
            send_full_pose=False,
        )


if __name__ == "__main__":
    run_one_shot()