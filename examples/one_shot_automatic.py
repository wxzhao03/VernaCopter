# -*- coding: utf-8 -*-
"""

one_shot_automatic.py

This script runs the system automatically in a "one-shot" mode without interactive user input. 
It uses predefined parameters and task descriptions for the selected scenario to generate and execute a trajectory.

The results, including system messages, task success, and waypoints, can be saved if specified in the parameters.

This mode is particularly useful for testing and benchmarking system performance.
"""

from basics.config import One_shot_parameters               # Import the one-shot parameters
from main import main                                       # Import the main function
from experiments.save_results import save_results           # Import the save_results function
import numpy as np

try:
    from deployment.deploy_online import deploy
    deployment = True
except Exception as e:
    deployment = False
    print("Motion capturing not installed, no real deployment possible")
    print(e)

def run_one_shot(scenario_name="reach_avoid"): # treasure_hunt, reach_avoid

    pars = One_shot_parameters(scenario_name = scenario_name)   # Get the parameters

    try:
        if pars.use_simulation:
            messages, task_accomplished, waypoints, all_rho_np, all_u, spec = main(pars)
            print("waypoints after main:", waypoints.shape)
            print("all_u after main",all_u.shape)
        else:
            messages, task_accomplished, waypoints, all_rho_np, all_u, spec = main(pars)
        # messages, task_accomplished, waypoints, all_rho_np = main(pars)     # Run the main program
        # messages, task_accomplished, waypoints, all_rho_np, spec = main(pars)   #for online
    except Exception as e:
        print(e)
        
        task_accomplished = False
        messages = []
        all_rho_np = None
        waypoints = None  
        all_u = None       
        spec = None        

    if pars.save_results:
        save_results(pars, messages, task_accomplished, waypoints) # Save the results

    # TODO: ask user to load old feasible trajectory
    # waypoints = None # remove this, once we add a trajectory checker
    # if pars.deploy_on_drone and deployment and (waypoints is not None):
    #     from basics.scenarios import Scenarios
    #     scenario = Scenarios(pars.scenario_name)
    #     # deploy(waypoints, 
    #     #    scenario=scenario,       
    #     #    all_rho=all_rho_np)
    #     print("waypoints shape:", waypoints.shape)
    #     print("waypoints first 3 cols vel:", waypoints[3:6, :3])
    #     # deploy(waypoints, scenario=scenario,       
    #     #    all_rho=all_rho_np, spec_str=spec,dt=pars.dt, use_online_mpc=False)
    # #     deploy(waypoints, scenario=scenario, all_rho=all_rho_np,
    # #    spec_str=spec, dt=pars.dt, tracking_test=True)
    #     deploy(waypoints, scenario=scenario,       
    #         all_rho=all_rho_np, spec_str=spec, dt=pars.dt,
    #         max_acc=pars.max_acc, max_speed=pars.max_speed,
    #         use_online_mpc=True)
        

    if pars.use_simulation and (waypoints is not None):
        from basics.scenarios import Scenarios
        scenario = Scenarios(pars.scenario_name)
        from deployment.simulation_two_mode import simulate_deployment
        if pars.automated_translator:
            newspec = scenario.automated_translator_newspec
        # print("all_u shape:", all_u.shape)


        # for simulate_mpc_online
        # simulate_deployment(waypoints, all_u, scenario=scenario,
                        # all_rho=all_rho_np, noise_std=pars.noise_std,
                        # spec_str=spec, dt=pars.dt,
                        # max_acc=pars.max_acc, max_speed=pars.max_speed,use_voice=True)
       #use_voice=True
 
    #  for simulate_online_change
    #     simulate_deployment(
    #         waypoints, all_u,
    #         scenario=scenario,
    #         all_rho=all_rho_np,
    #         spec_str=spec,              
    #         spec_str_phase2=(
    #         'STL_formulas.inside_cuboid(objects["goal"], name="goal").always(35, 35) & '
    #         '(STL_formulas.outside_cuboid(objects["obstacle1"], name="!obstacle1") & '
    #         'STL_formulas.outside_cuboid(objects["obstacle2"], name="!obstacle2") & '
    #         'STL_formulas.outside_cuboid(objects["obstacle4"], name="!obstacle4") & '
    #         'STL_formulas.outside_cuboid(objects["obstacle3"], name="!obstacle3") & '
    #         'STL_formulas.outside_cuboid(objects["obstacle5"], name="!obstacle5") & '
    #         'STL_formulas.outside_cuboid(objects["obstacle6"], name="!obstacle6") & '
    #         'STL_formulas.outside_cuboid(objects["obstacle7"], name="!obstacle7")).always(0, 35)'
    #     ),
    #         switch_step=15,
    #         noise_std=pars.noise_std,
    #         dt=pars.dt,
    #         max_acc=pars.max_acc,
    #         max_speed=pars.max_speed,
    #     )

    # # same spec
    #     simulate_deployment(waypoints, all_u, scenario=scenario,all_rho=all_rho_np, noise_std=pars.noise_std,spec_str=spec,dt=pars.dt,max_acc=pars.max_acc, max_speed=pars.max_speed,use_voice=False)
    # cswitch spec
        print("waypoint for simulation:", waypoints.shape)
        print("all_u for simulation", all_u.shape)
        simulate_deployment(waypoints, all_u, scenario=scenario,all_rho=all_rho_np, noise_std=pars.noise_std, spec_str=spec, 
                            spec_str_phase2=newspec, switch_step=5,dt=pars.dt,max_acc=pars.max_acc, max_speed=pars.max_speed,use_voice=True)
    elif pars.deploy_on_drone and deployment and (waypoints is not None):
        from basics.scenarios import Scenarios
        scenario = Scenarios(pars.scenario_name)
        newspec = scenario.automated_translator_newspec if pars.automated_translator else None
        deploy(waypoints, all_u, scenario=scenario, all_rho=all_rho_np,
            spec_str=spec,
            spec_str_phase2=newspec,
            switch_step=9,
            dt=pars.dt,
            max_acc=pars.max_acc, max_speed=pars.max_speed,
            use_online_mpc=True)
    return messages, task_accomplished, waypoints


if __name__ == "__main__":
    # Allow running standalone
    _, task_accomplished, _= run_one_shot()
    print("Task accomplished:", task_accomplished)