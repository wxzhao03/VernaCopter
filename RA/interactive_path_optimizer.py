"""
interactive_path_optimizer.py

Interactive module for iteratively improving drone trajectories based on user feedback.
"""

from STL.STL_to_path import STLSolver
from RA.NL_to_Optimization import NL_to_Optimization
from basics.logger import color_text
from visuals.visualization import Visualizer
import matplotlib.pyplot as plt
import numpy as np
from LLM.voice_openai import get_user_input_voice, listen_for_yes_no_voice, set_input_mode, get_input_mode, VoiceOpenAI

def integrate_interactive_optimizer(
    scenario,
    spec,
    x0,
    T,
    dt,
    pars,
    initial_x,
    initial_u,
    initial_rho,
    initial_rho_series,
    max_iterations=10
):
    """
    Main integration function for interactive path optimization.
    
    Parameters:
        scenario: Scenario object
        spec: STL specification string
        x0: Initial state
        T: Time horizon
        dt: Time step
        pars: System parameters object
        initial_x: Initial trajectory
        initial_u: Initial control inputs
        initial_rho: Initial global robustness
        initial_rho_series: Initial robustness time series
        
    Returns:
        tuple: (x_final, u_final, rho_final, rho_series_final)
    """
    # Initialize translator
    nl_optimizer = NL_to_Optimization(
        scenario=scenario,
        current_trajectory=initial_x,
        current_rho_series=initial_rho_series,
        GPT_model=pars.GPT_model
    )
    
    # Current state
    current_x = initial_x
    current_u = initial_u
    current_rho = initial_rho
    current_rho_series = initial_rho_series

    iteration = 0
    
    print("\n" + "="*100)
    print(color_text("INTERACTIVE PATH OPTIMIZATION", 'green'))
    print("="*100)

    # Ask user to choose input mode once at the beginning
    print(color_text("Choose input mode for this session:", 'yellow'))
    print("  [V] Voice input")
    print("  [T] Type input")
    choice = input(color_text("Choose (V/T, default=V): ", 'orange')).strip().lower()
    
    if choice == 't':
        set_input_mode('text')
        print(color_text("✓ Text input mode selected\n", 'green'))
    else:
        set_input_mode('voice')
        print(color_text("✓ Voice input mode selected\n", 'green'))

    # Show available obstacles
    obstacle_names = [name for name in scenario.objects.keys() 
                     if 'obs' in name.lower() or 'obstacle' in name.lower()]
    if obstacle_names:
        print(f"\n  Available obstacles: {', '.join([f'{i+1}:{name}' for i, name in enumerate(obstacle_names)])}")
    print()
    
    voice_system = VoiceOpenAI()
    voice_system.speak(f"If you wish to make adjustments to the path, please inform us of your requirements. What would you like to do?")
    voice_system.close()

    while True:
        # Step 1: Wait for user command
        user_input = get_user_input_voice()
        
        # Step 2: Translate command using GPT
        command = nl_optimizer.parse_adjustment_command(user_input)
        if command is None:
            print(color_text("Failed to parse command. Please try again.\n", 'red'))
            continue
        
        action = command['action']
        params = command['parameters']
        
        # Step 3: Check if finish command
        if action == 'finish_adjustment':
            voice_system = VoiceOpenAI()
            voice_system.speak(f"Optimization complete. Final robustness is {current_rho:.2f}. Deploying to drone.")
            voice_system.close()
            print(color_text("\nOptimization complete!", 'green'))
            return current_x, current_u, current_rho, current_rho_series
        
        # Step 4: Execute adjustment based on type

        # Type 1: Global safety adjustment
        if action == 'adjust_global_safety':
            direction = params['direction']
            
            print(color_text(f"\nExecuting: Global {direction}", 'green'))
            
            # Calculate new rho_min based on direction
            if direction == 'increase':
                new_rho_min = current_rho + 0.05  # Increase safety
            else:  # decrease
                new_rho_min = max(0.0, current_rho - 0.05)  # Decrease safety, but not below 0
            try:
                # Generate new trajectory with adjusted rho_min
                print(color_text("\n→ Generating new trajectory...", 'yellow'))
                solver = STLSolver(spec, scenario.objects, x0, T)
                
                new_x, new_u, new_rho, new_rho_series, runtime = solver.generate_trajectory(
                    dt=dt,
                    max_acc=pars.max_acc,
                    max_speed=pars.max_speed,
                    verbose=False,
                    include_dynamics=True,
                    rho_min=new_rho_min,
                    obstacle_adjustments=None 
                )
                
                # Check if solution was found
                if new_x is None or np.isnan(new_x).any():
                    print(color_text("✗ Failed to find feasible trajectory with new constraint.", 'red'))
                    print("  Try a different adjustment.\n")
                    continue
                
                print(color_text(f"✓ New trajectory generated in {runtime:.2f}s!", 'green'))
                
                # Visualize current vs proposed trajectory
                visualize_comparison(scenario, current_x, current_rho_series, 
                                    new_x, new_rho_series)
                
                # User accepts or rejects
                if accept_trajectory(current_rho, new_rho):
                    print(color_text("\n✓ New trajectory accepted!", 'green'))
                    current_x = new_x
                    current_u = new_u
                    current_rho = new_rho
                    current_rho_series = new_rho_series
                    
                    # Prompt user for next action based on input mode
                    if get_input_mode() == 'voice':
                        voice = VoiceOpenAI()
                        voice.speak(f"Trajectory accepted. Do you have any other adjustment requests?or finish the adjustment?")
                        voice.close()
                    else:
                        print(color_text(f"\n📢 Trajectory accepted.", 'cyan'))
                        print(color_text("Do you have any other adjustment requests?or finish the adjustment", 'cyan'))
                else:
                    print(color_text("\n✗ Trajectory rejected. Keeping current trajectory.", 'yellow'))
                    
                    # Prompt user for next action based on input mode
                    if get_input_mode() == 'voice':
                        voice = VoiceOpenAI()
                        voice.speak(f"Trajectory rejected. Do you have other requirement?or finish the adjustment")
                        voice.close()
                    else:
                        print(color_text(f"\n📢 Trajectory rejected.", 'cyan'))
                        print(color_text("Do you have other requirement?or finish the adjustment", 'cyan'))

                iteration += 1
                
            except Exception as e:
                print(color_text(f"✗ Error: {e}\n", 'red'))
                continue
        
        # Type 2: Local safety adjustment 
        elif action == 'adjust_local_safety':
            obs_idx = params['obstacle_index']
            direction = params['direction']
            
            obs_name = obstacle_names[obs_idx-1] # Get obstacle name from index
            
            weights_to_try = [1]
            success = False
            
            for attempt, weight in enumerate(weights_to_try, 1):
                print(f"\n  Attempt {attempt}/{len(weights_to_try)}: weight = {weight}")
                
                # Set up obstacle-specific adjustment parameters
                obstacle_adjustments = {
                    obs_name: (direction, weight)
                }
                
                try:
                    print(color_text("  → Generating trajectory...", 'yellow'))
                    solver = STLSolver(spec, scenario.objects, x0, T)
                    
                    new_x, new_u, new_rho, new_rho_series, runtime = solver.generate_trajectory(
                        dt=dt,
                        max_acc=pars.max_acc,
                        max_speed=pars.max_speed,
                        verbose=False,
                        rho_min=current_rho,
                        obstacle_adjustments=obstacle_adjustments
                    )
                    
                    if new_x is None or np.isnan(new_x).any():
                        print(color_text(f"  ✗ Infeasible, trying smaller weight...", 'yellow'))
                        continue
                    
                    print(color_text(f"  ✓ Trajectory generated in {runtime:.2f}s", 'green'))
                    
                    visualize_comparison(scenario, current_x, current_rho_series, new_x, new_rho_series)
                    # User decision on proposed trajectory
                    if accept_trajectory(current_rho, new_rho):
                        current_x = new_x
                        current_u = new_u
                        current_rho = new_rho
                        current_rho_series = new_rho_series
                        print(color_text("\n✓ Trajectory accepted!", 'green'))
                        
                        if get_input_mode() == 'voice':
                            voice = VoiceOpenAI()
                            voice.speak(f"Accepted. Robustness is {current_rho:.2f}.")
                            voice.close()
                    else:
                        print(color_text("\n✗ Trajectory rejected.", 'yellow'))
                    
                    success = True
                    break
                    
                except Exception as e:
                    print(color_text(f"  ✗ Error: {str(e)[:80]}...", 'yellow'))
                    continue
            
            if not success:
                print(color_text("\n✗ All attempts failed.", 'red'))
                print("  This adjustment may be too constraining given current constraints.")
                print("  Suggestions:")
                print("  - Try a different obstacle")
                print("  - Try the opposite direction (closer↔farther)")
                print("  - First increase global safety, then adjust obstacles\n")
                
                if get_input_mode() == 'voice':
                    voice = VoiceOpenAI()
                    voice.speak("Could not find a feasible trajectory. Please try a different adjustment.")
                    voice.close()
            
            iteration += 1


def accept_trajectory(current_rho, new_rho):
    """
    Ask user to accept or reject the proposed trajectory
    """
    prompt = f"Do you accept the new path?"
    return listen_for_yes_no_voice(prompt)


def visualize_comparison(scenario, x_current, rho_current, x_proposed, rho_proposed):
    """
    Display side-by-side comparison of current and proposed trajectories.
    """
    plt.close('all')
    
    visualizer = Visualizer(x_current, scenario)
    
    fig, ax = visualizer.visualize_comparison_2d(
        rho_original=rho_current,
        x_new=x_proposed,
        rho_series_new=rho_proposed,
        label_original="Current Trajectory",
        label_new="Proposed Trajectory"
    )
    
    plt.show(block=False)
    mng = plt.get_current_fig_manager()
    # Attempt to maximize window based on backend
    try:
        if hasattr(mng, 'window') and hasattr(mng.window, 'state'):
            mng.window.state('zoomed')
            # print("✓ Window maximized (Tk)")
        elif hasattr(mng, 'window') and hasattr(mng.window, 'showMaximized'):
            mng.window.showMaximized()
            # print("✓ Window maximized (Qt)")
    except Exception as e:
        print(f"⚠ Could not maximize: {e}")

    plt.show()
    
    plt.pause(0.5)
    
    print(color_text("\n→ Comparison visualization displayed", 'cyan'))