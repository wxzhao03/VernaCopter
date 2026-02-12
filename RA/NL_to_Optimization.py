
"""
Natural language interface for trajectory optimization adjustments.
Translates user commands into structured optimization parameters using GPT.
"""

import json
from LLM.GPT import GPT
from basics.logger import color_text
import os
from LLM.voice_openai import listen_for_response_voice


class NL_to_Optimization:

    def __init__(self, scenario, current_trajectory, current_rho_series, GPT_model="gpt-3.5-turbo"):

        self.scenario = scenario
        self.current_x = current_trajectory
        self.current_rho = current_rho_series
        self.gpt = GPT(GPT_model)
        
        # Load available obstacles
        self.obstacles = {name: bounds for name, bounds in scenario.objects.items() 
                         if 'obs' in name.lower() or 'obstacle' in name.lower()}
    
    def load_instructions(self, filename='adjustment_instructions.txt'):
        """Load GPT instructions from file."""
        current_path = os.path.dirname(os.path.abspath(__file__))
        parent_path = os.path.dirname(current_path)
        folder = 'LLM/instructions'
        path = os.path.join(parent_path, folder, filename)
        
        with open(path, 'r', encoding='utf-8') as f:
            instructions = f.read()
        
        return instructions
    
    def parse_adjustment_command(self, user_input, max_retries=2):
        instructions = self.load_instructions()
        # Initialize conversation with system instructions
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_input}
        ]
        
        for attempt in range(max_retries):
            try:
                response = self.gpt.chatcompletion(messages)
                # print(color_text(f"GPT Response: {response}", 'yellow'))
                
                # Check if GPT is asking for clarification
                response_stripped = response.strip()
                if not response_stripped.startswith('{'):
                    print(color_text(f"GPT: {response}", 'cyan'))
                    # Get user's clarification
                    clarification = listen_for_response_voice(response)
                    # Continue conversation
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": clarification})
                    continue
                
                # Extract JSON command
                command = self.extract_command(response)
                
                # Validate command
                if self.validate_command(command):
                    return command
                else:
                    print(color_text("Invalid command format. Retrying...", 'red'))
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "system", 
                                   "content": "Invalid format. Return only a valid JSON object."})
                    
            except ValueError as e:
                print(color_text(f"Error: {e}", 'red'))
                if attempt < max_retries - 1:
                    messages.append({"role": "system", 
                                   "content": f"Error: {e}. Please return valid JSON only."})
                else:
                    return None
            
            except Exception as e:
                print(color_text(f"Unexpected error: {e}", 'red'))
                return None
        
        return None
    
    def extract_command(self, response):
        response = response.strip()
        
        # Remove markdown formatting if present
        response = response.replace("```json", "").replace("```", "").strip()
        
        # Remove angle brackets if present
        if response.startswith("<") and response.endswith(">"):
            print(color_text("Warning: Removing unexpected < > brackets", 'yellow'))
            response = response[1:-1].strip()
        
        # Parse JSON
        try:
            command = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}\nResponse: {response[:200]}")
        
        return command
    
    #Validate command structure and parameter value
    def validate_command(self, command):
        if not isinstance(command, dict):
            return False
        
        if 'action' not in command or 'parameters' not in command:
            return False
        
        action = command['action']
        params = command['parameters']
        
        # Type 1: Global safety
        if action == 'adjust_global_safety':
            if 'direction' not in params:
                return False
            if params['direction'] not in ['increase', 'decrease']:
                return False
            return True
        
        # Type 2: Local safety
        elif action == 'adjust_local_safety':
            if 'obstacle_index' not in params or 'direction' not in params:
                return False
            if params['direction'] not in ['increase', 'decrease']:
                return False
            # Check if obstacle index is valid
            obstacle_count = len(self.obstacles)
            obs_idx = params['obstacle_index']
            if not isinstance(obs_idx, int) or obs_idx < 0 or obs_idx >= obstacle_count:
                return False
            return True
        
        # Type 3: Finish adjustment
        elif action == 'finish_adjustment':
            return True
        
        else:
            return False