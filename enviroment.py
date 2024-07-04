# Import necessary libraries
from nes_py.wrappers import JoypadSpace


def game_settings(env, movement_set):
    # Ensure the environment is wrapped correctly
    
    env = JoypadSpace(env, movement_set)
    
    return env

def play_game(env, movement_set):
    # Create a flag - restart or not
    done = True
    
    # Loop through each frame in the game
    for step in range(100000):
        # Start the game to begin with
        if done:
            env.reset()
            
            
        action = env.action_space.sample()
        
        # Do random actions
        observation, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        
        # Show the game on the screen
        env.render()
    
    # Close the game
    env.close()






