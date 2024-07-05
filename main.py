# Import necessary libraries
from enviroment import game_settings, initialize_agent, train_agent
from games import game_selection
from wrappers import apply_wrappers
from agent import Agent
import os
#state = 4 frames stacked together
#cnn convolutional neural network

EPOCH = 50000
TRAIN = True
CKPT = 5000
folder_name = ""
ckpt_name = "" 

def main():
    
    model_path = os.path.join("models")
    os.makedirs(model_path, exist_ok=True)
    
    # Select the game
    game, movement_set, na = game_selection()
    
    #nn = neural network neurons need for each game
    if game is not None:
        # Define the game settings
        env = game_settings(game, movement_set)
        
        # Play the game
        #play_game(env, movement_set)
        
        env = apply_wrappers(env)
        
        # Create an agent
        
        agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)
        
        initialize_agent(agent, TRAIN, model_path, folder_name, ckpt_name)
        train_agent(agent, env, EPOCH, model_path, TRAIN, CKPT)
        
    else:
        print("Game setup failed. Please select a valid game.")

if __name__ == "__main__":
    # Run the main function
    main()