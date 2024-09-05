import os
from enviroment import game_settings, initialize_agent, train_agent
from games import game_selection
from wrappers import apply_wrappers
from agent import Agent

EPOCH = 50000
TRAIN = True  # Keep this True to continue training
LOAD_MODEL = True  # Set this to True to load a pre-trained model
CKPT = 5000

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    game, movement_set, na, game_name = game_selection()
    
    if game is not None:
        game_folder = os.path.join(current_dir, game_name)
        os.makedirs(game_folder, exist_ok=True)
        
        env = game_settings(game, movement_set)
        env = apply_wrappers(env)
        
        agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)
        
        initialize_agent(agent, TRAIN, LOAD_MODEL, game_folder, game_name)
        
        train_agent(agent, env, EPOCH, game_folder, TRAIN, CKPT, game_name)
    else:
        print("Game setup failed. Please select a valid game.")

if __name__ == "__main__":
    main()