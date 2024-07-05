# Import necessary libraries
from nes_py.wrappers import JoypadSpace
import os

def initialize_agent(agent, SHOULD_TRAIN, model_path, folder_name="", ckpt_name=""):
    if not SHOULD_TRAIN:
        agent.load_model(os.path.join(model_path, folder_name, ckpt_name))
        agent.epsilon = 0.2
        agent.eps_min = 0.0
        agent.eps_decay = 0.0

def run_episode(env, agent, SHOULD_TRAIN):
    state, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.choose_action(state)
        new_state, reward, done, truncated, info = env.step(action)
        total_reward += reward

        if SHOULD_TRAIN:
            agent.store_in_memory(state, action, reward, new_state, done)
            agent.learn()

        state = new_state

    return total_reward

def save_model(agent, model_path, episode_num):
    agent.save_model(os.path.join(model_path, "model_" + str(episode_num) + "_iter.pt"))

def train_agent(agent, env, num_episodes, model_path, SHOULD_TRAIN, CKPT_SAVE_INTERVAL):
    for i in range(num_episodes):
        print("Episode:", i)
        total_reward = run_episode(env, agent, SHOULD_TRAIN)
        print("Total reward:", total_reward, "Epsilon:", agent.epsilon, 
            "Size of replay buffer:", len(agent.replay_buffer), 
            "Learn step counter:", agent.learn_step_counter)

        if SHOULD_TRAIN and (i + 1) % CKPT_SAVE_INTERVAL == 0:
            save_model(agent, model_path, i + 1)

    env.close()

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






