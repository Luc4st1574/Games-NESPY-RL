import os
import signal
import sys
from nes_py.wrappers import JoypadSpace

def initialize_agent(agent, SHOULD_TRAIN, LOAD_MODEL, model_folder, game_name):
    model_path = os.path.join(model_folder, f"{game_name}_model.pt")
    if LOAD_MODEL and os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        agent.load_model(model_path)
        if SHOULD_TRAIN:
            print("Continuing training with the loaded model")
        else:
            agent.epsilon = 0.05  # Set a low epsilon for exploitation
            agent.eps_min = 0.05
            agent.eps_decay = 1.0  # No decay
            print(f"Model loaded for inference. Epsilon set to {agent.epsilon}")
    else:
        print("Starting with a new model")

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

def save_model(agent, model_folder, episode_num, game_name):
    model_path = os.path.join(model_folder, f"{game_name}_model_{episode_num}_iter.pt")
    agent.save_model(model_path)
    print(f"Model saved to {model_path}")

def train_agent(agent, env, num_episodes, model_folder, SHOULD_TRAIN, CKPT_SAVE_INTERVAL, game_name):
    def save_and_exit(signal, frame):
        print("\nSignal received. Saving model...")
        save_model(agent, model_folder, "final", game_name)
        env.close()
        sys.exit(0)

    # Register the signal handlers
    signal.signal(signal.SIGINT, save_and_exit)
    signal.signal(signal.SIGTERM, save_and_exit)

    try:
        for i in range(num_episodes):
            print(f"Episode {i} - Game: {game_name}")
            total_reward = run_episode(env, agent, SHOULD_TRAIN)
            print(f"Total reward: {total_reward}, Epsilon: {agent.epsilon}, "
                    f"Size of replay buffer: {len(agent.replay_buffer)}, "
                    f"Learn step counter: {agent.learn_step_counter}")

            if SHOULD_TRAIN and (i + 1) % CKPT_SAVE_INTERVAL == 0:
                save_model(agent, model_folder, i + 1, game_name)

    except Exception as e:
        print(f"Exception occurred: {e}. Saving model...")
        save_model(agent, model_folder, "final", game_name)
        raise
    finally:
        env.close()

def game_settings(env, movement_set):
    return JoypadSpace(env, movement_set)

def play_game(env, movement_set):
    done = True
    for step in range(100000):
        if done:
            env.reset()
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
    env.close()