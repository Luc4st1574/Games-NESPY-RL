# Import necessary libraries
from enviroment import game_settings
from games import game_selection
from wrappers import apply_wrappers
from agent import Agent


#state = 4 frames stacked together
#cnn convolutional neural network

# 5 = Na OF THE OUTPUT

epoch = 50000


def main():
    # Select the game
    game, movement_set, na = game_selection()
    
    #print("Game selected: ", game)
    #print("Movement set: ", movement_set)
    #print("Neural network neurons: ", na)
    
    #nn = neural network neurons need for each game
    if game is not None:
        # Define the game settings
        env = game_settings(game, movement_set)
        
        # Play the game
        #play_game(env, movement_set)
        
        env = apply_wrappers(env)
        
        # Create an agent
        
        agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)
        
        for i in range(epoch):
            done = False
            state, _ = env.reset()
            while not done:
                a = agent.choose_action(state)
                new_state, reward, done, truncated, info  = env.step(a)
                
                agent.store_in_memory(state, a, reward, new_state, done)
                agent.learn()

                state = new_state
        env.close()
    else:
        print("Game setup failed. Please select a valid game.")

if __name__ == "__main__":
    main()