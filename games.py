import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT as mario_movement
import gym_zelda_1
from gym_zelda_1.actions import MOVEMENT as zelda_movement
import gym_tetris
from gym_tetris.actions import MOVEMENT as tetris_movement

def game_selection():
    game = input("Select a game to play: \n 1. Super Mario Bros \n 2. The Legend of Zelda \n 3. Tetris \n")
    if game == "1":
        game = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True, render_mode="human")
        movement_set = mario_movement
        na = 7
        game_name = "SuperMarioBros"
    elif game == "2":
        game = gym_zelda_1.make('Zelda1-v0', apply_api_compatibility=True, render_mode="human")
        movement_set = zelda_movement
        na = 20
        game_name = "Zelda"
    elif game == "3":
        game = gym_tetris.make('TetrisA-v0', apply_api_compatibility=True, render_mode="human")
        movement_set = tetris_movement
        na = 12
        game_name = "Tetris"
    else:
        print("Invalid game selected")
        game = None
        movement_set = None
        na = None
        game_name = None
    return game, movement_set, na, game_name