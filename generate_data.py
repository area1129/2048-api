from game2048.game import Game
from game2048.agents import ExpectiMaxAgent
import numpy as np
trys = 2
num_try = 2000
dataset = np.zeros((trys*num_try,17),dtype = np.int)

for n in range(trys):
    print("n = ", n)
    game = Game(size=4,score_to_win=8192)
    agent = ExpectiMaxAgent(game=game)
    cnt = 0
    while (not agent.game.end) and cnt<2000:
        direction = agent.step()
        for i in range(4):
            for j in range(4):
                dataset[2000*n+cnt,4*i+j]=game.board[i, j]
        dataset[2000*n+cnt,16]=direction
        game.move(direction)
        cnt = cnt+1

np.save('./my_data/X0.npy',dataset[:,:16])
np.save('./my_data/Y0.npy',dataset[:,16])