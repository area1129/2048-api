from game2048.agents import Agent
from keras.models import load_model
import numpy as np

class myAgent(Agent):
    
    def __init__(self, game, display=None,filepath='model_best_cnn3.h5'):

        super().__init__(game, display)
        self.model = load_model(filepath)

    def step(self):
        transformed_board = self.transform_board(self.game.board)
        y = self.model.predict(transformed_board)
        maxposition = np.where(y==np.max(y,axis=1))
        return int(maxposition[1])

    def transform_board(self,board):
        transformed_board = np.zeros((1,4,4,16))
        for p in range(4):
            for q in range(4):
                if board[p,q] == 0:
                    transformed_board[0,p, q, 0] = 1
                else:
                    transformed_board[0,p, q, int(np.log2(board[p,q]))] = 1
        return transformed_board