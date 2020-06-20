# 2048-api
A 2048 game api for training supervised learning (imitation learning) or reinforcement learning agents

# How to run the code
1. Put the your data into the folder "data" (both X.npy and Y.npy) 
2. or run generate_data.py to generate data.
3. Run cnn5.py.(My code about the .npy operation is at the bottom)

# Code structure
* [`game2048/`](game2048/): the main package.
    * [`game.py`](game2048/game.py): the core 2048 `Game` class.
    * [`agents.py`](game2048/agents.py): the `Agent` class with instances.
    * [`displays.py`](game2048/displays.py): the `Display` class with instances, to show the `Game` state.
    * [`expectimax/`](game2048/expectimax): a powerful ExpectiMax agent by [here](https://github.com/nneonneo/2048-ai).
* [`explore.ipynb`](explore.ipynb): introduce how to use the `Agent`, `Display` and `Game`.
* [`static/`](static/): frontend assets (based on Vue.js) for web app.
* [`webapp.py`](webapp.py): run the web app (backend) demo.
* [`evaluate.py`](evaluate.py): evaluate your self-defined agent.
* [`my_agent.py`](my_agent.py): used in evaluate.py to use the model we had.
* [`cnn5.py`](cnn5.py): run this .py to train the model and save the model to the filepath assigned in the py.
* [`generate_data.py`](generate_data.py): generate data.

# Requirements
* code only tested on linux system (ubuntu 16.04)
* Python 3 (Anaconda 3.6.3 specifically) with numpy and flask

# To define your own agents
```python
from game2048.agents import Agent

class YourOwnAgent(Agent):

    def step(self):
        '''To define the agent's 1-step behavior given the `game`.
        You can find more instance in [`agents.py`](game2048/agents.py).
        
        :return direction: 0: left, 1: down, 2: right, 3: up
        '''
        direction = some_function(self.game)
        return direction

```

# To compile the pre-defined ExpectiMax agent

```bash
cd game2048/expectimax
bash configure
make
```

# To run the web app
```bash
python webapp.py
```
![demo](preview2048.gif)

# LICENSE
The code is under Apache-2.0 License.

# For EE369 / EE228 students from SJTU
Please read course project [requirements](EE369.md) and [description](https://docs.qq.com/slide/DS05hVGVFY1BuRVp5). 
