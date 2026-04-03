import random
import numpy as np
from pprint import pprint


import nasim
from torch.utils.tensorboard import SummaryWriter

class TabularQfuntion:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.q_func = dict()

    def __call__(self):
        return self.forward(x)
    
    def forward(self, state):
        key = str(state.astype(int)) if isinstance(state, np.ndarray) else state
        if key not in self.q_func:
            self.q_func[key] = np.zeros(self.num_actions, dtype = np.float32)

        return self.q_func[key]
    
    def forward_batch(self, state_batch):
        return np.asarray([self.forward(state) for state in state_batch])
    
    def update(self, s, a, q_val):
        q_state = self.forward(s)
        q_state[a] += q_val
    
    def get_action(self, s):
        return int(self.forward(s).argmax())
    
    def display(self):
        pprint(self.q_func)


