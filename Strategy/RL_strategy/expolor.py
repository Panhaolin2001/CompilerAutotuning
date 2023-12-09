import numpy as np

class EpsilonGreedy():

    def __init__(self,n_act,e_greed,decay_rate):
        self.n_act = n_act
        self.epsilon = e_greed
        self.decay = decay_rate
    
    def act(self,predict_method,obs):
        if np.random.uniform(0,1) < self.epsilon:
            action = np.random.choice(self.n_act)
        else:
            action = predict_method(obs)
        
        self.epsilon = max(0.01, self.epsilon - self.decay)
        return action