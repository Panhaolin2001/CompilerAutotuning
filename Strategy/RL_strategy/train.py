import torch
from .expolor import EpsilonGreedy
from .agents.DQN import DQNAgent
from .model.GCN import GCN
from .model.GRNN import GraphRNN
from .model.MLP import MLP
from .model.TGCN import TGCN
from .model.Transformer import Transformer
from .utility.replay_buffer import ReplayBuffer
from ..common import Actions


class TrainManager():

    def __init__(self,env,episodes=1000,lr=0.001,gamma=0.9,
                 e_greed=0.1,e_greed_decay=1e-6,memory_size=2000,replay_start_size=400,batch_size=32,num_steps=4,
                 update_target_steps=200):

        self.env = env
        self.episodes = episodes
        n_obs = env.feature_dim
        rb = ReplayBuffer(memory_size, num_steps, self.env.obs_model)

        if self.env.obs_model == "GCN":
            print("------------------------------")
            print("             GCN              ")
            print("------------------------------")
            q_func = GCN(n_obs, env.n_act)

        elif self.env.obs_model == "MLP":
            print("------------------------------")
            print("             MLP              ")
            print("------------------------------")
            q_func = MLP(n_obs, env.n_act)
            
        elif self.env.obs_model == "Transformer":
            print("------------------------------")
            print("         Transformer          ")
            print("------------------------------")
            q_func = Transformer(n_obs, env.n_act)
        
        elif self.env.obs_model == "T-GCN":
            print("------------------------------")
            print("            T-GCN             ")
            print("------------------------------")
            q_func = TGCN(n_obs, env.n_act)

        elif self.env.obs_model == "GRNN":
            print("------------------------------")
            print("             GRNN             ")
            print("------------------------------")
            q_func = GraphRNN(n_obs, env.n_act)
        
        optimizer = torch.optim.AdamW(q_func.parameters(), lr=lr)
        
        explorer = EpsilonGreedy(env.n_act,e_greed,e_greed_decay)
        self.agent = DQNAgent(
            q_func=q_func,
            optimizer=optimizer,
            replay_buffer = rb,
            batch_size=batch_size,
            replay_start_size = replay_start_size,
            n_act=env.n_act,
            update_target_steps=update_target_steps,
            explorer=explorer,
            gamma=gamma,
            obs_model=self.env.obs_model
        )
    
    def train(self):
        for e in range(self.episodes):
            ep_reward = self.train_episode()
            print("Episode %s: reward=%.3f"%(e,ep_reward))

            if e % 200 == 0 and e >= 1:
                test_reward = self.test_episode()
                print("Test reward = %.3f" % (test_reward))
        
        test_reward = self.test_episode()
        print("Test reward = %.3f" % (test_reward))

    def train_episode(self):
        total_reward = 0
        obs = self.env.reset()
        while True:
            action_idx = self.agent.act(obs)
            action = list(Actions)[action_idx]
            # print(f"Action : {action}")
            reward,next_obs,done = self.env.step(action)

            if self.env.obs_model == "MLP":
                next_obs = tuple(element.item() for element in next_obs)
                obs = tuple(element.item() for element in obs)

            self.agent.learn(obs,action_idx,reward,next_obs,done)

            if self.env.obs_model == "MLP":
                next_obs = torch.FloatTensor(next_obs)
                obs = torch.FloatTensor(obs)

            obs = next_obs
            total_reward += reward
            if done:
                break
        return total_reward

    def test_episode(self):
        total_reward = 0
        obs = self.env.reset()
        while True:
            action_idx = self.agent.predict(obs)
            action = list(Actions)[action_idx]
            print(f"Action : {action}")
            reward,next_obs,done = self.env.step(action)
            obs = next_obs
            total_reward += reward

            if done:
                break
        return total_reward
