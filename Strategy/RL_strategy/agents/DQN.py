import torch
import copy
from torch.utils.data import DataLoader
from ..utility.torchUtils import one_hot, CustomDataset, GNN_collate_fn, Transformer_collate_fn, TGCN_collate_fn

class DQNAgent():

    def __init__(self,q_func,optimizer,replay_buffer,batch_size,replay_start_size,
                 n_act,update_target_steps,explorer,gamma=0.9,obs_model="MLP"):
        
        self.explorer = explorer

        self.pred_func = q_func
        self.target_func = copy.deepcopy(self.pred_func)
        self.update_target_steps = update_target_steps

        self.global_step = 0
        self.rb = replay_buffer
        self.obs_model = obs_model
        self.batch_size = batch_size
        self.replay_start_size = replay_start_size
        self.optimizer = optimizer
        self.criterion = torch.nn.MSELoss()
        self.n_act = n_act
        self.gamma = gamma

    def predict(self, obs):
        Q_list = self.pred_func(obs)
        action = int(torch.argmax(Q_list).detach().numpy())
        return action

    def act(self, obs):
        return self.explorer.act(self.predict,obs)
    
    def learn_batch(self, batch_obs, batch_action, batch_reward, batch_next_obs, batch_done):

        match self.obs_model:
            case "GCN":
                dataset = CustomDataset(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)
                data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=GNN_collate_fn)

                for batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_done in data_loader:
                    pred_VS = self.pred_func(batch_obs)
                    action_onehot = one_hot(batch_action, self.n_act)
                    predict_Q = (pred_VS * action_onehot).sum(dim=1)
                    target_Q = batch_reward + (1 - batch_done) * self.gamma * self.target_func(batch_next_obs).max(1)[0]

                    self.optimizer.zero_grad()
                    loss = self.criterion(predict_Q, target_Q)
                    loss.backward()
                    self.optimizer.step()

            case "MLP":
                pred_VS = self.pred_func(batch_obs)
                action_onehot = one_hot(batch_action, self.n_act)
                predict_Q = (pred_VS * action_onehot).sum(dim=1)
                target_Q = batch_reward + (1 - batch_done) * self.gamma * self.target_func(batch_next_obs).max(1)[0]

                self.optimizer.zero_grad()
                loss = self.criterion(predict_Q, target_Q)
                loss.backward()
                self.optimizer.step()

            case "Transformer":
                dataset = CustomDataset(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)
                data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=Transformer_collate_fn)

                for batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_done in data_loader:
                    pred_VS = self.pred_func(batch_obs)
                    action_onehot = one_hot(batch_action, self.n_act)
                    predict_Q = (pred_VS * action_onehot).sum(dim=1)
                    target_Q = batch_reward + (1 - batch_done) * self.gamma * self.target_func(batch_next_obs).max(1)[0]

                    self.optimizer.zero_grad()
                    loss = self.criterion(predict_Q, target_Q)
                    loss.backward()
                    self.optimizer.step()

            case "T-GCN" | "GRNN" :
                dataset = CustomDataset(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)
                data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=TGCN_collate_fn)

                for batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_done in data_loader:
                    pred_VS = self.pred_func(batch_obs).squeeze(0)
                    action_onehot = one_hot(batch_action, self.n_act)
                    predict_Q = (pred_VS * action_onehot).sum(dim=1)
                    target_Q = batch_reward + (1 - batch_done) * self.gamma * self.target_func(batch_next_obs).max(2)[0].squeeze(0)

                    self.optimizer.zero_grad()
                    loss = self.criterion(predict_Q, target_Q)
                    loss.backward()
                    self.optimizer.step()

            case _:
                raise ValueError(f"Unknown obs model: {self.obs_model}, please choose 'GCN', 'MLP', 'Transformer', 'T-GCN', 'GRNN' ")

    def learn(self, obs, action, reward, next_obs, done):
        self.global_step+=1
        self.rb.append((obs, action, reward, next_obs, done))
        
        if len(self.rb) > self.replay_start_size and self.global_step % self.rb.num_steps == 0:
            self.learn_batch(*self.rb.sample(self.batch_size))
        
        if self.global_step % self.update_target_steps==0:
            self.sync_target()
    
    def sync_target(self):
        for target_param, param in zip(self.target_func.parameters(), self.pred_func.parameters()):
            target_param.data.copy_(param.data)