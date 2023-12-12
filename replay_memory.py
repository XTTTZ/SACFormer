import random
import numpy as np

class ReplayMemory:
    def __init__(self, capacity, seed, max_ep_len,env):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.max_ep_len = max_ep_len
        self.act_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape[0]

    def push(self,trajectory):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (trajectory)
        self.position = (self.position + 1) % self.capacity


    
    def discount_cumsum(self,x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0]-1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
        return discount_cumsum

    def sample(self, batch_size, K):
        # Sample a batch of trajectories
        trajectories = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, timesteps = [], [], [], [], [], []
        
        rtgs = []

        # Process each trajectory to only keep K items
        processed_trajectories = []
        for trajectory in trajectories:
            if len(trajectory) >= K:
                rtg_reward = []
                # If trajectory is longer than K, randomly sample K items from it
                # sampled_trajectory = random.sample(trajectory, K)
                indices = random.sample(range(len(trajectory)), K)
                indices.sort()
                idx_start = indices[0]
                for state, action, reward, next_state, done in trajectory[idx_start:]:
                    rtg_reward.append(reward)
                rtg_reward = np.array(rtg_reward)
                rtg = self.discount_cumsum(rtg_reward, 0.99)
                for item in indices: 
                    rtgs.append(rtg[item-idx_start])
                # indices = random.sample(range(len(trajectory)), K)
                # indices = np.sort(indices)
                sampled_trajectory = [trajectory[i] for i in indices]
                for state, action, reward, next_state, done in sampled_trajectory:
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    next_states.append(next_state)
                    dones.append(done)
                indices = [min(idx, self.max_ep_len-1) for idx in indices]
                timesteps.append(indices)
            else:
                rtg_reward = []
                # If trajectory is shorter than or equal to K, use it as is
                while True:
                    new_trajectory = random.sample(self.buffer, 1)
                    new_trajectory = new_trajectory[0]
                    # print("new_trajectory", len(new_trajectory[0]))
                    if len(new_trajectory) >= K:

                        indices = random.sample(range(len(new_trajectory)), K)
                        indices.sort()
                        idx_start = indices[0]
                        for state, action, reward, next_state, done in new_trajectory[idx_start:]:
                            rtg_reward.append(reward)
                        rtg_reward = np.array(rtg_reward)
                        rtg = self.discount_cumsum(rtg_reward, 0.99)
                        for item in indices: 
                            rtgs.append(rtg[item-idx_start])
                        # indices = random.sample(range(len(new_trajectory)), K)
                        # indices = np.sort(indices)
                        sampled_trajectory = [new_trajectory[i] for i in indices]
                        for state, action, reward, next_state, done in sampled_trajectory:
                            states.append(state)
                            actions.append(action)
                            rewards.append(reward)
                            next_states.append(next_state)
                            dones.append(done)
                        indices = [min(idx, self.max_ep_len-1) for idx in indices]
                        timesteps.append(indices)
                        break
                    else:
                        pass
        states = np.array(states)
        states = states.reshape(batch_size,K,-1)
        actions = np.array(actions)
        actions = actions.reshape(batch_size,K,-1)
        rewards = np.array(rewards)
        rewards = rewards.reshape(batch_size,K,-1)
        rtgs = np.array(rtgs)
        rtgs = rtgs.reshape(batch_size,K,-1)
        next_states = np.array(next_states)
        next_states = next_states.reshape(batch_size,K,-1)
        dones = np.array(dones)
        dones = dones.reshape(batch_size,K,-1)
        timesteps = np.array(timesteps)
        timesteps = timesteps.reshape(batch_size,K,-1)
        timesteps = timesteps.squeeze()
        rewards = rewards.squeeze()
        rtgs = rtgs.squeeze()
        # print("states", states.shape)



        return states, actions, rewards, next_states, dones, timesteps, rtgs
    
   


    def __len__(self):
        return len(self.buffer)
    