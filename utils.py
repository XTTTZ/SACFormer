import math
import torch
import gym
import numpy as np

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class evaluate():
    def __init__(self,args):
        self.env = gym.make(args.env_name)
        self.env.seed(args.seed)
        self.env.action_space.seed(args.seed)


    def ev(self,agent,total_numsteps,writer,goal,args):
        # if i_episode % 10 == 0 and args.eval is True:
        avg_reward = 0.
        episodes = 4
        for _  in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            counter = 0
            states = np.array(state).reshape(1, self.env.observation_space.shape[0])
            timesteps = np.array(0, dtype=np.int64).reshape(1, 1)
            rewards = np.array(0, dtype=np.float32).reshape(1, 1)
            rtg = np.array(goal, dtype=np.float32).reshape(1, 1)

            while not done:
                # print(states.shape)
                if states.shape[0] > 512:
                    states = states[1:,:]
                    timesteps = timesteps[:,1:]
                    rewards = rewards[:,1:]
                    rtg = rtg[:,1:]
                # action = agent.select_action(states,rewards,timesteps, evaluate=True)
                action = agent.select_action(states,rtg,timesteps, evaluate=True)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward

                cur_state = np.array(next_state).reshape(1, self.env.observation_space.shape[0])
                states = np.concatenate([states, cur_state], axis=0)
                ones_array = np.ones((1, 1), dtype=np.int64) * (counter + 1)
                zero_array = np.zeros((1, 1), dtype=np.float32)
                timesteps = np.concatenate([timesteps, ones_array], axis=1)
                rewards[0,-1] = reward
                rewards = np.concatenate([rewards, zero_array], axis=1)
                new = rtg[0,-1] - reward/1000
                new = np.array(new, dtype=np.float32).reshape(1, 1)
                rtg = np.concatenate([rtg, new], axis=1)
                counter += 1
                # print(timesteps)

                # states = next_state
            avg_reward += episode_reward
        avg_reward /= episodes


        writer.add_scalar('avg_reward/test', avg_reward, total_numsteps)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")
