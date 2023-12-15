import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
import pickle
import random

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')

'''
Log: change the default value of K from 10 to 4
    change n_layer from 3 to 2
    change optimizer from AdamW to Adam
    change N_in from 4* to 2*

'''


parser.add_argument('--K', type=int, default=4)


parser.add_argument('--embed_dim', type=int, default=128)
parser.add_argument('--n_layer', type=int, default=3)
parser.add_argument('--n_head', type=int, default=1)
parser.add_argument('--activation_function', type=str, default='relu')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
parser.add_argument('--warmup_steps', type=int, default=10000)
parser.add_argument('--num_eval_episodes', type=int, default=100)
parser.add_argument('--max_iters', type=int, default=10)
parser.add_argument('--num_steps_per_iter', type=int, default=10000)
parser.add_argument('--device', type=str, default='cuda')


parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=10000010, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--max_ep_len', type=int, default=1100)
parser.add_argument('--goal', type=float, default=4)
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)
env.seed(args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)


#Tesnorboard
writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size, args.seed,args.max_ep_len,env)
'''
############################################################################################################
dataset_path = f'data/{env_name}-{dataset}-v2.pkl'
with open(dataset_path, 'rb') as f:
    trajectories = pickle.load(f)

# save all path information into separate lists
# mode = variant.get('mode', 'normal')
env_name = args.env_name
dataset = 'medium'
mode = 'normal'
states, traj_lens, returns = [], [], []
for path in trajectories:
    if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
        path['rewards'][-1] = path['rewards'].sum()
        path['rewards'][:-1] = 0.
    states.append(path['observations'])
    traj_lens.append(len(path['observations']))
    returns.append(path['rewards'].sum())
traj_lens, returns = np.array(traj_lens), np.array(returns)

# used for input normalization
states = np.concatenate(states, axis=0)
state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

num_timesteps = sum(traj_lens)

print('=' * 50)
print(f'Starting new experiment: {env_name} {dataset}')
print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
print('=' * 50)

K = args.K
batch_size = args.batch_size
# num_eval_episodes = variant['num_eval_episodes']
pct_traj = 1.

# only train on top pct_traj trajectories (for %BC experiment)
num_timesteps = max(int(pct_traj*num_timesteps), 1)
sorted_inds = np.argsort(returns)  # lowest to highest
num_trajectories = 1
timesteps = traj_lens[sorted_inds[-1]]
ind = len(trajectories) - 2
while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
    timesteps += traj_lens[sorted_inds[ind]]
    num_trajectories += 1
    ind -= 1
sorted_inds = sorted_inds[-num_trajectories:]

# used to reweight sampling so we sample according to timesteps instead of trajectories
p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])
state_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
max_ep_len = args.max_ep_len
scale = 1000

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

def get_batch(batch_size=256, max_len=K):
    batch_inds = np.random.choice(
        np.arange(num_trajectories),
        size=batch_size,
        replace=True,
        p=p_sample,  # reweights so we sample according to timesteps
    )

    s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
    for i in range(batch_size):
        traj = trajectories[int(sorted_inds[batch_inds[i]])]
        si = random.randint(0, traj['rewards'].shape[0] - 1)

        # get sequences from dataset
        s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
        a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
        r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
        if 'terminals' in traj:
            d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
        else:
            d.append(traj['dones'][si:si + max_len].reshape(1, -1))
        timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
        timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
        rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
        if rtg[-1].shape[1] <= s[-1].shape[1]:
            rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

        # padding and state + reward normalization
        tlen = s[-1].shape[1]
        s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
        s[-1] = (s[-1] - state_mean) / state_std
        a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
        r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
        d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
        rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
        timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
        mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

    s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
    a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
    r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
    d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
    rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
    timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
    mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

    return s, a, r, d, rtg, timesteps, mask

############################################################################################################

'''


# Training Loop
total_numsteps = 0
updates = 0
device = torch.device("cuda" if args.cuda else "cpu")
goal = args.goal
for i_episode in itertools.count(1):
    counter_t = 0
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    trajectory = []
    states = np.array(state).reshape(1, env.observation_space.shape[0])
    timesteps = np.array(0, dtype=np.int64).reshape(1, 1)
    rewards = np.array(0, dtype=np.float32).reshape(1, 1)
    rtg = np.array(goal, dtype=np.float32).reshape(1, 1)


    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            # print(states.shape)
            if states.shape[0] > 512:
                states = states[1:,:]
                timesteps = timesteps[:,1:]
                rewards = rewards[:,1:]
                rtg = rtg[:,1:]
            # states = torch.from_numpy(state).reshape(1, env.observation_space.shape[0]).to(device=device, dtype=torch.float32)
            # timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
            # action = agent.select_action(states,rewards,timesteps)  # Sample action from policy
            action = agent.select_action(states,rtg,timesteps)  # Sample action from policy
            # action = agent.select_action(state,timesteps)
            # action.squeeze()
            
            # print("in")

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
            # if i_episode % 2:
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        next_state, reward, done, _ = env.step(action) # Step

        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        ones_array = np.ones((1, 1), dtype=np.int64) * (counter_t + 1)
        timesteps = np.concatenate([timesteps, ones_array], axis=1)
        cur_state = np.array(next_state).reshape(1, env.observation_space.shape[0])
        states = np.concatenate([states, cur_state], axis=0)
        zero_array = np.zeros((1, 1), dtype=np.float32)
        rewards[0,-1] = reward
        rewards = np.concatenate([rewards, zero_array], axis=1)
        new = rtg[0,-1] - reward/1000
        new = np.array(new, dtype=np.float32).reshape(1, 1)
        rtg = np.concatenate([rtg, new], axis=1)

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        # print(action.shape)
        trajectory.append([state, action, reward, next_state, mask])

        # memory.push(state, action, reward, next_state, mask) # Append transition to memory
        # print(timesteps)
        counter_t += 1

        state = next_state
        # states = next_state
    memory.push(trajectory)
    

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
    # if i_episode % 10 == 0 and args.eval is True and i_episode > args.batch_size:
    if i_episode % 10 == 0 and args.eval is True:
        avg_reward = 0.
        episodes = 10
        for _  in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            counter = 0
            states = np.array(state).reshape(1, env.observation_space.shape[0])
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
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward

                cur_state = np.array(next_state).reshape(1, env.observation_space.shape[0])
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


        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")
    if i_episode % 100 == 0:
        agent.save_checkpoint(args.env_name,suffix="episode"+str(i_episode))

env.close()


