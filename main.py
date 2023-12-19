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
from utils import evaluate

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')

parser.add_argument('--K', type=int, default=4)
parser.add_argument('--embed_dim', type=int, default=128)
parser.add_argument('--n_layer', type=int, default=3)
parser.add_argument('--n_head', type=int, default=1)
parser.add_argument('--activation_function', type=str, default='relu')
parser.add_argument('--dropout', type=float, default=0.1)
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
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
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

evfn = evaluate(args)

#Tesnorboard
writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size, args.seed,args.max_ep_len,env)




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
            # action = agent.select_action(states,rewards,timesteps)  # Sample action from policy
            action = agent.select_action(states,rtg,timesteps)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
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

        counter_t += 1

        state = next_state

        if total_numsteps % 2000 == 0 and args.eval is True:
            evfn.ev(agent,total_numsteps,writer,goal,args)
        # states = next_state
    memory.push(trajectory)
    

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))


env.close()


