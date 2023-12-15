import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
# from model import GaussianPolicy, QNetwork, DeterministicPolicy
from model import Qformer, Pformer,Qformerwithr,PformerD
import numpy as np


class SAC(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.K = args.K

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.critic = Qformer(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        # self.critic = Qformer(
        #     state_dim=num_inputs,
        #     act_dim=action_space.shape[0],
        #     max_length=args.K,
        #     max_ep_len=args.max_ep_len,
        #     hidden_size=args.embed_dim,
        #     n_layer=args.n_layer,
        #     n_head=args.n_head,
        #     n_inner=4*args.embed_dim,
        #     activation_function=args.activation_function,
        #     n_positions=1024,
        #     resid_pdrop=args.dropout,
        #     attn_pdrop=args.dropout,
        # ).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        # self.critic_optim = torch.optim.AdamW(
        # self.critic.parameters(),
        # lr=args.lr,
        # weight_decay=args.weight_decay,
        # )
        # self.critic_scheduler = torch.optim.lr_scheduler.LambdaLR(
        # self.critic_optim,
        # lambda steps: min((steps+1)/args.warmup_steps, 1)
        # )

        # self.critic_target = Qformer(
        #     state_dim=num_inputs,
        #     act_dim=action_space.shape[0],
        #     max_length=args.K,
        #     max_ep_len=args.max_ep_len,
        #     hidden_size=args.embed_dim,
        #     n_layer=args.n_layer,
        #     n_head=args.n_head,
        #     n_inner=4*args.embed_dim,
        #     activation_function=args.activation_function,
        #     n_positions=1024,
        #     resid_pdrop=args.dropout,
        #     attn_pdrop=args.dropout,
        #     action_space=action_space,
        # ).to(self.device)
        self.critic_target = Qformer(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = Pformer(
                state_dim=num_inputs,
                act_dim=action_space.shape[0],
                max_length=args.K,
                max_ep_len=args.max_ep_len,
                hidden_size=args.embed_dim,
                n_layer=args.n_layer,
                n_head=args.n_head,
                n_inner=4*args.embed_dim,
                activation_function=args.activation_function,
                n_positions=1024,
                resid_pdrop=args.dropout,
                attn_pdrop=args.dropout,
                action_space=action_space
            ).to(self.device)

            # self.policy = Pformer(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            # self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy = PformerD(
                state_dim=num_inputs,
                act_dim=action_space.shape[0],
                max_length=args.K,
                max_ep_len=args.max_ep_len,
                hidden_size=args.embed_dim,
                n_layer=args.n_layer,
                n_head=args.n_head,
                n_inner=4*args.embed_dim,
                activation_function=args.activation_function,
                n_positions=1024,
                resid_pdrop=args.dropout,
                attn_pdrop=args.dropout,
                action_space=action_space
            ).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state,reward,timesteps, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        reward = torch.FloatTensor(reward).to(self.device)
        timesteps = torch.LongTensor(timesteps).to(self.device)
        if evaluate is False:
            action, _, _ = self.policy.sample(state,reward,timesteps)
            # action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state,reward,timesteps)
            # _, _, action = self.policy.sample(state)
        # print("action shape", action[0,-1].shape)
        return action[0,-1].detach().cpu().numpy()

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch, timesteps, rtgs = memory.sample(batch_size=batch_size, K=self.K)
        # s, a, r, d, rtg, timesteps, mask
        # state_batch, action_batch, reward_batch, next_state_batch, mask_batch, timesteps = memory(batch_size=batch_size,K=self.K)
        # state_batch, action_batch, reward_batch, next_state_batch, mask_batch, timesteps = memory.get_batch(batch_size=batch_size, max_len=self.K)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).squeeze()
        rtgs = torch.FloatTensor(rtgs).to(self.device)
        rtgs = rtgs/1000

        # print("state_batch shape", state_batch.shape)
        # print("timesteps shape", timesteps.shape)
        timesteps = torch.LongTensor(timesteps).to(self.device)

        # next_timesteps = np.tile(np.arange(self.K), (batch_size, 1))
        # next_timesteps = torch.LongTensor(next_timesteps).to(self.device)
        # next_timesteps = timesteps
        next_timesteps = np.array([np.sort(np.random.choice(range(1000), self.K, replace=True)) for _ in range(batch_size)])

        # next_timesteps = np.tile(np.arange(self.K), (batch_size, 1))
        next_timesteps = torch.LongTensor(next_timesteps).to(self.device)
        next_timesteps = timesteps+1
        next_rtgs = torch.clone(rtgs)
        next_rtgs = next_rtgs - 0.003

        with torch.no_grad():


            # next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch,next_reward,next_timesteps)
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch,next_rtgs,next_timesteps)


            # qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action,next_timesteps)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            # qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_rtgs, next_state_action, next_timesteps)
            # print("qf1_next_target shape", qf1_next_target.shape)
            # print("qf2_next_target shape", qf2_next_target.shape)

            qf1_next_target = qf1_next_target.squeeze()
            qf2_next_target = qf2_next_target.squeeze()
            next_state_log_pi = next_state_log_pi.squeeze()
            # print("qf1_next_target shape", qf1_next_target.shape)
            # print("qf2_next_target shape", qf2_next_target.shape)
            # print("next_state_log_pi shape", next_state_log_pi.shape)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            # next_q_value = reward_last + mask_last * self.gamma * (min_qf_next_target)
            # print("next_q_value shape", reward_last.shape,mask_last.shape,next_state_log_pi.shape) 
        # qf1, qf2 = self.critic(state_batch, action_batch,timesteps)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1, qf2 = self.critic(state_batch, action_batch)
        # qf1, qf2 = self.critic(state_batch, rtgs, action_batch, timesteps)
        qf1 = qf1.squeeze()
        qf2 = qf2.squeeze()
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), .25) #############
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch,rtgs,timesteps)
        # pi, log_pi, _ = self.policy.sample(state_batch)

        # state_last = state_batch[:,-1,:].squeeze()

        # pi = torch.cat((action_batch[:,1:,:],pi.unsqueeze(1)),dim=1)

        # qf1_pi, qf2_pi = self.critic(state_batch, pi,timesteps)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        # qf1_pi, qf2_pi = self.critic(state_batch, rtgs, pi, timesteps)
        qf1_pi = qf1_pi.squeeze()
        qf2_pi = qf2_pi.squeeze()
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), .25) #############
        self.policy_optim.step()
        # self.policy_scheduler.step()
        # self.critic_scheduler.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()

