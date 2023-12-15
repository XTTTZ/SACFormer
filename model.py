import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np
import torch
import torch.nn as nn

import transformers

from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.trajectory_gpt2 import GPT2Model

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class Qformer(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(Qformer, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 2)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2
    
# class Qformer(TrajectoryModel):

#     """
#     This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
#     """

#     def __init__(
#             self,
#             state_dim,
#             act_dim,
#             hidden_size,
#             max_length=None,
#             max_ep_len=4096,
#             action_tanh=True,
#             **kwargs
#     ):
#         super().__init__(state_dim, act_dim, max_length=max_length)

#         self.hidden_size = hidden_size
#         config = transformers.GPT2Config(
#             vocab_size=1,  # doesn't matter -- we don't use the vocab
#             n_embd=hidden_size,
#             **kwargs
#         )

#         # note: the only difference between this GPT2Model and the default Huggingface version
#         # is that the positional embeddings are removed (since we'll add those ourselves)
#         self.transformer = GPT2Model(config)
#         self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
#         self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
#         self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
#         self.embed_ln = nn.LayerNorm(hidden_size)
#         self.predict_q = torch.nn.Linear(hidden_size, 1)
#         # self.predict_q = nn.Sequential(
#         #     *([nn.Linear(2*hidden_size, hidden_size)] + ([nn.Linear(hidden_size, 1)]))
#         # )
#         # self.predict_q = nn.Sequential(
#         #     *([nn.Linear(2*hidden_size, 1)] + ([nn.Tanh()] if action_tanh else []))
#         # )
#         self.rtg_emded = torch.nn.Linear(1, hidden_size)
#         self.rtg_emded_1 = torch.nn.Linear(1, hidden_size)

#         self.transformer_1 = GPT2Model(config)
#         self.embed_timestep_1 = nn.Embedding(max_ep_len, hidden_size)
#         self.embed_state_1 = torch.nn.Linear(self.state_dim, hidden_size)
#         self.embed_action_1 = torch.nn.Linear(self.act_dim, hidden_size)
#         self.embed_ln_1 = nn.LayerNorm(hidden_size)
#         self.predict_q_1 = torch.nn.Linear(hidden_size, 1)
#         # self.predict_q_1 = nn.Sequential(
#         #     *([nn.Linear(2*hidden_size, hidden_size)] + ([nn.Linear(hidden_size, 1)]))
#         # )
#         # self.predict_q_1 = nn.Sequential(
#         #     *([nn.Linear(2*hidden_size, 1)] + ([nn.Tanh()] if action_tanh else []))
#         # )
#         self.apply(weights_init_)

#     def forward(self, states, actions, rtg, timesteps, attention_mask=None):

#         batch_size, seq_length = states.shape[0], states.shape[1]

#         rtg = rtg.reshape(batch_size, seq_length, 1)

#         if attention_mask is None:
#             # attention mask for GPT: 1 if can be attended to, 0 if not
#             attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
#             attention_mask = attention_mask.to(states.device)

#         # embed each modality with a different head
#         state_embeddings = self.embed_state(states)
#         action_embeddings = self.embed_action(actions)
#         time_embeddings = self.embed_timestep(timesteps)
#         state_embeddings = state_embeddings + time_embeddings
#         action_embeddings = action_embeddings + time_embeddings

#         state_embeddings_1 = self.embed_state_1(states)
#         action_embeddings_1 = self.embed_action_1(actions)
#         time_embeddings_1 = self.embed_timestep_1(timesteps)
#         state_embeddings_1 = state_embeddings_1 + time_embeddings_1
#         action_embeddings_1 = action_embeddings_1 + time_embeddings_1

#         rtg_emded = self.rtg_emded(rtg)
#         rtg_emded_1 = self.rtg_emded_1(rtg)
#         rtg_emded = rtg_emded + time_embeddings
#         rtg_emded_1 = rtg_emded_1 + time_embeddings_1

#         # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
#         # which works nice in an autoregressive sense since states predict actions
#         stacked_inputs = torch.stack(
#             (rtg_emded,state_embeddings, action_embeddings), dim=1
#         ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
#         stacked_inputs = self.embed_ln(stacked_inputs)
#         stacked_inputs_1 = torch.stack(
#             (rtg_emded_1, state_embeddings_1, action_embeddings_1), dim=1
#         ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
#         stacked_inputs_1 = self.embed_ln_1(stacked_inputs_1)

#         # to make the attention mask fit the stacked inputs, have to stack it as well
#         stacked_attention_mask = torch.stack(
#             (attention_mask, attention_mask, attention_mask), dim=1
#         ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

#         # we feed in the input embeddings (not word indices as in NLP) to the model
#         transformer_outputs = self.transformer(
#             inputs_embeds=stacked_inputs,
#             attention_mask=stacked_attention_mask,
#         )

#         transformer_outputs_1 = self.transformer_1(
#             inputs_embeds=stacked_inputs_1,
#             attention_mask=stacked_attention_mask,
#         )
#         x = transformer_outputs['last_hidden_state']
#         x_1 = transformer_outputs_1['last_hidden_state']

#         # reshape x so that the second dimension corresponds to the original
#         # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t

#         #######################

#         x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
#         x_1 = x_1.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
#         # x = x.mean(dim=2)
#         # x_1 = x_1.mean(dim=2)
#         # x = x.reshape(batch_size, 2*self.hidden_size)
#         # x_1 = x_1.reshape(batch_size,2*self.hidden_size)

#         ########################

#         # x = torch.sum(x, dim=1)
#         # x_1 = torch.sum(x_1, dim=1)
#         # print(x.shape)


#         # x = x.reshape(batch_size, seq_length, 3*self.hidden_size)
#         # x_1 = x_1.reshape(batch_size, seq_length, 3*self.hidden_size)



#         q_1 = self.predict_q(x[:,2])  # predict next return given state and action
#         q_2 = self.predict_q_1(x_1[:,2])  # predict next return given state and action
#         # q_1 = self.predict_q(x)  # predict next return given state and action
#         # q_2 = self.predict_q_1(x_1)  # predict next return given state and action
#         q_1 = q_1.unsqueeze(2)
#         q_2 = q_2.unsqueeze(2)
#         # print(q_1)

#         return q_1, q_2
    

class Qformerwithr(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)
        self.predict_q = torch.nn.Linear(hidden_size, 1)

        self.transformer_1 = GPT2Model(config)
        self.embed_timestep_1 = nn.Embedding(max_ep_len, hidden_size)
        self.embed_state_1 = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action_1 = torch.nn.Linear(self.act_dim, hidden_size)
        self.embed_return_1 = torch.nn.Linear(1, hidden_size)
        self.embed_ln_1 = nn.LayerNorm(hidden_size)
        self.predict_q_1 = torch.nn.Linear(hidden_size, 1)
        self.apply(weights_init_)

    def forward(self, states, rtg, actions, timesteps, attention_mask=None):
        rtg = rtg.unsqueeze(2)

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
            attention_mask = attention_mask.to(states.device)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        time_embeddings = self.embed_timestep(timesteps)
        return_embeddings = self.embed_return(rtg)
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        return_embeddings = return_embeddings + time_embeddings

        state_embeddings_1 = self.embed_state_1(states)
        action_embeddings_1 = self.embed_action_1(actions)
        time_embeddings_1 = self.embed_timestep_1(timesteps)
        return_embeddings_1 = self.embed_return(rtg)
        state_embeddings_1 = state_embeddings_1 + time_embeddings_1
        action_embeddings_1 = action_embeddings_1 + time_embeddings_1
        return_embeddings_1 = return_embeddings_1 + time_embeddings_1

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (return_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)
        stacked_inputs_1 = torch.stack(
            (return_embeddings_1, state_embeddings_1, action_embeddings_1), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs_1 = self.embed_ln_1(stacked_inputs_1)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            output_hidden_states=True
        )

        transformer_outputs_1 = self.transformer_1(
            inputs_embeds=stacked_inputs_1,
            attention_mask=stacked_attention_mask,
            output_hidden_states=True
        )
        # x = transformer_outputs['last_hidden_state']
        # x_1 = transformer_outputs_1['last_hidden_state']
        x = transformer_outputs.hidden_states[2]
        x_1 = transformer_outputs_1.hidden_states[2]

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        # x = x.reshape(batch_size, seq_length, 2, self.hidden_size).permute(0, 2, 1, 3)
        # x_1 = x_1.reshape(batch_size, seq_length, 2, self.hidden_size).permute(0, 2, 1, 3)
        # x = torch.sum(x, dim=1)
        # x_1 = torch.sum(x_1, dim=1)
        # print(x.shape)
        # x = x.reshape(batch_size, seq_length, 3*self.hidden_size)
        # x_1 = x_1.reshape(batch_size, seq_length, 3*self.hidden_size)
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        x_1 = x_1.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        q_1 = self.predict_q(x[:,1])  # predict next return given state and action
        q_2 = self.predict_q_1(x_1[:,1])  # predict next return given state and action
        # q_1 = self.predict_q(x)  # predict next return given state and action
        # q_2 = self.predict_q_1(x_1)  # predict next return given state and action
        q_1 = q_1.unsqueeze(2)
        q_2 = q_2.unsqueeze(2)

        return q_1, q_2
    

class Pformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            action_space=None,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

        self.transformer = GPT2Model(config)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_reward = torch.nn.Linear(1, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)
        self.mean_linear = nn.Linear(hidden_size, act_dim)
        self.log_std_linear = nn.Linear(hidden_size, act_dim)
        # hid = int(hidden_size/2)
        # self.mean_linear = nn.Sequential(
        #     *([nn.Linear(hidden_size, hid)] + ([nn.Linear(hid, act_dim)]))
        # )
        # self.log_std_linear = nn.Sequential(
        #     *([nn.Linear(hidden_size, hid)] + ([nn.Linear(hid, act_dim)]))
        # )
        # self.apply(weights_init_)


    def forward(self, states,reward, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]
        reward = reward.reshape(batch_size, seq_length, 1)



        if seq_length == 1:
            timesteps = timesteps.reshape(batch_size, seq_length)

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
            attention_mask = attention_mask.to(states.device)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        time_embeddings = self.embed_timestep(timesteps)



        reward_embeddings = self.embed_reward(reward)
        reward_embeddings = reward_embeddings + time_embeddings



        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        # state_embeddings = self.embed_ln(state_embeddings)

        # print(seq_length)

        stacked_inputs = torch.stack(
            (reward_embeddings, state_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 2*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 2*seq_length)


        # we feed in the input embeddings (not word indices as in NLP) to the model
        # transformer_outputs = self.transformer(
        #     inputs_embeds=state_embeddings,
        #     attention_mask=attention_mask,
        # )
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            output_hidden_states=True
        )
        x = transformer_outputs['last_hidden_state']
        # x = transformer_outputs.hidden_states[2]
        

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        # x = x.reshape(batch_size, seq_length, 2*self.hidden_size)
        x = x.reshape(batch_size, seq_length, 2, self.hidden_size).permute(0, 2, 1, 3)
        # x = x.mean(dim=1) ###################
        x = x[:,0]
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std
    def sample(self, state,reward,timesteps):
        # print(state.shape)
        mean, log_std = self.forward(state,reward,timesteps)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(2, keepdim=False)
        # log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        # print(action.shape)
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(Pformer, self).to(device)
    

# class Pformer(TrajectoryModel):

#     """
#     This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
#     """

#     def __init__(
#             self,
#             state_dim,
#             act_dim,
#             hidden_size,
#             max_length=None,
#             max_ep_len=4096,
#             action_tanh=True,
#             action_space=None,
#             **kwargs
#     ):
#         super().__init__(state_dim, act_dim, max_length=max_length)

#         self.hidden_size = hidden_size
#         config = transformers.GPT2Config(
#             vocab_size=1,  # doesn't matter -- we don't use the vocab
#             n_embd=hidden_size,
#             **kwargs
#         )
#         if action_space is None:
#             self.action_scale = torch.tensor(1.)
#             self.action_bias = torch.tensor(0.)
#         else:
#             self.action_scale = torch.FloatTensor(
#                 (action_space.high - action_space.low) / 2.)
#             self.action_bias = torch.FloatTensor(
#                 (action_space.high + action_space.low) / 2.)

#         self.transformer = GPT2Model(config)
#         self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
#         self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
#         self.embed_reward = torch.nn.Linear(1, hidden_size)
#         self.embed_ln = nn.LayerNorm(hidden_size)
#         self.mean_linear = nn.Linear(hidden_size, act_dim)
#         self.log_std_linear = nn.Linear(hidden_size, act_dim)
#         self.apply(weights_init_)


#     def forward(self, states,reward, timesteps, attention_mask=None):

#         batch_size, seq_length = states.shape[0], states.shape[1]
#         reward = reward.reshape(batch_size, seq_length, 1)



#         if seq_length == 1:
#             timesteps = timesteps.reshape(batch_size, seq_length)

#         if attention_mask is None:
#             # attention mask for GPT: 1 if can be attended to, 0 if not
#             attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
#             attention_mask = attention_mask.to(states.device)

#         # embed each modality with a different head
#         state_embeddings = self.embed_state(states)
#         time_embeddings = self.embed_timestep(timesteps)



#         reward_embeddings = self.embed_reward(reward)
#         reward_embeddings = reward_embeddings + time_embeddings



#         # time embeddings are treated similar to positional embeddings
#         state_embeddings = state_embeddings + time_embeddings
#         # state_embeddings = self.embed_ln(state_embeddings)

#         # print(seq_length)

#         stacked_inputs = torch.stack(
#             (reward_embeddings, state_embeddings), dim=1
#         ).permute(0, 2, 1, 3).reshape(batch_size, 2*seq_length, self.hidden_size)
#         stacked_inputs = self.embed_ln(stacked_inputs)
#         stacked_attention_mask = torch.stack(
#             (attention_mask, attention_mask), dim=1
#         ).permute(0, 2, 1).reshape(batch_size, 2*seq_length)


#         # we feed in the input embeddings (not word indices as in NLP) to the model
#         # transformer_outputs = self.transformer(
#         #     inputs_embeds=state_embeddings,
#         #     attention_mask=attention_mask,
#         # )
#         transformer_outputs = self.transformer(
#             inputs_embeds=stacked_inputs,
#             attention_mask=stacked_attention_mask,
#             output_hidden_states=True
#         )
#         # x = transformer_outputs['last_hidden_state']
#         x = transformer_outputs.hidden_states[2]
        

#         # reshape x so that the second dimension corresponds to the original
#         # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
#         # x = x.reshape(batch_size, seq_length, self.hidden_size)
#         x = x.reshape(batch_size, seq_length, 2, self.hidden_size).permute(0, 2, 1, 3)
#         # x = x.mean(dim=1) ###################
#         x = x[:,1]
#         mean = self.mean_linear(x)
#         log_std = self.log_std_linear(x)
#         log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

#         return mean, log_std
#     def sample(self, state,reward,timesteps):
#         # print(state.shape)
#         mean, log_std = self.forward(state,reward,timesteps)
#         std = log_std.exp()
#         normal = Normal(mean, std)
#         x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
#         y_t = torch.tanh(x_t)
#         action = y_t * self.action_scale + self.action_bias
#         log_prob = normal.log_prob(x_t)
#         # Enforcing Action Bound
#         log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
#         log_prob = log_prob.sum(2, keepdim=False)
#         # log_prob = log_prob.sum(1, keepdim=True)
#         mean = torch.tanh(mean) * self.action_scale + self.action_bias
#         # print(action.shape)
#         return action, log_prob, mean

#     def to(self, device):
#         self.action_scale = self.action_scale.to(device)
#         self.action_bias = self.action_bias.to(device)
#         return super(Pformer, self).to(device)
    
class PformerD(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            action_space=None,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)
        self.noise = torch.Tensor(act_dim)

        self.transformer = GPT2Model(config)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_reward = torch.nn.Linear(1, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)
        self.mean_linear = nn.Linear(hidden_size, act_dim)
        self.log_std_linear = nn.Linear(hidden_size, act_dim)
        self.apply(weights_init_)


    def forward(self, states,reward, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]
        reward = reward.reshape(batch_size, seq_length, 1)

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
            attention_mask = attention_mask.to(states.device)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        time_embeddings = self.embed_timestep(timesteps)


        reward_embeddings = self.embed_reward(reward)
        reward_embeddings = reward_embeddings + time_embeddings



        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        # state_embeddings = self.embed_ln(state_embeddings)

        stacked_inputs = torch.stack(
            (reward_embeddings, state_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 2*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 2*seq_length)


        # we feed in the input embeddings (not word indices as in NLP) to the model
        # transformer_outputs = self.transformer(
        #     inputs_embeds=state_embeddings,
        #     attention_mask=attention_mask,
        # )
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            output_hidden_states=True
        )
        x = transformer_outputs['last_hidden_state']
        # x = transformer_outputs.hidden_states[2]
        

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        # x = x.reshape(batch_size, seq_length, self.hidden_size)
        x = x.reshape(batch_size, seq_length, 2, self.hidden_size).permute(0, 2, 1, 3)
        # x = x.mean(dim=1) ###################
        x = x[:,1]
        mean = self.mean_linear(x)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return mean
    def sample(self, state,reward,timesteps):
        mean = self.forward(state,reward,timesteps)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(PformerD, self).to(device)


# class Pformer(nn.Module):
#     def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
#         super(Pformer, self).__init__()
        
#         self.linear1 = nn.Linear(num_inputs, hidden_dim)
#         self.linear2 = nn.Linear(hidden_dim, hidden_dim)

#         self.mean_linear = nn.Linear(hidden_dim, num_actions)
#         self.log_std_linear = nn.Linear(hidden_dim, num_actions)

#         self.apply(weights_init_)


#         # action rescaling
#         if action_space is None:
#             self.action_scale = torch.tensor(1.)
#             self.action_bias = torch.tensor(0.)
#         else:
#             self.action_scale = torch.FloatTensor(
#                 (action_space.high - action_space.low) / 2.)
#             self.action_bias = torch.FloatTensor(
#                 (action_space.high + action_space.low) / 2.)
#         self.action_scale = self.action_scale.to("cuda")
#         self.action_bias = self.action_bias.to("cuda")  


#     def forward(self, state):
#         x = F.relu(self.linear1(state))
#         x = F.relu(self.linear2(x))
#         mean = self.mean_linear(x)
#         log_std = self.log_std_linear(x)
#         log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
#         return mean, log_std

#     def sample(self, state):
#         mean, log_std = self.forward(state)
#         std = log_std.exp()
#         normal = Normal(mean, std)
#         x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
#         y_t = torch.tanh(x_t)
#         action = y_t * self.action_scale + self.action_bias
#         log_prob = normal.log_prob(x_t)
#         # Enforcing Action Bound
#         # print(log_prob.shape)
#         log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
#         if len(log_prob.shape) == 3:
#             log_prob = log_prob.sum(2, keepdim=False)
#         else:
#             log_prob = log_prob.sum(1, keepdim=True)
#         # log_prob = log_prob.sum(2, keepdim=False)
#         mean = torch.tanh(mean) * self.action_scale + self.action_bias

#         return action, log_prob, mean

class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
