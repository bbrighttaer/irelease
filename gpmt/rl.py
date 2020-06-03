# Author: bbrighttaer
# Project: GPMT
# Date: 4/9/2020
# Time: 8:02 PM
# File: rl.py

from __future__ import absolute_import, division, print_function, unicode_literals

from collections import namedtuple
import copy
import numpy as np
import torch
import torch.nn.functional as F
from ptan.actions import ActionSelector
from ptan.agent import BaseAgent
from torch.optim.lr_scheduler import StepLR
from tqdm import trange

from gpmt.utils import seq2tensor, get_default_tokens, pad_sequences, canonical_smiles

EpisodeStep = namedtuple('EpisodeStep', ['state', 'action'])
Trajectory = namedtuple('Trajectory', ['terminal_state', 'traj_prob'])


class MolEnvProbabilityActionSelector(ActionSelector):
    """Selects an action"""

    def __init__(self, actions):
        self.actions = actions

    def __call__(self, probs):
        assert isinstance(probs, np.ndarray)
        action_idx = np.random.choice(len(self.actions), p=probs)
        action_prob = probs[action_idx]
        action = self.actions[action_idx]
        return action, action_prob


class StateActionProbRegistry:
    """Helper class to retrieve action probabilities"""

    def __init__(self):
        self._probs_dict = {}

    def add(self, state, action, prob):
        assert isinstance(state, list) and isinstance(action, str) and isinstance(prob, float)
        self._probs_dict[(''.join(state), action)] = prob

    def get(self, state, action):
        """
        Retrieves the probability of the action in the given state.
        :param state: list
        :param action: str
        :return: float
        """
        assert isinstance(state, list) and isinstance(action, str)
        return self._probs_dict[(''.join(state), action)]

    def clear(self):
        self._probs_dict.clear()


class PolicyAgent(BaseAgent):
    def __init__(self, model, action_selector, states_preprocessor=seq2tensor, initial_state=None,
                 initial_state_args=None, apply_softmax=True, device='cpu', probs_registry=None):
        assert callable(states_preprocessor)
        if probs_registry:
            assert isinstance(probs_registry, StateActionProbRegistry)
        if initial_state:
            assert callable(initial_state)
            assert isinstance(initial_state_args, dict)
        self.model = model
        self.action_selector = action_selector
        self.states_preprocessor = states_preprocessor
        self.apply_softmax = apply_softmax
        self.device = device
        self.init_state = initial_state
        self.initial_state_args = initial_state_args
        self.probs_reg = probs_registry

    def initial_state(self):
        return self.init_state(batch_size=1, **self.initial_state_args)

    @torch.no_grad()
    def __call__(self, states, agent_states=None, **kwargs):
        """
        Selects agent actions.

        :param states: state of the environment
        :param agent_states: hidden states (in the case of RNNs)
        :return: action and agent states
        """
        if 'monte_carlo' in kwargs and kwargs['monte_carlo'] and agent_states is None:
            agent_states = [self.initial_state()]
        if agent_states is None:
            agent_states = [None] * len(states)
        state, agent_states = states[0][-1], agent_states[0]
        state, _ = self.states_preprocessor(state, self.action_selector.actions)
        state = torch.from_numpy(state).long().to(self.device)
        x = [state] + agent_states
        outputs = self.model(x)
        if isinstance(outputs, list):  # RNN case
            probs_v = outputs[0][-1]
            agent_states = outputs[1:]
        else:  # trans-decoder
            probs_v = outputs
        if self.apply_softmax:
            probs_v = torch.softmax(probs_v, dim=-1)
        probs = probs_v.data.cpu().squeeze().numpy()
        action, action_prob = self.action_selector(probs)
        if self.probs_reg and 'monte_carlo' not in kwargs:
            self.probs_reg.add(list(states[0]), action, float(action_prob))
        return action, [agent_states]


class DRLAlgorithm(object):
    """Base class for all DRL algorithms"""

    def fit(self, *args, **kwargs):
        """Implements the training procedure of the algorithm"""
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        self.fit(*args, **kwargs)


def calc_Qvals(rewards, gamma):
    qval = []
    sum_r = 0.
    for r in reversed(rewards):
        sum_r *= gamma
        sum_r += r
        qval.append(sum_r)
    return list(reversed(qval))


def unpack_batch(trajs, gamma):
    batch_states, batch_actions, batch_qvals = [], [], []
    for traj in trajs:
        rewards = []
        for exp in traj:
            batch_states.append(exp.state)
            batch_actions.append(exp.action)
            rewards.append(exp.reward)
        batch_qvals.extend(calc_Qvals(rewards, gamma))
    return batch_states, batch_actions, batch_qvals


def unpack_trajectory(traj, gamma):
    states, actions, rewards = [], [], []
    for exp in traj:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
    q_values = calc_Qvals(rewards, gamma)
    return states, actions, q_values


class REINFORCE(DRLAlgorithm):
    def __init__(self, model, optimizer, initial_states_func, initial_states_args, gamma=0.97, grad_clipping=None,
                 lr_decay_gamma=0.1, prior_data_gen=None, xent_lambda=0.3, lr_decay_step=100, device='cpu'):
        assert callable(initial_states_func)
        assert isinstance(initial_states_args, dict)
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)
        self.device = device
        self.gamma = gamma
        self.initial_states_func = initial_states_func
        self.initial_states_args = initial_states_args
        self.grad_clipping = grad_clipping
        self.prior_data_gen = prior_data_gen
        self.xent_lambda = xent_lambda

    @torch.enable_grad()
    def fit(self, trajectories):
        """
        Implements the REINFORCE training algorithm.

        Arguments:
        --------------
        :param trajectories: list
        """
        rl_loss = 0.
        self.optimizer.zero_grad()
        for t in trange(len(trajectories), desc='REINFORCE opt...'):
            trajectory = trajectories[t]
            states, actions, q_values = unpack_trajectory(trajectory, self.gamma)
            (states, state_len), actions = _preprocess_states_actions(actions, states, self.device)
            hidden_states = self.initial_states_func(1, **self.initial_states_args)
            trajectory_input = states[-1]  # since the last state captures all previous states
            for p in range(len(trajectory)):
                outputs = self.model([trajectory_input[p].reshape(1, 1)] + hidden_states)
                output, hidden_states = outputs[0], outputs[1:]
                log_prob = torch.log_softmax(output.view(1, -1), dim=1)
                top_i = actions[p]
                rl_loss = rl_loss - (q_values[p] * log_prob[0, top_i])

        # Ensure pretraining effort isn't wiped out.
        xent_loss = 0.
        # if self.prior_data_gen is not None:
        #     criterion = torch.nn.CrossEntropyLoss()
        #     for i in range(10):
        #         inputs, labels = self.prior_data_gen.random_training_set(batch_size=1)
        #         hidden_states = self.initial_states_func(inputs.shape[0], **self.initial_states_args)
        #         outputs = self.model([inputs] + hidden_states)
        #         predictions = outputs[0]
        #         predictions = predictions.permute(1, 0, -1)
        #         predictions = predictions.contiguous().view(-1, predictions.shape[-1])
        #         labels = labels.contiguous().view(-1)
        #         xent_loss = xent_loss + criterion(predictions, labels)

        xent_loss = xent_loss / len(trajectories)
        rl_loss = (1 - self.xent_lambda) * (rl_loss / len(trajectories)) + self.xent_lambda * xent_loss
        rl_loss.backward()
        if self.grad_clipping is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clipping)
        self.optimizer.step()
        self.lr_scheduler.step()
        return rl_loss.item()


def _preprocess_states_actions(actions, states, device):
    # Process states and actions
    states = [''.join(list(state)) for state in states]
    states, states_len = pad_sequences(states)
    states, _ = seq2tensor(states, get_default_tokens())
    states = torch.from_numpy(states).long().to(device)
    states_len = torch.tensor(states_len).long().to(device)
    actions, _ = seq2tensor(actions, get_default_tokens())
    actions = torch.from_numpy(actions.reshape(-1)).long().to(device)
    return (states, states_len), actions


class PPO(DRLAlgorithm):
    """
    Proximal Policy Optimization, see: https://arxiv.org/abs/1707.06347
    Credits:
    https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter15/04_train_ppo.py
    for a good tutorial on PPO.

    Arguments:
    -----------
    :param actor:
    :param critic:
    :param actor_opt:
    :param critic_opt:
    :param initial_states_func:
    :param device:
    """

    def __init__(self, actor, critic, actor_opt, critic_opt, initial_states_func, initial_states_args, gamma=0.99,
                 gae_lambda=0.95, ppo_eps=0.2, ppo_epochs=10, ppo_batch=64, device='cpu'):
        assert callable(initial_states_func)
        assert isinstance(initial_states_args, dict)
        self.actor = actor
        self.critic = critic
        self.actor_opt = actor_opt
        self.critic_opt = critic_opt
        self.initial_states_func = initial_states_func
        self.initial_states_args = initial_states_args
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_eps = ppo_eps
        self.ppo_epochs = ppo_epochs
        self.ppo_batch = ppo_batch

    @property
    def model(self):
        return self.actor

    def calc_adv_ref(self, trajectory):
        states, actions, _ = unpack_batch([trajectory], self.gamma)
        last_state = ''.join(list(states[-1]))
        inp, _ = seq2tensor([last_state], tokens=get_default_tokens())
        inp = torch.from_numpy(inp).long().to(self.device)
        values_v = self.critic(inp)
        values = values_v.view(-1, ).data.cpu().numpy()
        last_gae = 0.0
        result_adv = []
        result_ref = []
        for val, next_val, exp in zip(reversed(values[:-1]), reversed(values[1:]), reversed(trajectory[:-1])):
            if exp.last_state is None:  # for terminal state
                delta = exp.reward - val
                last_gae = delta
            else:
                delta = exp.reward + self.gamma * next_val - val
                last_gae = delta + self.gamma * self.gae_lambda * last_gae
            result_adv.append(last_gae)
            result_ref.append(last_gae + val)

        adv_v = torch.FloatTensor(list(reversed(result_adv))).to(self.device)
        ref_v = torch.FloatTensor(list(reversed(result_ref))).to(self.device)
        return states[:-1], actions[:-1], adv_v, ref_v

    @torch.enable_grad()
    def fit_batch(self, trajectories):
        # Calculate GAE
        batch_states, batch_actions, batch_adv, batch_ref = [], [], [], []
        for traj in trajectories:
            states, actions, adv_v, ref_v = self.calc_adv_ref(traj)
            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_adv.extend(adv_v)
            batch_ref.extend(ref_v)

        # Normalize advantages
        batch_ref = torch.tensor(batch_ref).float().to(self.device)
        batch_adv = torch.tensor(batch_adv).float().to(self.device)
        batch_adv = (batch_adv - batch_adv.mean()) / batch_adv.std()

        # Calculate old probs of actions
        (states, states_len), actions, = _preprocess_states_actions(batch_actions, batch_states, self.device)
        hidden_states = self.initial_states_func(batch_size=states.shape[0], **self.initial_states_args)
        with torch.set_grad_enabled(False):
            outputs = self.actor([states] + hidden_states)
        x = outputs[0]
        states_len = states_len - 1  # to select actions since samples are padded
        x = torch.cat([x[states_len[i], i, :].reshape(1, -1) for i in range(x.shape[1])], dim=0).to(self.device)
        old_log_probs = torch.log_softmax(x, dim=-1).detach()
        old_log_probs = old_log_probs[range(old_log_probs.shape[0]), actions]

        sum_loss_value = []
        sum_loss_policy = []

        for epoch in trange(self.ppo_epochs, desc='PPO optimization...'):
            for batch_ofs in range(0, len(batch_states), self.ppo_batch):
                # Select batch data
                states_v = states[batch_ofs:batch_ofs + self.ppo_batch]
                states_len_v = states_len[batch_ofs:batch_ofs + self.ppo_batch]
                actions_v = actions[batch_ofs:batch_ofs + self.ppo_batch]
                batch_adv_v = batch_adv[batch_ofs:batch_ofs + self.ppo_batch]
                batch_ref_v = batch_ref[batch_ofs:batch_ofs + self.ppo_batch]
                old_log_probs_v = old_log_probs[batch_ofs:batch_ofs + self.ppo_batch]
                hidden_states_v = self.initial_states_func(batch_size=states_v.shape[0], **self.initial_states_args)

                # Critic training
                self.critic_opt.zero_grad()
                value_v = self.critic(states_v)
                value_v = torch.cat([value_v[states_len_v[i], i, :] for i in range(value_v.shape[1])], dim=0)
                value_v = value_v.to(self.device)
                loss_value_v = F.mse_loss(value_v, batch_ref_v)
                loss_value_v.backward()
                self.critic_opt.step()
                sum_loss_value.append(loss_value_v.item())

                # Actor training
                self.actor_opt.zero_grad()
                x_v = self.actor([states_v] + hidden_states_v)
                x_v = x_v[0]
                x_v = torch.cat([x_v[states_len_v[i], i, :].reshape(1, -1) for i in range(x_v.shape[1])], dim=0)
                logprob_pi_v = torch.log_softmax(x_v, dim=-1)
                logprob_pi_v = logprob_pi_v[range(logprob_pi_v.shape[0]), actions_v]
                ratio_v = torch.exp(logprob_pi_v - old_log_probs_v)
                surr_obj_v = batch_adv_v * ratio_v
                clipped_surr_v = batch_adv_v * torch.clamp(ratio_v, 1.0 - self.ppo_eps, 1.0 + self.ppo_eps)
                loss_policy_v = torch.min(surr_obj_v, clipped_surr_v).mean()
                sum_loss_policy.append(loss_policy_v.item())
                loss_policy_v = -loss_policy_v  # for maximization
                loss_policy_v.backward()
                self.actor_opt.step()
        return np.mean(sum_loss_value), np.mean(sum_loss_policy)

    @torch.enable_grad()
    def fit(self, trajectories):
        sq2ten = lambda x: torch.from_numpy(seq2tensor(x, get_default_tokens())[0]).long().to(self.device)
        t_states, t_actions, t_adv, t_ref = [], [], [], []
        t_old_probs = []
        for traj in trajectories:
            states, actions, adv_v, ref_v = self.calc_adv_ref(traj)
            if len(states) == 0:
                continue
            t_states.append(states)
            t_actions.append(actions)
            t_adv.append(adv_v)
            t_ref.append(ref_v)

            with torch.set_grad_enabled(False):
                hidden_states = self.initial_states_func(batch_size=1, **self.initial_states_args)
                trajectory_input = sq2ten(states[-1])
                actions = sq2ten(actions)
                old_probs = []
                for p in range(len(trajectory_input)):
                    outputs = self.model([trajectory_input[p].reshape(1, 1)] + hidden_states)
                    output, hidden_states = outputs[0], outputs[1:]
                    log_prob = torch.log_softmax(output.view(1, -1), dim=1)
                    old_probs.append(log_prob[0, actions[p]].item())
                t_old_probs.append(old_probs)

        if len(t_states) == 0:
            return 0., 0.

        for epoch in trange(self.ppo_epochs, desc='PPO optimization...'):
            cr_loss = 0.
            ac_loss = 0.
            for i in range(len(t_states)):
                traj_last_state = t_states[i][-1]
                traj_actions = t_actions[i]
                traj_adv = t_adv[i]
                traj_ref = t_ref[i]
                traj_old_probs = t_old_probs[i]
                hidden_states = self.initial_states_func(1, **self.initial_states_args)
                for p in range(len(traj_last_state)):
                    state, action, adv = traj_last_state[p], traj_actions[p], traj_adv[p]
                    old_log_prob = traj_old_probs[p]
                    state, action = sq2ten(state), sq2ten(action)

                    # Critic
                    pred = self.critic(state)
                    cr_loss = cr_loss + F.mse_loss(pred.reshape(-1, 1), traj_ref[p].reshape(-1, 1))

                    # Actor
                    outputs = self.actor([state] + hidden_states)
                    output, hidden_states = outputs[0], outputs[1:]
                    logprob_pi_v = torch.log_softmax(output.view(1, -1), dim=-1)
                    logprob_pi_v = logprob_pi_v[0, action]
                    ratio_v = torch.exp(logprob_pi_v - old_log_prob)
                    surr_obj_v = adv * ratio_v
                    clipped_surr_v = adv * torch.clamp(ratio_v, 1.0 - self.ppo_eps, 1.0 + self.ppo_eps)
                    loss_policy_v = torch.min(surr_obj_v, clipped_surr_v)
                    ac_loss = ac_loss - loss_policy_v
            # Update weights
            self.critic_opt.zero_grad()
            self.actor_opt.zero_grad()
            cr_loss = cr_loss / len(trajectories)
            ac_loss = ac_loss / len(trajectories)
            cr_loss.backward()
            ac_loss.backward()
            self.critic_opt.step()
            self.actor_opt.step()
        return cr_loss.item(), -ac_loss.item()


class GuidedRewardLearningIRL(DRLAlgorithm):
    """
    Implementation of:
    “Guided Cost Learning : Deep Inverse Optimal Control via Policy Optimization,” vol. 48, 2016.
    """

    def __init__(self, model, optimizer, demo_gen_data, agent_net, agent_net_init_func, agent_net_init_func_args, k=10,
                 use_buffer=True, buffer_size=1000, buffer_batch_size=100, drop_importance_wts=False, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.lr_sch = StepLR(self.optimizer, gamma=0.95, step_size=500)
        self.demo_gen_data = demo_gen_data
        self.k = k
        self.device = device
        self.use_buffer = use_buffer
        if use_buffer:
            self.replay_buffer = TrajectoriesBuffer(buffer_size)
        else:
            self.replay_buffer = None
        self.batch_size = buffer_batch_size
        self.agent_net = agent_net
        self.agent_net_init_func = agent_net_init_func
        self.agent_net_init_args = agent_net_init_func_args
        self._models = torch.nn.ModuleList()
        self.drop_imp_wts = drop_importance_wts

    @property
    def data_generator(self):
        return self.demo_gen_data

    @torch.no_grad()
    def calculate_z(self, inp, actions, seq_lens):
        if self.drop_imp_wts:
            return torch.ones(len(inp), 1).to(self.device)
        seq_lens = torch.tensor(seq_lens).long() - 1
        z = torch.zeros(inp.shape[0], len(self._models)).to(self.device)
        hidden = self.agent_net_init_func(inp.shape[0], **self.agent_net_init_args)
        for idx, model in enumerate(self._models):
            model = model.to(self.device)
            outputs = model([inp] + hidden)
            output = outputs[0].permute(1, 0, 2)
            output = output.reshape(-1, output.shape[-1])
            probs = torch.log_softmax(output, dim=-1)  # calculate q(t) in log space
            probs_a = probs[range(len(probs)), actions.view(-1)].view(*actions.shape)
            probs_a = torch.tensor([torch.sum(probs_a[i, :seq_lens[i]]) for i in range(len(probs_a))]).to(self.device)
            z[:, idx] = probs_a
        z = 1. / torch.mean(z, dim=1)
        return z

    @torch.enable_grad()
    def fit(self, trajectories):
        """Train the reward function / model using the GRL algorithm."""
        self._models.append(copy.deepcopy(self.agent_net).to('cpu'))  # maintain history of sample distributions
        if self.use_buffer:
            extra_trajs = self.replay_buffer.sample(self.batch_size)
            trajectories.extend(extra_trajs)
            self.replay_buffer.populate(trajectories)
        d_traj, d_traj_probs = [], []
        for traj in trajectories:
            d_traj.append(''.join(list(traj.terminal_state.state)) + traj.terminal_state.action)
            d_traj_probs.append(traj.traj_prob)
        _, valid_vec_samp = canonical_smiles(d_traj)
        valid_vec_samp = torch.tensor(valid_vec_samp).view(-1, 1).float().to(self.device)
        inp, target = [t[:-1] for t in d_traj], [t[1:] for t in d_traj]
        inp_padded, inp_seq_len = pad_sequences(inp)
        inp_tensor, _ = seq2tensor(inp_padded, tokens=get_default_tokens(), flip=False)
        inp_tensor = torch.from_numpy(inp_tensor).long().to(self.device)
        target_padded, tag_seq_len = pad_sequences(target)
        target_tensor, _ = seq2tensor(target_padded, tokens=get_default_tokens(), flip=False)
        target_tensor = torch.from_numpy(target_tensor).long().to(self.device)
        d_samp = torch.cat([inp_tensor, target_tensor[:, -1].reshape(-1, 1)], dim=1)
        z_samp = self.calculate_z(inp_tensor, target_tensor, inp_seq_len)
        losses = []
        for i in trange(self.k, desc='IRL optimization...'):
            # D_demo processing
            demo_states, demo_actions, demo_seq_lens = self.demo_gen_data.random_training_set(return_seq_len=True)
            z_demo = self.calculate_z(demo_states, demo_actions, demo_seq_lens[0])
            d_demo = torch.cat([demo_states, demo_actions[:, -1].reshape(-1, 1)], dim=1)
            valid_vec_demo = torch.ones(d_demo.shape[0]).view(-1, 1).float().to(self.device)
            d_demo_out = self.model([d_demo, valid_vec_demo])

            # D_samp processing
            d_samp_out = self.model([d_samp, valid_vec_samp])
            if d_samp_out.shape[0] < 100:
                d_samp_out = torch.cat([d_samp_out, d_demo_out], dim=0)
                z_samp = torch.cat([z_samp, z_demo])
            d_samp_out = z_samp * torch.exp(d_samp_out)

            # objective
            loss = torch.mean(d_demo_out) - torch.log(torch.mean(d_samp_out))
            losses.append(loss.item())
            loss = -loss  # for maximization

            # update params
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.lr_sch.step()
        return np.mean(losses)


class TrajectoriesBuffer:
    """
    Stores trajectories generated by different background distributions
    adapted from ptan.ExperienceReplayBuffer
    """

    def __init__(self, buffer_size):
        self.capacity = buffer_size
        self.buffer = []
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def sample(self, batch_size):
        if len(self.buffer) <= batch_size:
            return self.buffer
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        return [self.buffer[key] for key in keys]

    def _add(self, trajectory):
        assert isinstance(trajectory, Trajectory)
        if len(self.buffer) < self.capacity:
            self.buffer.append(trajectory)
        else:
            self.buffer[self.pos] = trajectory
        self.pos = (self.pos + 1) % self.capacity

    def populate(self, samples):
        for entry in samples:
            self._add(entry)
