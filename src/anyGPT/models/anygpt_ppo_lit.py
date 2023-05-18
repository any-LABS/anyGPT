import gymnasium as gym
import lightning.pytorch as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from anyGPT.config.settings import AnyGPTSettings
from anyGPT.data.experience_dataset import ExperienceDataset
from anyGPT.models.ppo_policy import PPOPolicy


class AnyGPTPPOLit(pl.LightningModule):
    def __init__(self, settings: AnyGPTSettings):
        super().__init__()
        self.ppo_config = settings.ppo_config
        self.io_config = settings.io_config
        self.model_config = settings.model_config
        self.env = self._make_env()
        self.policy = PPOPolicy(settings)
        self.save_hyperparameters()
        self.state = torch.tensor(self.env.reset()[0], dtype=torch.long)[None, ...]
        self.batch_states = []
        self.batch_actions = []
        self.batch_adv = []
        self.batch_qvals = []
        self.batch_logp = []
        self.batch_logp_ref = []
        self.ep_rewards = []
        self.ep_values = []
        self.epoch_rewards = []
        self.avg_ep_reward = 0.0

    def forward(self, *args, **kwargs):
        return self.policy(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        state, action, old_log_probs, old_log_probs_ref, qvalues, advantages = batch

        advantages = (advantages - advantages.mean()) / advantages.std()

        (
            new_action,
            new_probs,
            new_log_probs,
            new_log_probs_ref,
            new_values,
        ) = self.policy.generate(
            state.squeeze(),
            self.ppo_config.action_size,
            self.model_config.block_size,
            self.device,
        )

        # losses
        # TODO add pretraining loss from InstructGPT paper

        actor_loss = self._actor_loss(new_log_probs, old_log_probs, advantages)
        critic_loss = self._critic_loss(new_values, qvalues)
        kl_loss = self._kl_penalty(new_probs, new_log_probs, new_log_probs_ref)
        entropy_loss, entropy = self._entropy_loss(new_probs, new_log_probs)

        loss = actor_loss + 0.5 * critic_loss + kl_loss + entropy_loss

        self.log("rewards/avg_ep_reward", self.avg_ep_reward, prog_bar=True)
        self.log("metrics/entropy", entropy)
        self.log("losses/total_loss", loss)
        self.log("losses/actor_loss", actor_loss)
        self.log("losses/critic_loss", critic_loss)
        self.log("losses/kl_loss", kl_loss)
        self.log("losses/entropy_loss", entropy_loss)

        return loss

    def _actor_loss(self, log_probs, old_log_probs, advantages):
        old_log_probs = old_log_probs.squeeze()
        log_probs = log_probs.squeeze()
        advantages = advantages.squeeze()
        ratio = torch.exp(log_probs - old_log_probs)
        clip_adv = (
            torch.clamp(
                ratio, 1 - self.ppo_config.clip_ratio, 1 + self.ppo_config.clip_ratio
            )
            * advantages
        )
        loss_actor = -(torch.min(ratio * advantages, clip_adv)).mean()
        return loss_actor

    @staticmethod
    def _critic_loss(new_values, qval):
        # loss_critic = (qval.squeeze() - new_values).pow(2).mean()
        loss_critic = F.mse_loss(new_values, qval.squeeze())
        return loss_critic

    def _kl_penalty(self, prob_p, logp, logq):
        logp = logp.squeeze()
        logq = logq.squeeze()
        prob_p = prob_p.squeeze()
        kl_pq = prob_p * (logp - logq)
        kl_loss = self.ppo_config.beta_kl * kl_pq.sum(dim=1).mean()
        return kl_loss

    def _entropy_loss(self, probs, log_probs):
        entropy = -(probs * log_probs).sum(dim=1).mean()
        entropy_loss = -self.ppo_config.beta * entropy
        return entropy_loss, entropy

    # investigate separate optimizers
    # def configure_optimizers(self):
    #     actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(),
    #                                        self.ppo_config.actor_lr)
    #     critic_optimizer = torch.optim.Adam(self.policy.critic.parameters(),
    #                                         self.ppo_config.critic_lr)
    #     return actor_optimizer, critic_optimizer

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.policy.params(), self.ppo_config.learning_rate
        )
        return optimizer

    # def optimizer_step(self, *args, **kwargs):
    #     for _ in range(self.ppo_config.num_optim_iters):
    #         super().optimizer_step(*args, **kwargs)

    def train_dataloader(self):
        return self._dataloader()

    def _sample_trajectories(self):
        for step in range(self.ppo_config.buffer_size):
            self.state = self.state.to(device=self.device)

            with torch.no_grad():
                action, _, log_probs, log_probs_ref, values = self.policy.generate(
                    self.state,
                    self.ppo_config.action_size,
                    self.model_config.block_size,
                    self.device,
                )
            _, reward, terminated, _, _ = self.env.step(action.cpu().numpy())

            self.batch_states.append(self.state)
            self.batch_actions.append(action)
            self.batch_logp.append(log_probs)
            self.batch_logp_ref.append(log_probs_ref)

            # TODO look into how to broadcast reward properly for the sequence, many not need this
            # broadcast reward scaled by probs (alternative is to use sparse reward or scale by logprobs)
            # self.ep_rewards.append(
            #     reward * torch.ones_like(action, dtype=torch.float16) * probs
            # )

            # sparse reward only at the end of the sequence for nwo
            ep_reward = torch.zeros_like(action, dtype=torch.float16)
            ep_reward[:, -1] = reward
            self.ep_rewards.append(ep_reward)
            values[:, self.ppo_config.action_size - 1] = 0.0
            self.ep_values.append(values)

            if terminated:
                self.state = torch.tensor(self.env.reset()[0], dtype=torch.long)[
                    None, ...
                ]

        self.batch_qvals = list(
            torch.unbind(
                self._discount_rewards(self.ep_rewards, self.ppo_config.gamma), dim=0
            )
        )
        self.batch_adv = list(
            torch.unbind(
                self._compute_advantages(self.ep_rewards, self.ep_values), dim=0
            )
        )

        # self.batch_states = torch.reshape(torch.stack(self.batch_states),
        #                                   (self.ppo_config.steps_per_epoch * self.ppo_config.action_size,))
        # self.batch_actions = torch.reshape(torch.stack(self.batch_actions),
        #                                    (self.ppo_config.steps_per_epoch * self.ppo_config.action_size,))
        # self.batch_logp = torch.reshape(torch.stack(self.batch_logp),
        #                                 (self.ppo_config.steps_per_epoch * self.ppo_config.action_size,))
        # self.batch_logp_ref = torch.reshape(torch.stack(self.batch_logp_ref),
        #                                     (self.ppo_config.steps_per_epoch * self.ppo_config.action_size,))
        # self.batch_qvals = torch.reshape(self.batch_qvals,
        #                                  (self.ppo_config.steps_per_epoch * self.ppo_config.action_size,))
        # self.batch_adv = torch.reshape(self.batch_adv,
        #                                (self.ppo_config.steps_per_epoch * self.ppo_config.action_size,))

        self.avg_ep_reward = torch.stack(self.ep_rewards).squeeze().sum(dim=1).mean()

        train_data = zip(
            self.batch_states,
            self.batch_actions,
            self.batch_logp,
            self.batch_logp_ref,
            self.batch_qvals,
            self.batch_adv,
        )
        for state, action, logp, logp_ref, qval, adv in train_data:
            yield state, action, logp, logp_ref, qval, adv

        self.batch_states.clear()
        self.batch_actions.clear()
        self.batch_logp.clear()
        self.batch_logp_ref.clear()
        self.batch_qvals.clear()
        self.batch_adv.clear()

    def _discount_rewards(self, rewards, gamma):
        if isinstance(rewards, list):
            rewards = torch.stack(rewards)
        cumulative_reward = torch.zeros(rewards.shape, dtype=torch.float16).to(
            self.device
        )
        sum_r = torch.zeros((rewards.shape[0], 1), dtype=torch.float16).to(self.device)
        for t in reversed(range(self.ppo_config.action_size)):
            sum_r = (sum_r * gamma) + rewards[:, :, t]
            cumulative_reward[:, :, t] = sum_r

        return cumulative_reward

    def _compute_advantages(self, rewards, values):
        rewards = torch.stack(rewards)
        values = torch.stack(values)
        delta = torch.zeros(rewards.shape, dtype=torch.float16).to(self.device)
        for t in range(self.ppo_config.action_size - 1):
            delta[:, :, t] = (
                rewards[:, :, t]
                + self.ppo_config.gamma * values[:, :, t + 1]
                - values[:, :, t]
            )

        adv = self._discount_rewards(
            delta, self.ppo_config.gamma * self.ppo_config.lamda
        )

        return adv

    def _dataloader(self):
        dataset = ExperienceDataset(self._sample_trajectories)
        dataloader = DataLoader(dataset=dataset, batch_size=self.ppo_config.batch_size)
        return dataloader

    def _make_env(self):
        env = gym.make(
            self.ppo_config.env,
            block_size=self.ppo_config.observation_size,
            **self.ppo_config.env_kwargs
        )
        return env
