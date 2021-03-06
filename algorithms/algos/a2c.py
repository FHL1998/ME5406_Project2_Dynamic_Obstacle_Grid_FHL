import numpy
import torch
import torch.nn.functional as F

from algorithms.algos.base import BaseAlgo


class A2CAlgo(BaseAlgo):
    """The Advantage Actor-Critic algorithm."""

    def __init__(self, envs, actor_critic_model, device=None, num_frames_per_proc=None, discount=0.99, lr=0.01, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-8, preprocess_obss=None):
        num_frames_per_proc = num_frames_per_proc or 8

        super().__init__(envs, actor_critic_model, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss)

        self.optimizer = torch.optim.RMSprop(self.actor_critic_model.parameters(), lr, alpha=rmsprop_alpha, eps=rmsprop_eps)

    def update_parameters(self, exps):
        # Compute starting indexes

        global memory
        inds = self._get_starting_indexes()

        # Initialize update values

        update_entropy = 0
        update_value = 0
        update_policy_loss = 0
        update_value_loss = 0
        update_loss = 0

        # Initialize memory

        if self.actor_critic_model.recurrent:
            memory = exps.memory[inds]

        for i in range(self.recurrence):
            # Create a sub-batch of experience
            sub_batch = exps[inds + i]

            # Compute loss
            if self.actor_critic_model.recurrent:
                dist, value, memory = self.actor_critic_model(sub_batch.obs, memory * sub_batch.mask)
            else:
                dist, value = self.actor_critic_model(sub_batch.obs)
            entropy = dist.entropy().mean()
            policy_loss = -(dist.log_prob(sub_batch.action) * sub_batch.advantage).mean()
            value_loss = (value - sub_batch.returnn).pow(2).mean()
            loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

            # Update batch values
            update_entropy += entropy.item()
            update_value += value.mean().item()
            update_policy_loss += policy_loss.item()
            update_value_loss += value_loss.item()
            update_loss += loss

        # Update update values
        update_entropy /= self.recurrence
        update_value /= self.recurrence
        update_policy_loss /= self.recurrence
        update_value_loss /= self.recurrence
        update_loss /= self.recurrence

        # Update actor-critic
        self.optimizer.zero_grad()
        update_loss.backward()
        update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.actor_critic_model.parameters()) ** 0.5
        torch.nn.utils.clip_grad_norm_(self.actor_critic_model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Log some values

        logs = {
            "entropy": update_entropy,
            "value": update_value,
            "policy_loss": update_policy_loss,
            "value_loss": update_value_loss,
            "grad_norm": update_grad_norm
        }
        return logs

    def _get_starting_indexes(self):
        """At first, it returns the indices of the observations fed into the model
        and the experiences needed to compute the loss.

        The indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`. If the model is not recurrent, they are all the
        integers from 0 to `self.num_frames`.

        Returns
        -------
        starting_indexes : list of int
            the indexes of the experiences to be used at first
        """

        starting_indexes = numpy.arange(0, self.num_frames, self.recurrence)
        return starting_indexes
