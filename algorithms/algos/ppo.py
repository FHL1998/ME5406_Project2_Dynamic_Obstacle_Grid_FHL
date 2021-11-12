import numpy
import torch
import torch.nn.functional as F

from algorithms.algos.base import BaseAlgo


class PPOAlgo(BaseAlgo):
    """The Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, actor_critic_model, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, actor_critic_model, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        assert self.batch_size % self.recurrence == 0

        self.optimizer = torch.optim.Adam(self.actor_critic_model.parameters(), lr, eps=adam_eps)
        self.batch_num = 0

    def update_parameters(self, exps):
        # Collect experiences

        global memory, log_entropies, log_values, log_policy_losses, log_value_losses, log_grad_norms
        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            for index in self._get_batches_starting_indexes():
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # Initialize memory

                if self.actor_critic_model.recurrent:
                    memory = exps.memory[index]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience
                    sub_batch = exps[index + i]

                    # Compute loss
                    if self.actor_critic_model.recurrent:
                        dist, value, memory = self.actor_critic_model(sub_batch.obs, memory * sub_batch.mask)
                    else:
                        dist, value = self.actor_critic_model(sub_batch.obs)
                    entropy = dist.entropy().mean()
                    ratio = torch.exp(dist.log_prob(sub_batch.action) - sub_batch.log_prob)
                    surr1 = ratio * sub_batch.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sub_batch.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # V loss为两部分取大值：第一部分是网络预测value值和R的差平方；第二部分是被clip过的预测value值和return的差平方。
                    # 主要目的应该是惩罚value值的过大更新。
                    value_clipped = sub_batch.value + torch.clamp(value - sub_batch.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sub_batch.returnn).pow(2)
                    surr2 = (value_clipped - sub_batch.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    # loss = policy_loss - ent_coef * entropy + vf_coef * vf_loss
                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                    # Update memories for next epoch

                    if self.actor_critic_model.recurrent and i < self.recurrence - 1:
                        exps.memory[index + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic

                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.actor_critic_model.parameters()) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.actor_critic_model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)

        # Log some values

        logs = {
            "entropy": numpy.mean(log_entropies),
            "value": numpy.mean(log_values),
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
            "grad_norm": numpy.mean(log_grad_norms)
        }

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.recurrence) % self.num_frames_per_proc != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i + num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes
