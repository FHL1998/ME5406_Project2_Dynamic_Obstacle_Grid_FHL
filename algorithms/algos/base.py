from abc import ABC, abstractmethod
import torch

from algorithms.format import default_preprocess_obss
from algorithms.utils import DictList, ParallelEnv


class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, actor_critic_model, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        actor_critic_model : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update 每个进程每一次更新所收集的帧数
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective 熵成本在最终目标中的权重
        value_loss_coef : float
            the weight of the value loss in the final objective 最终目标中价值损失的权重
        max_grad_norm : float
            gradient will be clipped to be at most this value 梯度将被裁剪为最多这个值
        recurrence : int
            the number of steps the gradient is propagated back in time 梯度反向传播回时间的步数
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle 一个函数，它接受环境返回的观察结果
             并将它们转换成模型可以处理的格式
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input 一个塑造奖励的函数，需要一个
             （观察、行动、奖励、完成）元组作为输入
        """

        # Store parameters

        self.env = ParallelEnv(envs)
        self.actor_critic_model = actor_critic_model
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss

        # at present, both a2c and ppo have none reshape reward function

        # Control parameters
        # for ACModel, recurrent = False, for RecurrentACModel, recurrent=True
        assert self.actor_critic_model.recurrent or self.recurrence == 1
        assert self.num_frames_per_proc % self.recurrence == 0

        # Configure actor_critic_model

        self.actor_critic_model.to(self.device)
        self.actor_critic_model.train()

        # Store helpers values
        # for this case, self.num_procs=6
        self.num_procs = len(envs)

        # num_frame 等于每个进程的frame数 * 进程数 如 4*128=512
        self.num_frames = self.num_frames_per_proc * self.num_procs

        # Initialize experience values
        # shape[0]: num_frames_per_proc每个进程每一次更新所收集的帧数 shape[1]: num_procs并行计算的数量
        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        # 初始化 obss(list) 为每个进程每次更新所收集frames的长度 store the list of observations
        self.obss = [None] * (shape[0])
        if self.actor_critic_model.recurrent:
            self.memory = torch.zeros(shape[1], self.actor_critic_model.memory_size, device=self.device)
            self.memories = torch.zeros(*shape, self.actor_critic_model.memory_size, device=self.device)
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_procs,
                                              device=self.device)  # tensor([0., 0., 0., 0.], device='cuda:0')
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs  # self.log_return [0, 0, 0, 0]
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs  # self.log_num_frames [0, 0, 0, 0]

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.
        同时运行多个环境。 下一个动作是在批处理模式下同时为所有环境计算的。 所有环境的推出和优势都结合在一起。
        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """

        global done, reward
        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction

            # 将 dict 转化为 torch.tensor
            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            # 强制之后的内容不进行计算图构建
            with torch.no_grad():
                if self.actor_critic_model.recurrent:
                    #  unsqueeze（）是来增加一个维度的
                    # dist: distribution
                    dist, value, memory = self.actor_critic_model(preprocessed_obs,
                                                                  self.memory * self.mask.unsqueeze(1))
                else:
                    dist, value = self.actor_critic_model(preprocessed_obs)
            action = dist.sample()

            #  numpy不能读取CUDA tensor 需要将它转化为 CPU tensor, 所以写成.cpu().numpy()
            # 此处的done 是一个 4元数组，存的是每个 sub_process 的 done flag
            obs, reward, done, _ = self.env.step(action.cpu().numpy())

            # Update experiences values

            self.obss[i] = self.obs
            self.obs = obs
            if self.actor_critic_model.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask

            # if done=True, mask=tensor(0., device='cuda:0') if done=False, mask=tensor(1., device='cuda:0')
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value

            self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            #  enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
            for j, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[j].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[j].item())
                    self.log_num_frames.append(self.log_episode_num_frames[j].item())

            # *self.mask 将 done位置的元素置为0
            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            #  for ACModel, recurrent = False, for RecurrentACModel, recurrent=True
            if self.actor_critic_model.recurrent:
                _, next_value, _ = self.actor_critic_model(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            else:
                _, next_value = self.actor_critic_model(preprocessed_obs)

        for i in reversed(range(self.num_frames_per_proc)):
            # 如果索引大于每个进程每一次更新所收集的帧数,则next_mask = self.mask
            next_mask = self.masks[i + 1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i + 1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i + 1] if i < self.num_frames_per_proc - 1 else 0

            # update advantage value
            # δ = r + λQ(s',a') − Q(s,a)
            # A(GAE) = δ + λγδ(t+1)
            # gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        # exps:experience 里包含 action, value, reward, advantage, Q, log_prob, observation
        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        if self.actor_critic_model.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)

        # Q(s,a) =  V(s) + A(s,a)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values
        # 记录的值为 self.log_done_counter 和 self.num_procs(进程数) 中的最大值
        keep = max(self.log_done_counter, self.num_procs)

        # "rewards_shape": self.rewards.shape,  # torch.Size([128, 4])
        # "self.log_episode_return_shape": self.log_episode_return.shape,  # torch.Size([4])
        logs = {
            "dist": dist,
            "done": done,
            "reward": reward,
            "rewards": self.rewards,
            "log_done_counter": self.log_done_counter,
            "return_per_episode": self.log_return[-keep:],
            "return_per_episode_shape": len(self.log_return[-keep:]),
            "self.log_episode_return": self.log_episode_return,
            "self.log_return": self.log_return,
            "self.log_return_shape": len(self.log_return),
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs

    @abstractmethod
    def update_parameters(self, exp):
        pass
