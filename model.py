import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
from torch.distributions.categorical import Categorical


# Function from https://github.com/ikostrikov/py_toporch-a2c-ppo-acktr/blob/master/model.py
# also refer to https://github.com/facebookresearch/modeling_long_term_future
# import algorithms


class ACModel:
    recurrent = False

    @abstractmethod
    def __init__(self, obs_space, action_space):
        pass

    @abstractmethod
    def forward(self, obs):
        pass


class RecurrentACModel(ACModel):
    recurrent = True

    @abstractmethod
    def forward(self, obs, memory):
        pass

    @property
    @abstractmethod
    def memory_size(self):
        pass


def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACModel(nn.Module, RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False):
        super().__init__()

        # Decide which components are enabled
        self.use_memory = use_memory

        # Define image embedding
        # Conv2d(in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1,
        # groups=1,bias=True, padding_mode=‘zeros’)
        self.image_conv = nn.Sequential(
            # 3 input image channel, 16 output channels, 2x2 square convolution kernel
            nn.Conv2d(3, 16, (2, 2)),  # （nSample）x C x H x W
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]

        # for FOV=7 -> m=n=7 -> self.image_embedding_size=64
        self.image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Resize image embedding
        # semi_memory_size = image_embedding_size
        self.embedding_size = self.semi_memory_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    # 只要在nn.Module的子类中定义了forward函数，backward函数就会被自动实现(利用Autograd)
    # [batch_size,output channel, height, width]
    def forward(self, obs, memory):
        # obs.image torch.Size([6, 7, 7, 3]) -> torch.Size([6, 3, 7, 7]) ->torch.Size([6, 3, 7, 7])
        x = obs.image.transpose(1, 3).transpose(2, 3)
        # print('x_shape', x.shape)
        x = self.image_conv(x)  # x_shape torch.Size([6, 3, 7, 7]) -> torch.Size([6, 64, 1, 1])
        # print('x_shape', x.shape)
        x = x.reshape(x.shape[0], -1)  # -1表示列数自动计算，d= a*b /m -> x_shape torch.Size([6, 64])
        # print('x_shape', x.shape)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            # torch.cat将两个张量(tensor)拼接在一起
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        # torch.Size([6, 3]) -> torch.Size([6, 64])
        x = self.actor(embedding)
        # print('F.log_softmax(x, dim=1)', F.log_softmax(x, dim=1))
        dist = Categorical(logits=F.log_softmax(x, dim=1))
        # print('dist', dist)
        x = self.critic(embedding)
        # print('x', x)
        value = x.squeeze(1)
        # print('value', value)

        return dist, value, memory
