import torch

import utils
from model import ACModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, obs_space, action_space, model_dir,
                 argmax=False, num_envs=1, use_memory=False, use_text=False):
        obs_space, self.preprocess_obss = utils.get_obss_preprocessor(obs_space)
        self.actor_critic_model = ACModel(obs_space, action_space, use_memory=use_memory)
        self.argmax = argmax
        self.num_envs = num_envs

        if self.actor_critic_model.recurrent:
            self.memories = torch.zeros(self.num_envs, self.actor_critic_model.memory_size, device=device)

        self.actor_critic_model.load_state_dict(utils.get_model_state(model_dir))
        self.actor_critic_model.to(device)
        self.actor_critic_model.eval()
        # if hasattr(self.preprocess_obss, "vocab"):
        #     self.preprocess_obss.vocab.load_vocab(utils.get_vocab(model_dir))

    def get_actions(self, obss):
        preprocessed_obss = self.preprocess_obss(obss, device=device)

        with torch.no_grad():
            if self.actor_critic_model.recurrent:
                dist, _, self.memories = self.actor_critic_model(preprocessed_obss, self.memories)
            else:
                dist, _ = self.actor_critic_model(preprocessed_obss)

        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
            # print('actions', actions)
        else:
            actions = dist.sample()
            # print('actions', actions)

        return actions.cpu().numpy()

    def get_action(self, obs):
        return self.get_actions([obs])[0]

    def analyze_feedbacks(self, rewards, dones):
        if self.actor_critic_model.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float, device=device).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])
