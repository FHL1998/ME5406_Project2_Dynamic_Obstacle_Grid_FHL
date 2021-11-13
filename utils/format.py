import os
import json
import numpy
import re
import torch
import algorithms
import gym

import utils


def get_obss_preprocessor(obs_space):
    # Check if obs_space is an image space
    if isinstance(obs_space, gym.spaces.Box):
        obs_space = {"image": obs_space.shape}

        def preprocess_obss(obss, device=None):
            return algorithms.DictList({
                "image": preprocess_images(obss, device=device)
            })

    # Check if it is a MiniGrid observation space
    elif isinstance(obs_space, gym.spaces.Dict) and list(obs_space.spaces.keys()) == ["image"]:
        obs_space = {"image": obs_space.spaces["image"].shape}

        def preprocess_obss(obss, device=None):
            return algorithms.DictList({
                "image": preprocess_images([obs["image"] for obs in obss], device=device),
            })

    else:
        raise ValueError("Unknown observation space: " + str(obs_space))

    return obs_space, preprocess_obss


def preprocess_images(images, device=None):
    # Bug of Pytorch: very slow if not first converted to numpy array
    images = numpy.array(images)
    return torch.tensor(images, device=device, dtype=torch.float)

