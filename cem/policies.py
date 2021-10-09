from typing import Any, Dict, List, Optional, Type

import gym
import torch as th
from gym import spaces
from torch import nn

from stable_baselines3.common.policies import BasePolicy, register_policy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, NatureCNN, create_mlp


class CEMPolicy(BasePolicy):
    """
    Policy network for CEM.
    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: Optional[List[int]] = None,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images=normalize_images,
            squash_output=isinstance(action_space, spaces.Box),
        )

        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                # Small network otherwise sampling is slow
                net_arch = [64]

        self.net_arch = net_arch
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim
        self.activation_fn = activation_fn

        if isinstance(action_space, spaces.Box):
            action_dim = get_action_dim(action_space)
            actor_net = create_mlp(self.features_dim, action_dim, net_arch, activation_fn, squash_output=True)
        elif isinstance(action_space, spaces.Discrete):
            actor_net = create_mlp(self.features_dim, action_space.n, net_arch, activation_fn)
        else:
            raise NotImplementedError("Error: CEM policy not implemented for action space" f"of type {type(action_space)}.")

        # Deterministic action
        self.action_net = nn.Sequential(*actor_net)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                activation_fn=self.activation_fn,
            )
        )
        return data

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # only outputs deterministic actions for now
        features = self.extract_features(obs)
        if isinstance(self.action_space, spaces.Box):
            return self.action_net(features)
        elif isinstance(self.action_space, spaces.Discrete):
            logits = self.action_net(features)
            return th.argmax(logits, dim=1)
        else:
            raise NotImplementedError()

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of CEM.
        #   Predictions are always deterministic for now.
        return self.forward(observation)


MlpPolicy = CEMPolicy

register_policy("MlpPolicy", CEMPolicy)
