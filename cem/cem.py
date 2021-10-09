import time
from copy import deepcopy
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch as th
from gym import spaces

from cem.policies import CEMPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.on_policy_algorithm import BaseAlgorithm, BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.her.her_replay_buffer import get_time_limit


class CEM(BaseAlgorithm):
    """
    The Cross Entropy Method
    The built-in policy corresponds to the centroid of the population at each generation
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param pop_size: Population size (number of individuals)
    :param elit_frac_size: Fraction of elite individuals to keep to compute centroid
        of the next generation
    :param sigma: Initial noise standard deviation.
    :param noise_multiplier: Noise decay. We add noise to the standard deviation
        to avoid early collapse.
    :param n_eval_episodes: Number of episodes to evaluate each individual.
    :param nb_epochs: Number of iterations to run the CEM algorithm,
        this will overwrite the parameter of ``.learn()``
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[CEMPolicy]],
        env: Union[GymEnv, str],
        max_episode_steps: Optional[int] = 700,
        save_path : str = "",
        pop_size: int = 10,
        elit_frac_size: float = 0.2,
        sigma: float = 0.2,
        noise_multiplier: float = 0.999,
        n_eval_episodes: int = 5,
        nb_epochs: Optional[int] = None,
        policy_base: Type[BasePolicy] = CEMPolicy,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super(CEM, self).__init__(
            policy,
            env,
            policy_base=policy_base,
            learning_rate=0.0,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
            ),
            support_multi_env=True,
        )
        self.max_episode_steps = max_episode_steps
        self.pop_size = pop_size
        self.elit_frac_size = elit_frac_size
        self.elites_nb = int(self.elit_frac_size * self.pop_size)
        self.n_eval_episodes = n_eval_episodes
        self.train_policy = None
        self.nb_epochs = nb_epochs
        self.best_score = -np.inf

        self.save_path = save_path

        self.sigma = sigma
        self.noise = None
        self.cov = None
        # Random number generator to sample weights
        # from the Gaussian distribution
        self.rng = None
        self.policy_dim = None
        self.noise_multiplier = noise_multiplier


        if _init_setup_model:
            self._setup_model()

        # Retrieve max episode step automatically
        if self.max_episode_steps is None:
            self.max_episode_steps = get_time_limit(self.env, max_episode_steps)

    def _setup_model(self) -> None:
        self.set_random_seed(self.seed)
        self.rng = np.random.default_rng(self.seed)

        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space, self.action_space, **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)
        params = self.get_params(self.policy)
        self.train_policy = deepcopy(self.policy)
        self.policy_dim = params.shape[0]
        if self.verbose > 0:
            print(f"policy dim: {self.policy_dim}")

        # Init the covariance matrix
        centroid = self.get_params(self.train_policy)
        self.init_covariance(centroid)

    def get_params(self, policy: CEMPolicy) -> np.ndarray:
        return policy.parameters_to_vector()  # note: also retrieves critic params...

    def set_params(self, policy: CEMPolicy, indiv: np.ndarray) -> None:
        policy.load_from_vector(indiv.copy())

    def update_noise(self) -> None:
        self.noise = self.noise * self.noise_multiplier

    def init_covariance(self, centroid: np.ndarray) -> None:
        self.noise = np.diag(np.ones(self.policy_dim) * self.sigma)
        self.cov = np.diag(np.ones(self.policy_dim) * np.var(centroid)) + self.noise

    def learn_one_epoch(self, callback: BaseCallback) -> bool:
        callback.on_rollout_start()

        centroid = self.get_params(self.train_policy)
        self.update_noise()
        scores = np.zeros(self.pop_size)
        weights = self.rng.multivariate_normal(centroid, self.cov, self.pop_size)

        # Separable CEM, useful when self.policy_dim >> 100
        # Use only diagonal of the covariance matrix
        # param_noise = np.random.randn(self.pop_size, self.policy_dim)
        # weights = centroid + param_noise * np.sqrt(np.diagonal(self.cov))

        for i in range(self.pop_size):
            # Evaluate individual
            self.set_params(self.train_policy, weights[i])
            episode_rewards, episode_lengths = evaluate_policy(
                self.train_policy,
                self.env,
                n_eval_episodes=self.n_eval_episodes,
                return_episode_rewards=True,
            )

            scores[i] = np.mean(episode_rewards)
            # Save best params
            # Update if it is same score but more recent policy
            if scores[i] >= self.best_score:
                self.best_score = scores[i]
                self.set_params(self.policy, weights[i])
                self.save( self.save_path + "score_" +str(scores[i])+".zip")

            # Mimic Monitor Wrapper
            infos = [
                {"episode": {"r": episode_reward, "l": episode_length}}
                for episode_reward, episode_length in zip(episode_rewards, episode_lengths)
            ]

            self._update_info_buffer(infos)

            self.num_timesteps += sum(episode_lengths)
            self._episode_num += len(episode_lengths)
            if self.verbose > 0:
                print(f"Indiv: {i + 1} score {scores[i]:.2f}")

        self.logger.record("train/mean_score", np.mean(scores))
        self.logger.record("train/best_score", sorted(scores)[-1])
        self.logger.record("train/diag_std", np.mean(np.sqrt(np.diagonal(self.cov))))
        self.logger.record("train/noise", np.mean(np.diagonal(self.noise)))
        self._dump_logs()

        # Keep only best individuals to compute the new centroid
        elites_idxs = scores.argsort()[-self.elites_nb :]
        elites_weights = weights[elites_idxs]
        centroid = np.array(elites_weights).mean(axis=0)

        # Update covariance
        self.cov = np.cov(elites_weights, rowvar=False) + self.noise
        self.set_params(self.train_policy, centroid)

        # Give access to local variables
        callback.update_locals(locals())
        if callback.on_step() is False:
            return False

        callback.on_rollout_end()
        return True

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        time_elapsed = time.time() - self.start_time
        fps = int(self.num_timesteps / (time_elapsed + 1e-8))
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total timesteps", self.num_timesteps, exclude="tensorboard")

        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)

    def learn(
        self,
        total_timesteps: Optional[int] = None,
        nb_epochs: Optional[int] = None,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "CEM",
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "BaseAlgorithm":

        # nb_epochs can be overwritten by the parameter passed in the constructor
        nb_epochs = nb_epochs or self.nb_epochs

        assert (
            total_timesteps is not None or nb_epochs is not None
        ), "You must specify either a total number of time steps or a number of epochs"

        if nb_epochs is not None:
            # Upper bound
            total_steps = nb_epochs * self.max_episode_steps * self.pop_size * self.env.num_envs * self.n_eval_episodes
        else:
            total_steps = total_timesteps
            # No limit
            nb_epochs = int(1e20)

        total_steps, callback = self._setup_learn(
            total_steps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())

        epoch_idx = 0
        while self.num_timesteps < total_steps and epoch_idx < nb_epochs:  # pytype: disable=unsupported-operands
            epoch_idx += 1
            self.logger.record("time/iterations", epoch_idx, exclude="tensorboard")
            continue_training = self.learn_one_epoch(callback)
            if continue_training is False:
                break

        callback.on_training_end()
        return self
