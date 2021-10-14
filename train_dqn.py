from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy
from gym import make
# from wrappers.timeWrapper import TimeLimitWrapper
from stable_baselines3.common.callbacks import CheckpointCallback


env = make("LunarLander-v2")

DEFAULT_HYPERPARAMS = {
"policy": "MlpPolicy",
"env": env,
# "callback": checkpoint_callback,
"tensorboard_log":"./dqn_lunarLand2_tensorboard/"
}

ZOO_HYPERPARAMS = {
  'learning_rate': 6.3e-4,
  'batch_size': 128,
  'buffer_size': 50000,
  'learning_starts': 0,
  'gamma': 0.99,
  'target_update_interval': 1,
  'train_freq': 4,
  'gradient_steps': -1,
  'exploration_fraction': 0.12,
  'exploration_final_eps': 0.1,
  'policy_kwargs': dict(net_arch=[256, 256])
}

kwargs = DEFAULT_HYPERPARAMS.copy()
kwargs.update(ZOO_HYPERPARAMS)

# Create the RL model
model = DQN(**kwargs)


learning_budget = 1e6
checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='./saved_policies/dqn/',
 name_prefix='dqn_pol')


model.learn(total_timesteps= learning_budget, callback = checkpoint_callback)
env.close()
