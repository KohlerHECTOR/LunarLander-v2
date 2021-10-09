from cem.cem import CEM
from cem.policies import MlpPolicy
from gym import make
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

vec_env = make_vec_env("LunarLander-v2", n_envs=4)
save_path = "./saved_policies/cem/"

ZOO_HYPERPARMS ={
  "policy": 'MlpPolicy',
  "pop_size": 50,
  "n_eval_episodes": 20,
  "sigma": 0.2,
  "policy_kwargs": dict(net_arch=[32])}


model = CEM( save_path = save_path, env = vec_env, **ZOO_HYPERPARMS, tensorboard_log = "./cem_lunarLand2_tensorboard")
model.learn(total_timesteps = 1e8)
vec_env.close()
