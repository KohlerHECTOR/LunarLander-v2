from gym import make
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN
from cem.cem import CEM
from cem.policies import MlpPolicy
from stable_baselines3.dqn.policies import MlpPolicy
from time import sleep

policy_path = "./saved_policies/cem/LunarLander-v2#cem#elrharbifleury_kohler.zip"
env = make("LunarLander-v2")

model = CEM.load(policy_path, env=env)
# model = DQN.load(policy_path, env=env)
mean_reward, std_reward = evaluate_policy(model, Monitor(env),  n_eval_episodes=100, deterministic=True)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

### To render policy on 5 episodes ###
for _ in range(5):
    obs = env.reset()
    dones = False
    while not dones:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        sleep(0.05)
        env.render()
env.close()
