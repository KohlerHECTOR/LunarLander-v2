# LunarLander-v2
Using Stable-Baselines 3 library to train a LunarLander-v2 agent. Methods used: DQN and CEM.
Train a dqn agent on LunarLander-v2:
```
python3 train_dqn.py
```
Use Tensorboard to visualize learning:
```
tensorboard --logdir dqn_lunarLand2_tensorboard/
```
# Key plots:
# Gif of the evolution of the policy learned with DQN (reward 260 +- 30):
![](/policy_gifs/dqn_pos.gif)
# Gif of the evolution of the policy learned with CEM (reward 280 +- 18):
![](/policy_gifs/cem_pos.gif)
## Visulaizations of the policy learned by CEM (reward 280 +- 18):
![image](/report_and_figures/figures/cem/CEM_POS.png)
![image](/report_and_figures/figures/cem/CEM_ANGLE.png)
![image](/report_and_figures/figures/cem/CEM_VEL.png)
## Visulaizations of the policy learned by DQN (reward 260 +- 30):
![image](/report_and_figures/figures/dqn/DQN_POS.png)
![image](/report_and_figures/figures/dqn/DQN_ANGLE.png)
![image](/report_and_figures/figures/dqn/DQN_VEL.png)

