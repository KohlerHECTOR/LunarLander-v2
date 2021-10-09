from gym import make
# from wrappers.timeAndSlowRenderWrapper import TimeLimitSlowRenderWrapper
# from wrappers.timeWrapper import TimeLimitWrapper
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy
from time import sleep
import numpy as np
from cem.cem import CEM
from cem.policies import MlpPolicy

# steps = np.arange(5e3,1e5+5e3,5e3)
env = make("LunarLander-v2")
model = DQN.load("./logs/withZOO/rl_model_170000_steps.zip", env=env)
x_pos = np.linspace(-1, 1, 100)
y_pos = np.linspace(0, 1.5, 200)
policy = np.zeros((200,100))
for x in range(100):
    for y in range(200):
        obs = [x_pos[x], y_pos[y], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        policy[199-y,x] = model.predict(np.array(obs))[0]
        # print(obs[:2], policy[y,x])
import matplotlib.pyplot as plt
from matplotlib import colors


data = policy

# create discrete colormap
cmap = colors.ListedColormap(['white','red', 'green', 'blue'])
bounds = [0,1,2, 3,4]
norm = colors.BoundaryNorm(bounds, cmap.N)

extent = [-1, 1, 0, 1.5]
plt.xlabel('Lander x position')
plt.ylabel('Lander y position')
img = plt.imshow(data, cmap=cmap, norm=norm, extent = extent, aspect = 2.0)

# draw gridlines
plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks = [0, 1, 2, 3], label = "Discrete actions")

plt.savefig("./renduIAR/figures/DQN_POS.png")
# plt.show()
plt.clf()




y_pos = np.linspace(0, 1.5, 100)
y_velocity = np.linspace(-1.0 , 0, 100)
policy = np.zeros((100,100))
for vel in range(100):
    for y in range(100):
        obs = [0., y_pos[y], 0.0, y_velocity[vel], 0.0, 0.0, 0.0, 0.0]
        policy[99-y,99-vel] = model.predict(np.array(obs))[0]

data = policy

# create discrete colormap
cmap = colors.ListedColormap(['white','red', 'green', 'blue'])
bounds = [0,1,2, 3,4]
norm = colors.BoundaryNorm(bounds, cmap.N)

plt.xlabel('Lander y velocity')
plt.ylabel('Lander y position')
img= plt.imshow(data, cmap=cmap, norm=norm, extent = [-1, 0, 0, 1.5], aspect = 0.8)
plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks = [0, 1, 2, 3], label = "Discrete actions")

plt.savefig("./renduIAR/figures/DQN_VEL.png")
# plt.show()
plt.clf()
angle = np.linspace(-0.5, 0.5, 100)
angle_vel = np.linspace(-0.5, 0.5, 100)
policy = np.zeros((100, 100))
for angles in range(100):
    for vel in range(100):
        obs = [0. , 0.75, 0, 0, angle[angles], angle_vel[vel], 0, 0]
        policy[99-vel, angles] = model.predict(np.array(obs))[0]


data = policy

# create discrete colormap
cmap = colors.ListedColormap(['white','red', 'green', 'blue'])
bounds = [0,1,2, 3,4]
norm = colors.BoundaryNorm(bounds, cmap.N)

plt.xlabel('Lander angle ')
plt.ylabel('Lander angle velocity')

img = plt.imshow(data, cmap=cmap, norm=norm, extent = [-0.5, 0.5, -0.5, 0.5])

plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks = [0, 1, 2, 3], label = "Discrete actions")
plt.savefig("./renduIAR/figures/DQN_ANGLE.png")
# plt.show()
plt.clf()
