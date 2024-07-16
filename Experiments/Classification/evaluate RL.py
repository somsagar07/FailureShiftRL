import config as cfg
import index as idx

from tqdm import tqdm
import torch
import numpy as np
from PIL import Image
import os
import random
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

import matplotlib.pyplot as plt
import plotly.graph_objs as go

index_dict = idx.INDEX_DICT

def classifier_prediction(model, img):
    image_array = np.array(img.convert('RGB'))
    pil_image = Image.fromarray(image_array.astype('uint8'))
    batch = cfg.TRANSFORMS(pil_image).unsqueeze(0)
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    return class_id

def get_reward(model, img , category_name, label):
    image_array = np.array(img.convert('RGB'))
    pil_image = Image.fromarray(image_array.astype('uint8'))
    batch = cfg.TRANSFORMS(pil_image).unsqueeze(0)
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    lab = label.split('_')[-3][7:]
    # print(f'catogory : {category_name} , label : {index_dict[int(lab)]}')
    if category_name not in index_dict[int(lab)]:
        reward = abs(np.log(score / (1 - score + 0.0000001)))*10
    else:
        reward = -1
    return reward

def most_frequent(List, n):
    counts = dict()
    for action in List:
        counts[action] = counts.get(action, 0) + 1
    frequent_actions = sorted(counts, key=counts.get, reverse=True)[:n]

    return frequent_actions

# Dataset
all_image_paths = [os.path.join(cfg.DATA_FOLDER, img) for img in os.listdir(cfg.DATA_FOLDER)]
random.shuffle(all_image_paths)
train_split = 0.9
train_size = int(train_split * len(all_image_paths))
train_image_paths = all_image_paths[:train_size]
test_image_paths = all_image_paths[train_size:]

action_taken = []
rewards_learn = []
action_count = [0] * (cfg.NUM_ACTIONS ** 3)
store_img = [[] for _ in range(cfg.NUM_ACTIONS ** 3)]
store_img_reward = [[] for _ in range(cfg.NUM_ACTIONS ** 3)]


# Define the environment
class ClassifierEnv(gym.Env):
    def __init__(self, classifier_model, image_paths):
        super(ClassifierEnv, self).__init__()
        self.classifier_model = classifier_model
        self.image_paths = image_paths
        self.current_image = None
        self.current_label = None
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(224, 224, 3), dtype=np.float32)
        self.action_space = gym.spaces.Discrete((cfg.NUM_ACTIONS)**3)
        self.reward_model = classifier_model
        self.current_angle = None

        self.action_dict = {}
        count = 0
        for i in range(cfg.NUM_ACTIONS):
            for j in range(cfg.NUM_ACTIONS):
                for k in range(cfg.NUM_ACTIONS):
                    self.action_dict[count] = (i, j, k)
                    count += 1

    def reset(self):
        self.current_image_path = random.choice(self.image_paths)
        self.current_image = Image.open(self.current_image_path).convert('RGB').resize((224, 224))
        return self.current_image

    def step(self, action):
        rotation_action, darkening_action, saturation_action = self.action_dict[action]

        action_count[action] += 1
        angle = cfg.ANGLE_LIST[rotation_action]
        dark = cfg.DARKEN_LIST[darkening_action]
        saturate = cfg.SATURATION_LIST[saturation_action]

        self.current_image = cfg.rotate_image(self.current_image, angle)
        self.current_image = cfg.saturation_image(self.current_image, saturate)
        self.current_image = cfg.darken_image(self.current_image, dark)

        prediction = classifier_prediction(self.classifier_model,self.current_image)

        category_name = cfg.WEIGHTS.meta["categories"][prediction]

        reward = get_reward(self.classifier_model, self.current_image , category_name, self.current_image_path )
        # print('Reward', {reward})

        if reward != -1:
          store_img[action].append(self.current_image)
          store_img_reward[action].append(reward)

        rewards_learn.append(reward)
        action_taken.append((angle,dark,saturate))

        done = True
        return prediction, reward, done, {'current_image': self.current_image}
    
env = DummyVecEnv([lambda: ClassifierEnv(cfg.MODEL, train_image_paths)])

# Load your model
dqn_model_alex = DQN.load("./Experiments/Classification/saved RL models/dqn_model_alex_pretrained", env=env, buffer_size=1000)

# Evaluate the model

episode_rewards = []
episode_actions = []
fault_images = [[] for _ in range(cfg.NUM_ACTIONS ** 3)]
faults = []
proxy_prob = []
correct_actions = []
count = 0

with tqdm(total=5000, desc="Evaluating") as pbar:
    for episode in range(5000):
        total_reward = 0
        done = False
        obs = env.reset()
        q_values = dqn_model_alex.policy.q_net(torch.tensor(obs, dtype=torch.float32, device=dqn_model_alex.device))
        action_probabilities = torch.nn.functional.softmax(q_values, dim=1)
        proxy_prob.append(action_probabilities)
        while not done:
            action, _ = dqn_model_alex.predict(obs)
            episode_actions.append(action)
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)
            
            if reward != -1:
                count+=1
                correct_action = 1
                fault_images[action[0]].append(info[0]['current_image'])
            else:
                correct_action = 0
            correct_actions.append(correct_action)
            faults.append(count)
        pbar.update(1)

e_actions = [float(item[0]) for item in episode_actions]
e_rewards = [float(item[0]) for item in episode_rewards]

# correct_actions
assert len(e_actions) == len(correct_actions)

# Count correct actions
correct_counts = {}
for action, correct in zip(e_actions, correct_actions):
    if correct == 1:
        if action in correct_counts:
            correct_counts[action] += 1
        else:
            correct_counts[action] = 1

# Fill in zeros for actions that were never correct
for action in range(1, 126):
    if action not in correct_counts:
        correct_counts[action] = 0

N = 10
smoothed_rewards = np.convolve(np.squeeze(episode_rewards), np.ones(N)/N, mode='valid')

sorted_actions = sorted(correct_counts.keys())
sorted_counts = [correct_counts[action] for action in sorted_actions]

fig, axs = plt.subplots(1, 3, figsize=(20, 5))

# Plotting count
axs[0].plot(faults, color='red')
axs[0].set_xlabel('Steps')
axs[0].set_ylabel('Faults')
axs[0].set_title('Faults Over Steps')
axs[0].legend(['Fault'])

# Plotting reward
axs[1].plot(smoothed_rewards, color='dodgerblue')
axs[1].plot(episode_rewards, color='dodgerblue', alpha=0.3)
axs[1].set_xlabel('Steps')
axs[1].set_ylabel('Reward')
axs[1].set_title('Reward Over Steps')
axs[1].legend(['Smoothed Reward', 'Actual Reward'])

# Plotting bar graph of actions
unique_actions, action_counts = np.unique(np.concatenate(episode_actions), return_counts=True)
axs[2].bar(sorted_actions, sorted_counts, color='green')
axs[2].bar(unique_actions, action_counts, color='green', alpha=0.3)
axs[2].set_xlabel('Action')
axs[2].set_ylabel('Frequency')
axs[2].set_title('Bar Graph of Actions')
axs[2].legend(['Correct Actions', 'Total Actions'])


fig.suptitle('AlexNet Pretrained', fontsize=16)
plt.tight_layout()
plt.show()

action_dict = {}
count = 0
for i in range(cfg.NUM_ACTIONS):
    for j in range(cfg.NUM_ACTIONS):
        for k in range(cfg.NUM_ACTIONS):
            action_dict[count] = (i, j, k)
            count += 1

top3_frequent_actions = most_frequent(e_actions, 3)
top3_frequent_action_values = [action_dict[action] for action in top3_frequent_actions]

print("Top 3 Most Frequent Actions:", top3_frequent_action_values)




concatenated_data = torch.cat(proxy_prob, dim=0)
transposed_data = concatenated_data.t()

# Calculate the mean and standard deviation for each index
proxy_mean = torch.mean(transposed_data, dim=1)
proxy_std = torch.std(transposed_data, dim=1)

proxy_std_log = 1*torch.log(proxy_std).cpu()

proxy_std_log_np = proxy_std_log.detach().numpy()
sorted_indices = np.argsort(proxy_std_log_np)
ranks = np.empty_like(sorted_indices)
ranks[sorted_indices] = np.arange(len(proxy_std_log_np))


ranks += 1
ranks_tensor = torch.tensor(ranks).cpu()

proxy_mean = proxy_mean.reshape(5, 5, 5).detach().cpu().numpy()
scale = (ranks_tensor.reshape(5, 5, 5).detach().cpu().numpy())/2

threshold  = np.array(action_count).mean() - 2*np.array(action_count).std()
action_CT = np.array(action_count).reshape(5,5,5)

xs = np.arange(5)
ys = np.arange(5)
zs = np.arange(5)


uncertain_threshold = np.argwhere(action_CT <= threshold)
certain_threshold = np.argwhere(action_CT > threshold)

# Certain points
trace1 = go.Scatter3d(x=xs[certain_threshold[:, 0]],
                      y=ys[certain_threshold[:, 1]],
                      z=zs[certain_threshold[:, 2]],
                      mode='markers',
                      name='Certain Area',
                      marker=dict(
                          size=scale[certain_threshold[:, 0],
                                           certain_threshold[:, 1],
                                           certain_threshold[:, 2]],  # directly use rank_tensor as size
                          symbol='circle',
                          color=np.log(proxy_mean[certain_threshold[:, 0],
                                                            certain_threshold[:, 1],
                                                            certain_threshold[:, 2]]), # apply logarithm on color
                          colorbar=dict(thickness=10, ticklen=4, x=1),
                          colorscale='Viridis',   # choose a colorscale
                          opacity=0.8,
                          line=dict(color='Black', width=1)
                        ),
                      hovertemplate="Rotation: %{x}<br>Darken: %{y}<br>Saturation: %{z}<br>LogSoftQ: %{marker.color}<extra></extra>",
                    )
# Uncertain points
trace2 = go.Scatter3d(x=xs[uncertain_threshold[:, 0]],
                      y=ys[uncertain_threshold[:, 1]],
                      z=zs[uncertain_threshold[:, 2]],
                      mode='markers',
                      name='Uncertain Area',
                      marker=dict(
                          size=10,
                          symbol='square',
                          color='rgb(255,0,0)',                # set color to red
                          opacity=0.8,
                          line=dict(color='Black', width=1)
                        ),
                      showlegend=True,
                      hovertemplate="Rotation: %{x}<br>Darken: %{y}<br>Saturation: %{z}<br><extra></extra>",
                    )
layout = go.Layout(height=800, width=800, title='3D Heatmap',
                   scene=dict(
                              xaxis = dict(title='Rotation'),
                              yaxis = dict(title='Darken'),
                              zaxis = dict(title='Saturation')
                             ),
                  )



fig = go.Figure(data=[trace1, trace2], layout=layout)
fig.show()


print('Proxy Mean:', proxy_mean)
print('Proxy Std:', proxy_std)
print('Episode Actions:', e_actions)

# Make a directory to store the images
os.makedirs('fault_images_alenxet', exist_ok=True)

# Save the images
for k, v in action_dict.items():
    # Make a directory for each action
    dir_name = 'Rotation_' + str(v[0]) + '_Darken_' + str(v[1]) + '_Saturation_' + str(v[2])
    if not os.path.exists('fault_images_alexnet/' + dir_name):
        os.makedirs('fault_images_alexnet/' + dir_name)
    
    # Save the images
    for i, img in enumerate(fault_images[k]):
        img.save('fault_images_alexnet/' + dir_name + '/' + str(i) + '.jpg')