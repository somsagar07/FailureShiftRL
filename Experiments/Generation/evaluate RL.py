import config as cfg

from PIL import Image
import os
import numpy as np
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import random
import clip
import torch
import matplotlib.pyplot as plt

action_count = [0] * (cfg.NUM_ACTIONS ** 3)

class StablediffusionEnv(gym.Env):
    def __init__(self):
        super(StablediffusionEnv, self).__init__()
        self.prompt = None
        self.current_word = None
        embedding_size = 512
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(embedding_size,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(cfg.NUM_ACTIONS ** 3)  
        self.prompt_embed = None      
        
        self.action_dict = {}
        count = 0
        for i in range(cfg.NUM_ACTIONS):
            for j in range(cfg.NUM_ACTIONS):
                for k in range(cfg.NUM_ACTIONS):
                    self.action_dict[count] = (i, j, k)
                    count += 1

    def reset(self):
        self.current_word = None
        self.prompt = cfg.get_prompt(random.randint(0, 19))
        prompt_clip_text = clip.tokenize([self.prompt]).to(cfg.DEVICE)  # Ensure the device is consistent
        self.prompt_embed = cfg.CLIP_MODEL.encode_text(prompt_clip_text)
        return self.prompt_embed.cpu().detach().numpy()  # Convert to numpy after moving to CPU

    def step(self, action):
        action_count[action] += 1
        word1, word2, word3 = self.action_dict[action]
        self.current_word = cfg.create_prompt(word1,word2,word3,self.prompt)
        print(self.current_word)

        prediction = cfg.PIPELINE([self.current_word]).images
        
        #clip embeddings
        clip_text = clip.tokenize([self.current_word]).to(cfg.DEVICE)
        
        clip_image = cfg.CLIP_PREPROCESS(prediction[0]).unsqueeze(0).to(cfg.DEVICE)
        
        with torch.no_grad():
            image_features = cfg.CLIP_MODEL.encode_image(clip_image)
            text_features = cfg.CLIP_MODEL.encode_text(clip_text)

        
        cosine_similarity = torch.nn.functional.cosine_similarity(text_features, image_features, dim=1)
        
        reward = 100*(1-cosine_similarity.item())
        self.last_images = np.array(prediction[0])
        done = True
        return np.array(prediction[0]).mean(axis=2).mean(axis=1), reward, done, {'image' : prediction[0], 'cosine_similarity' : cosine_similarity}
    
env = DummyVecEnv([lambda: StablediffusionEnv()])
dqn_model = DQN.load('./Experiments/Generation/RL models/dqn_model_sdv1_4')

proxy_prob_clip = []

for episode in range(1000):
    done = False
    obs = env.reset()
    q_values = dqn_model.policy.q_net(torch.tensor(obs, dtype=torch.float32, device = dqn_model.device))
    action_probabilities = torch.nn.functional.softmax(q_values, dim=1)
    proxy_prob_clip.append(action_probabilities)

concatenated_data = torch.cat(proxy_prob_clip, dim=0)
transposed_data = concatenated_data.t()

proxy_mean = torch.mean(transposed_data, dim=1)

action_list = {}
action_total = {}
count = 0
for i in range(5):
    for j in range(5):
        for k in range(5):
            action_list[count] = 0
            action_total[count] = 0
            count += 1
action_count = [0] * (cfg.NUM_ACTIONS ** 3)
for episode in range(500):
    print(f"Episode: {episode}")
    done = False
    obs = env.reset()
    while not done:
        action, _ = dqn_model.predict(obs)
        obs, reward, done, info = env.step(action)
        if info[0]['cosine_similarity'][0].cpu().detach().numpy() < 0.3:
            action_list[action[0]] += 1
 
        action_total[action[0]] +=1  

print(action_list)
print(action_total)

episode_rewards = []
episode_actions = []
proxy_prob = []

store_img = [[] for _ in range(cfg.NUM_ACTIONS ** 3)]
action_count = [0] * (cfg.NUM_ACTIONS ** 3)
for episode in range(500):
    print(f"Episode: {episode}")
    done = False
    obs = env.reset()
    q_values = dqn_model.policy.q_net(torch.tensor(obs, dtype=torch.float32, device = dqn_model.device))
    action_probabilities = torch.nn.functional.softmax(q_values, dim=1)
    proxy_prob.append(action_probabilities)
    while not done:
        action, _ = dqn_model.predict(obs)
        obs, reward, done, info = env.step(action)
        store_img[action[0]].append(info[0]['image'])
        episode_rewards.append(reward)
        episode_actions.append(action)

print(episode_rewards)
print(episode_actions)

list_of_lists_of_images = store_img

base_dir = 'saved_images'
os.makedirs(base_dir, exist_ok=True)

with open('image_paths.txt', 'w') as file:
    for i, sublist in enumerate(list_of_lists_of_images):
        sublist_dir = os.path.join(base_dir, f'sublist_{i}')
        os.makedirs(sublist_dir, exist_ok=True)

        for j, np_image in enumerate(sublist):
            # Convert NumPy array to PIL Image
            pil_image = Image.fromarray(np.uint8(np_image))
            image_path = os.path.join(sublist_dir, f'image_{j}.png')
            pil_image.save(image_path)

            # Write the path to the file
            file.write(image_path + '\n')

# Plotting episode rewards
plt.plot(episode_rewards, label='reward')
plt.xlabel('steps')
plt.ylabel('reward')
plt.legend()
plt.show()

unique_actions, action_counts = np.unique(np.concatenate(episode_actions), return_counts=True)
plt.bar(unique_actions, action_counts, color="green")
plt.xlabel('Action')
plt.ylabel('Frequency')
plt.title('Stable diffusion v1-4')
plt.tight_layout()
plt.show()

action_dict2 = {}
count2 = 0
for i in range(cfg.NUM_ACTIONS):
    for j in range(cfg.NUM_ACTIONS):
        for k in range(cfg.NUM_ACTIONS):
            action_dict2[count2] = (i, j, k)
            count2 += 1

concatenated_data = torch.cat(proxy_prob, dim=0)
transposed_data = concatenated_data.t()

proxy_mean = torch.mean(transposed_data, dim=1)
proxy_std = torch.std(transposed_data, dim = 1)

proxy_std_log = torch.log(proxy_std)
proxy_std_log_np = proxy_std_log.cpu().detach().numpy()
sorted_indices = np.argsort(proxy_std_log_np)
ranks = np.empty_like(sorted_indices)
ranks[sorted_indices] = np.arange(len(proxy_std_log_np))

ranks+=1

ranks_tensor = torch.tensor(ranks)

proxy_mean = proxy_mean.reshape(3,3,3).cpu().detach().numpy()
scale = (ranks_tensor.reshape(3,3,3).detach().numpy())*2

threshold = np.array(action_count).mean() - 2*np.array(action_count).std()
action_CT = np.array(action_count).reshape(3,3,3)

import plotly.graph_objs as go
import numpy as np

xs = np.arange(3)
ys = np.arange(3)
zs = np.arange(3)


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
                      hovertemplate="Action 1: %{x}<br>Action 2: %{y}<br>Action 3: %{z}<br>LogSoftQ: %{marker.color}<extra></extra>",
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
                      hovertemplate="Action 1: %{x}<br>Action 2: %{y}<br>Action 3: %{z}<br><extra></extra>",
                    )
layout = go.Layout(height=800, width=800, title='3D Heatmap',
                   scene=dict(
                              xaxis = dict(title='Action 1'),
                              yaxis = dict(title='Action 2'),
                              zaxis = dict(title='Action 3')
                             ),
                  )

fig = go.Figure(data=[trace1, trace2], layout=layout)
fig.show()