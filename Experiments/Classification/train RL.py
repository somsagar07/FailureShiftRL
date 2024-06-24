import config as cfg
import index as idx

import numpy as np
from PIL import Image
import os
import random
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

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
dqn_model_alex = DQN("MlpPolicy", env, verbose=1, exploration_final_eps=0.6, buffer_size=10000, exploration_initial_eps=1.0)

print('Starting Model Training...')
dqn_model_alex.learn(50000)
dqn_model_alex.save("./Experiments/Classification/saved RL models/dqn_model_alex_pretrained")
print('Model Trained')