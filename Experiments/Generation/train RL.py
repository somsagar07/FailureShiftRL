import config as cfg

import numpy as np
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import random
import clip
import torch

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
dqn_model = DQN("MlpPolicy", env, buffer_size=1000, verbose=1, exploration_final_eps=0.6, exploration_initial_eps=1.0)


dqn_model.learn(total_timesteps=3000)
dqn_model.save("./Experiments/Generation/RL models/dqn_model_sdv1_4")