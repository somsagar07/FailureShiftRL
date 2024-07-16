import config as cfg

import sacrebleu
from datasets import load_dataset
import random
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
import numpy as np
import matplotlib.pyplot as plt

dataset = load_dataset("openai/summarize_from_feedback", 'axis')

def calculate_bleu_score(candidate, reference):
    bleu = sacrebleu.corpus_bleu([candidate], [[reference]])
    return bleu.score

class T5Embedder:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = cfg.TOKENIZER
        self.model = cfg.MODEL.to(self.device)
        self.target_dim = 1024

    def get_embedding(self, text):
        # Tokenize with truncation
        task_prefix = "summarize: "  # or any other task-specific prefix
        inputs = self.tokenizer(task_prefix + text, return_tensors='pt', max_length=1024, truncation=True).to(self.device)
        
        # Prepare decoder_input_ids
        decoder_input_ids = torch.full_like(inputs['input_ids'], self.model.config.decoder_start_token_id)
        with torch.no_grad():
            outputs = self.model(**inputs, decoder_input_ids=decoder_input_ids)
        embeddings = outputs.encoder_last_hidden_state
        embeddings = torch.mean(embeddings, dim=1)

        # Adjust the embedding to have the target dimension
        embedding_dim = embeddings.shape[1]
        if embedding_dim > self.target_dim:
            embeddings = embeddings[:, :self.target_dim]
        elif embedding_dim < self.target_dim:
            padding = torch.zeros((embeddings.shape[0], self.target_dim - embedding_dim)).to(self.device)
            embeddings = torch.cat([embeddings, padding], dim=1)

        return embeddings.cpu().numpy()


class NLPEnv(gym.Env):
    def __init__(self):
        super(NLPEnv, self).__init__()
        self.dataset = dataset
        self.current_word = None
        self.ground_truth = None
        self.embedder = T5Embedder()
        embedding_dim = 1024
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(embedding_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(16)


    def reset(self):
        random_index = random.randint(0, len(self.dataset['test']['info']) - 1)
        self.current_word = self.dataset['test']['info'][random_index]['article'][:3400]
        self.ground_truth = self.dataset['test']['summary'][random_index]['text']
        return self.embedder.get_embedding(self.current_word)

    def step(self, action):

        prediction = cfg.ACTION_LIST[action](self.current_word)
        bleu_score = calculate_bleu_score(prediction, cfg.get_summary(prediction))

        reward = bleu_score*(len(prediction) - len(self.current_word))

        print(f'Action: {cfg.ACTION_LIST_NAME[action]} Word length: {len(self.current_word)} Prediction length: {len(prediction)} Reward: {reward}')
        done = True

        return self.embedder.get_embedding(prediction), reward, done, {'current_word': self.current_word}

env = DummyVecEnv([lambda: NLPEnv()])
dqn_model_nlp_t5 = DQN.load("./Experiments/Summarization/RL models/dqn_model_nlp_t5")

episode_rewards = []
episode_actions = []
proxy_prob = []

for episode in range(500):
    print(f"Episode: {episode}")
    done = False
    obs = env.reset()
    q_values = dqn_model_nlp_t5.policy.q_net(torch.tensor(obs, dtype=torch.float32, device = dqn_model_nlp_t5.device))
    action_probabilities = torch.nn.functional.softmax(q_values, dim=1)
    proxy_prob.append(action_probabilities)
    while not done:
        action, _ = dqn_model_nlp_t5.predict(obs)
        obs, reward, done, info = env.step(action)
        print(f'Actions : {action}, Reward: {reward}')
        episode_rewards.append(reward)
        episode_actions.append(action)

e_actions = [float(item[0]) for item in episode_actions]
e_rewards = [float(item[0]) for item in episode_rewards]


N = 5
smoothed_rewards = np.convolve(np.squeeze(episode_rewards), np.ones(N)/N, mode='valid')


fig, axs = plt.subplots(1, 2, figsize=(20, 5))

# Plotting reward
axs[0].plot(smoothed_rewards, color='dodgerblue')
axs[0].plot(episode_rewards, color='dodgerblue', alpha=0.3)
axs[0].set_xlabel('Steps')
axs[0].set_ylabel('Reward')
axs[0].set_title('Reward Over Steps')
axs[0].legend(['Smoothed Reward', 'Actual Reward'])

# Plotting bar graph of actions
unique_actions, action_counts = np.unique(np.concatenate(episode_actions), return_counts=True)
plt.bar(unique_actions, action_counts, color='green')
axs[1].set_xlabel('Action')
axs[1].set_ylabel('Frequency')
axs[1].set_title('Bar Graph of Actions')
axs[1].legend(['Actions'])


fig.suptitle('T5 Pretrained', fontsize=16)
plt.tight_layout()
plt.show()

print(episode_rewards)
print(episode_actions)

# Convert the list of tensors to a single tensor
data_tensor = torch.stack(proxy_prob)

# Calculate the mean and standard deviation for each index
means = torch.mean(data_tensor, dim=0)
std_devs = torch.std(data_tensor, dim=0)

means_np = means.cpu().detach().numpy()
std_devs_np = std_devs.cpu().detach().numpy()


# Data for plotting
indices = range(0, 16)

# Creating error bar plot
plt.figure(figsize=(10, 6))
plt.errorbar(indices, means_np[0], yerr=std_devs_np * 10, fmt='o', ecolor='red', capsize=5)
plt.title('Error Bar Plot of Means and Standard Deviations')
plt.xlabel('Index')
plt.ylabel('Values')
plt.xticks(indices)
plt.grid(True)
plt.show()