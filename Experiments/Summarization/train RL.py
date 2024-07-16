import config as cfg

import sacrebleu
from datasets import load_dataset
import random
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
import numpy as np

dataset = load_dataset("openai/summarize_from_feedback", 'axis')

def calculate_bleu_score(candidate, reference):
    """
    Calculate the BLEU score for a candidate sentence given a reference sentence.

    Args:
    candidate (str): The summarized text (candidate translation).
    reference (str): The reference text (reference translation).

    Returns:
    float: The BLEU score.
    """
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
dqn_model_nlp_t5 = DQN("MlpPolicy", env, verbose=1, exploration_final_eps=0.6, exploration_initial_eps=1.0)

dqn_model_nlp_t5.learn(total_timesteps=1000)
dqn_model_nlp_t5.save("./Experiments/Summarization/RL models/dqn_model_nlp_t5")