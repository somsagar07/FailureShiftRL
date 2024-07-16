import config as cfg
from tqdm import tqdm
import random
import clip
import torch
import matplotlib.pyplot as plt

num_actions = 3

action_list = {}
action_total = {}
action_dict = {}
count = 0
for i in range(num_actions):
    for j in range(num_actions):
        for k in range(num_actions):
            action_dict[count] = (i, j, k)
            action_list[count] = 0
            action_total[count] = 0
            count += 1

epsilon = 0.01 # Epsilon parameter
probability = [1] * 20 # Number of prompts

with tqdm(total=500, desc="Evaluating") as pbar:
    for i in range(500):
        prompt_idx = random.choices(range(20), probability, k=1)[0]
        prompt = cfg.get_prompt(prompt_idx)
        action = random.randint(0, 26)

        w1, w2, w3 = action_dict[action]
        current_word = cfg.create_prompt(w1, w2, w3, prompt)

        prediction = cfg.PIPELINE([current_word]).images

        clip_text = clip.tokenize([current_word]).to(cfg.DEVICE)
        clip_image = cfg.CLIP_PREPROCESS(prediction[0]).unsqueeze(0).to(cfg.DEVICE)

        with torch.no_grad():
            image_features = cfg.CLIP_MODEL.encode_image(clip_image)
            text_features = cfg.CLIP_MODEL.encode_text(clip_text)
        
        cosine_similarity = torch.nn.functional.cosine_similarity(image_features, text_features, dim=1)
        if cosine_similarity < 0.3:
            action_list[action] += 1
            probability[prompt_idx] += epsilon
        action_total[action] += 1
        pbar.update(1)


# Plotting the bar graph
keys1 = list(action_list.keys())
values1 = list(action_list.values())

keys2 = list(action_total.keys())
values2 = list(action_total.values())

plt.bar(keys2, values2, color='green', alpha=0.3, label='Total')
plt.bar(keys1, values1, color='green', label='Failure')

plt.xlabel('Actions')
plt.ylabel('Frequency')
plt.title('Bar graph of actions - Epsilon Greedy')
plt.legend()
plt.show()