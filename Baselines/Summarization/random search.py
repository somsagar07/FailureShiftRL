import config as cfg
import sacrebleu
from datasets import load_dataset
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

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

action_list = {}
action_total = {}
count = 0
for i in range(16):
    action_list[count] = 0
    action_total[count] = 0
    count += 1

with tqdm(total=500, desc="Evaluating") as pbar:
    for i in range(500):
        random_index = random.randint(0, len(dataset['test']['info']) - 1)
        current_word = dataset['test']['info'][random_index]['article'][:3400]
        action = random.randint(0,15)
        prediction = cfg.ACTION_LIST_1[action](current_word)
        score = calculate_bleu_score(current_word, cfg.get_summary(prediction))

        if score < 10:
            action_list[action] += 1
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
plt.title('Bar graph of actions - Random Search')
plt.legend()
plt.show()