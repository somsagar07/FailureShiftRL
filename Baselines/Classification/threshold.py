import config as cfg
import index as idx

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

index_dict = idx.INDEX_DICT

action_dict = {}
action_list = {}
action_total = {}
count = 0
for i in range(5):
    for j in range(5):
        for k in range(5):
            action_dict[count] = (i, j, k)
            action_list[count] = 0
            action_total[count] = 0
            count += 1

def classifier_prediction(model, img):
    image_array = np.array(img.convert('RGB'))
    pil_image = Image.fromarray(image_array.astype('uint8'))

    preprocess = cfg.TRANSFORMS
    batch = preprocess(pil_image).unsqueeze(0)
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    return class_id

max_threshold = 5 # Threshold factor
current_threshold = 0
img_class_path = cfg.select_random_class_images()

with tqdm(total=5000, desc="Evaluating") as pbar:
    for i in range(5000):
        if current_threshold == max_threshold:
            current_threshold = 0
            img_class_path = cfg.select_random_class_images()

        ground_truth = int(img_class_path[0].split('/')[-1].split('\\')[-2])
        img_path = cfg.select_random_image_frompath(img_class_path)
        action_num = cfg.select_random_action()
        action = action_dict[action_num]
        rotation_action, darkening_action, saturation_action = action
        angle = cfg.ANGLE_LIST[rotation_action]
        dark = cfg.DARKEN_LIST[darkening_action]
        saturate = cfg.SATURATION_LIST[saturation_action]
        
        img = Image.open(img_path).resize((224, 224))
        img = cfg.rotate_image(img, angle)
        img = cfg.saturation_image(img, saturate)
        img = cfg.darken_image(img, dark)
        prediction = classifier_prediction(cfg.MODEL, img)
        action_total[action_num] += 1
        if prediction != ground_truth:
            action_list[action_num] += 1
        else:
            current_threshold += 1

        pbar.update(1)


# Plotting the bar graph
keys1 = list(action_list.keys())
values1 = list(action_list.values())

keys2 = list(action_total.keys())
values2 = list(action_total.values())


plt.bar(keys2, values2, color='green', alpha=0.3, label='Total')
plt.bar(keys1, values1, color='green', label='Misclassified')

plt.xlabel('Action')
plt.ylabel('Frequency')
plt.title('Bar graph of actions - Threshold')
plt.legend()

# Show the plot
plt.show()