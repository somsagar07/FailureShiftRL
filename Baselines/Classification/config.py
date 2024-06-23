from torchvision import models
from torchvision.transforms import v2
import os, random

def rotate_image(img, angle):
    return v2.functional.rotate(img, angle)

def darken_image(img, factor):
    return v2.functional.adjust_brightness(img, 1 - factor)

def saturation_image(img, factor):
    return v2.functional.adjust_saturation(img, factor)

ANGLE_LIST = [-10, -5, 0, 5, 10]
DARKEN_LIST = [ 0.1,0.2,0.3,0.4,0.5]
SATURATION_LIST = [0.5, 0.55, 0.6, 0.65, 0.7]

DATA_PATH = './Dataset/classification/imagenet-mini/train'
ALL_IMAGE_PATHS = [os.path.join(DATA_PATH, filename) for filename in os.listdir(DATA_PATH)]

def select_random_image_frompath(random_class_image_paths):
    random_path = random.randint(0,len(random_class_image_paths)-1)
    random_image = random_class_image_paths[random_path]
    return random_image

def select_random_class_images():
    random_class = random.randint(0,len(ALL_IMAGE_PATHS)-1)
    random_class_image_paths = [os.path.join(ALL_IMAGE_PATHS[random_class], filename) for filename in os.listdir(ALL_IMAGE_PATHS[random_class])]
    return random_class_image_paths

def select_random_action():
    return random.randint(0,124) # 0-124: Action space size (5 x 5 x 5)

def epsilon_random_class_images(probability):
    random_class = random.choices(range(len(ALL_IMAGE_PATHS)), probability, k=1)[0]
    random_class_image_paths = [os.path.join(ALL_IMAGE_PATHS[random_class], filename) for filename in os.listdir(ALL_IMAGE_PATHS[random_class])]
    return random_class_image_paths

# Update the model and transforms for your usecase
MODEL = models.alexnet(weights="IMAGENET1K_V1")
MODEL.eval()
TRANSFORMS = models.AlexNet_Weights.IMAGENET1K_V1.transforms()