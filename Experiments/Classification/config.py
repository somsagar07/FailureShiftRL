import torch
from torchvision import models
from torchvision.transforms import v2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Update the model and transforms for your usecase
MODEL = models.alexnet(weights="IMAGENET1K_V1")
MODEL.eval()
WEIGHTS = models.AlexNet_Weights.IMAGENET1K_V1
TRANSFORMS = models.AlexNet_Weights.IMAGENET1K_V1.transforms()

# Update the actions for your usecases
def rotate_image(img, angle):
    return v2.functional.rotate(img, angle)

def darken_image(img, factor):
    return v2.functional.adjust_brightness(img, 1 - factor)

def saturation_image(img, factor):
    return v2.functional.adjust_saturation(img, factor)

NUM_ACTIONS = 5
ANGLE_LIST = [-10, -5, 0, 5, 10]
DARKEN_LIST = [ 0.1,0.2,0.3,0.4,0.5]
SATURATION_LIST = [0.5, 0.55, 0.6, 0.65, 0.7]

DATA_FOLDER = "./Dataset/classification/combined_images"

TRAIN_FOLDER = "./Dataset/classification/imagenet-mini/train"