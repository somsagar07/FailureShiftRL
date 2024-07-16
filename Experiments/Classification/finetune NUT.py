import config as cfg
import index as idx

import random
import torch
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import copy
import os
from tqdm import tqdm

action = [4, 0, 1] # [angle, darken, saturation]: Action combination to be fine-tuned

def apply_perturbations(image):
    if random.random() <= 0.5:
        current_image = cfg.rotate_image(image, cfg.ANGLE_LIST[action[0]])
        current_image = cfg.darken_image(current_image, cfg.DARKEN_LIST[action[1]])
        current_image = cfg.saturation_image(current_image, cfg.SATURATION_LIST[action[2]])
        return current_image
    else:
        return image
    
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(apply_perturbations),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                        [0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(cfg.TRAIN_FOLDER, transform=train_transforms)

train_loader = DataLoader(full_dataset, shuffle=True)
dataset_len = len(full_dataset)

def fine_tune(model, dataloaders, criterion, optimizer, num_epochs=20):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    with tqdm(total=num_epochs, desc="Fine-tuning") as pbar:
        for epoch in range(num_epochs):
            model.train()  # Set model to training mode
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders:
                inputs = inputs.to(cfg.DEVICE)
                labels = labels.to(cfg.DEVICE)

                # Zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    loss.backward()
                    optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders.dataset)
            epoch_acc = running_corrects.double() / len(dataloaders.dataset)

            # print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc * 100))

            # Deep copy the model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
        pbar.update(1)

    print('Best training Acc: {:4f}'.format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

set_parameter_requires_grad(cfg.MODEL, True)
num_ftrs = cfg.MODEL.classifier[6].in_features
cfg.MODEL.classifier[6] = torch.nn.Linear(num_ftrs, 1000)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(cfg.MODEL.parameters(), lr=0.001, momentum=0.9)

model_ft = fine_tune(cfg.MODEL.to(cfg.DEVICE), train_loader, criterion, optimizer, num_epochs=1)

model_ft.eval()
os.makedirs('saved fine tuned models', exist_ok=True)
torch.save(model_ft.state_dict(), f'saved fine tuned models/alexnet_pretrained_finetuned.pth')
print('Fine-tuning complete!')