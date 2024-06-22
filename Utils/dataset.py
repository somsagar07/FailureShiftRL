# Download ImageNet 1k dataset
import json, os, shutil, kaggle

dataset_dir = './Dataset/classification'
os.makedirs(dataset_dir, exist_ok=True)

kaggle.api.authenticate()
kaggle.api.dataset_download_files('ifigotin/imagenetmini-1000', path='./Dataset/classification', unzip=True, quiet=False)

with open("./Utils/in_cls_idx.json", "r") as f:
    imagenet_id_to_name = {label: int(cls_id) for cls_id, (label, name) in json.load(f).items()}

mapping = dict(sorted(imagenet_id_to_name.items()))

directory_path = f'{dataset_dir}/imagenet-mini/train'

for old_folder_name in os.listdir(directory_path):
    old_file_name = old_folder_name
    new_file_name = mapping.get(old_folder_name)

    if new_file_name < 10:
        new_file_name = f"00{new_file_name}"
    elif new_file_name < 100:
        new_file_name = f"0{new_file_name}"
    else:
        new_file_name = f"{new_file_name}"

    if new_file_name is not None:
        old_file_path = directory_path + '/' + str(old_file_name)
        new_file_path = directory_path + '/' + str(new_file_name)

        if not os.path.exists(new_file_path):
            os.rename(old_file_path, new_file_path)
        else:
            print(f"File '{new_file_name}' already exists. Skipping renaming.")
    else:
        print(f"No mapping found for folder '{old_folder_name}'. Skipping renaming.")

print("Folder names replacement in the 'train' directory is complete.")


source_dir = f'{dataset_dir}/imagenet-mini/train'
destination_dir = './Dataset/combined_images'

print("Copying images from 'train' directory to 'combined_images' directory...")

if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

for root, dirs, files in os.walk(source_dir):
    for filename in files:
        source_file = os.path.join(root, filename)
        parent_folder = os.path.basename(root)
        new_filename = f"{parent_folder}_{filename}"
        destination_file = os.path.join(destination_dir, new_filename)
        shutil.copy(source_file, destination_file)

print("ImageNet 1k dataset has been downloaded and extracted.")