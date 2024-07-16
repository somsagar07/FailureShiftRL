# Failures Are Fated, But Can Be Faded

As models become more competent across various domains, our understanding of their capabilities and limitations becomes increasingly murky. Blindly mitigating failures often leads to even more failures. We introduce a reinforcement learning (RL) based framework to characterize the failure landscape of the model in question and use this information to improve the model further. We also tested this framework across different settings, including image **classification**, text-to-image **generation**, and text **summarization**. Our results show that our framework can effectively be used to identify the model's strengths and weaknesses and provide a roadmap for further improving the model's performance.

![alt text](<./assets/image 1.jpg>)
Our framework for failure discovery and mitigation involves three steps: Discover, Summarize, and Restructure. First, deep RL is applied to explore the failure landscape and identify regions where models might fail. Second, these results are summarized and presented to the user for action. Third, based on user feedback improvement takes place. As shown with the case of image generation,, after fine tuning model generated more human aligned images.

## Project Structure

This project is organized into several key directories:

- `Additional Analysis/`: Contains code for adversarial attack (FGSM), bayesian optimization, and scalability analysis.
- `Baselines/`: Contains code for baseline analysis for each task.
- `Dataset/`: Contains dataset used in generative tasks. This folder will also be used to download and store the classification dataset.
- `Experiments/`: Contains code for training RL agent, evaluating RL agent, and finetuning *Network Under Test* (NUT) for each task. It also contains code for additional experiments performed.
- `Utils/`: Contains code to download and prepare classification dataset and other plotting functions.

## Getting Started
These instructions will get your copy of the project up and running on your local machine for development and testing purposes.

### Create a virtual environment
First, create a virtual environment using the following commands:

```bash
python3 -m venv env
```

Now, using the default command prompt (cmd), run the following command to activate the virtual environment:

```bash
.\env\Scripts\activate
```
> [!TIP]
> If you face any difficulties while creating the virtual environment, you can check out this documentation [Create a Virtual Environment with Python](https://gist.github.com/loic-nazaries/c25ce9f7b01b107573796b026522a3ad).

### Install with pip

Now, install the required packages using the following command in the activated virtual environment:

```bash
pip install -r requirements.txt
```

### Getting dataset for classifier
After installing the necessary requirements, if you need to get the dataset used for the classification task, you need to run the `dataset.py` located in the `Utils` folder. This script will download and prepare the dataset in `./Dataset/classification` folder.

> [!IMPORTANT]
> We are using kaggle to download the dataset and you need set the API key before you download the dataset. You can follow [this link](https://www.kaggle.com/docs/api) for instructions on how to create and generate your API key.

## Important Information

1. We have provided the weights of RL agents for each task in their respective folders. But we have only provided the finetuned model for image generation task available in the `./Experiments/Generation` folder.
2. We have also provided the custom dataset we created for fine tuning generative model in the `./Dataset/generation` folder. For fine tuning the generative model you can follow the instructions provided in [Using LoRA for Efficient Stable Diffusion Fine-Tuning](https://huggingface.co/blog/lora) article from huggingface.
3. Each task folder contains a `config.py` file which contains the configuration for the task. You can change the configuration as per your requirements.

## Citation
```bibtex
@inproceedings{Sagar2024icml,
author = {Sagar, Som and Taparia, Aditya and Senanayake, Ransalu},
title = {Failures Are Fated, But Can Be Faded: Characterizing and Mitigating Unwanted Behaviors in Large-Scale Vision and Language Models},
booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
year = {2024},
volume = {},
pages = {},
}
```
