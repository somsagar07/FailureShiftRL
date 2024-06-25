# Failures Are Fated, But Can Be Faded

As models become more competent across various domains, our understanding of their capabilities and limitations becomes increasingly murky. Blindly mitigating failures often leads to even more failures. We introduce a reinforcement learning (RL) based framework to characterize the failure landscape of the model in question and use this information to improve the model further. We also tested this framework across different settings, including image **classification**, text-to-image **generation**, and text **summarization**. Our results show that our framework can effectively be used to identify the model's strengths and weaknesses and provide a roadmap for further improving the model's performance.

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
> [!NOTE]
> If you face any difficulties while creating the virtual environment, please refer to this documentation [Create a Virtual Environment with Python](https://gist.github.com/loic-nazaries/c25ce9f7b01b107573796b026522a3ad).

### Install with pip

Now, install the required packages using the following command in the activated virtual environment:

```bash
pip install -r requirments.txt
```

### Getting dataset for classifier

After installing the necessary requirements, you need to perform the following steps to prepare your dataset:

1. Navigate to the `utils` folder and open the `dataset.ipynb` file. This Jupyter notebook contains instructions for downloading the classification dataset from Kaggle.
2. To download the dataset, you must have a Kaggle API key. Follow [this link](https://www.kaggle.com/docs/api) for instructions on how to create and generate your API key.
3. After downloading the zip file, extract it into the `Dataset` folder. 
4. Then, execute the remaining code cells in the `dataset.ipynb` file as instructed. These steps will create additional necessary files within the `Dataset` folder.

## Note

1. We have provided the weights of RL agents for each task in their respective folders. But we have only provided the finetuned model for image generation task available in the `generation` folder.
2. We have also provided the custom dataset we created for fine tuning generative model in the `Dataset/lora` folder.
3. All the pretrained models are downloaded from the popular repositories and are used for the purpose of fine tuning and testing.
4. The code for all the visualizations is provided in the `utils` folder and they can easily be used by passing the required values obtained from the RL agents.

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