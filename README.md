# FailureShiftRL

We introduce a smart way to test and improve deep learning models using reinforcement learning (RL). It's designed to identify where the model performs well and where it fails by examining its action space This is useful because it helps us understand the model's strengths and weaknesses, ensuring it makes better decisions or predictions. We use this method to systematically explore different scenarios, determining how confident we can be in the model's success or failure in each. Based on this feedback, we focus on enhancing the model this could involve adjusting it to avoid certain errors or to perform better in areas that users deem important. The tasks our project tackles include imagee classification, text to image generation and text summarization.


> **Note:** In this branch, we use Jupyter Notebook (`.ipynb`) files exclusively to enhance code interactivity and better understanding.

## Structure

This project is organized into several key directories:

- `Classification/`: Contains code and models related to the classification tasks.
- `Generation/`: Contains code and models for image generation tasks.
- `Summarization/`: Contains code and models for summarization tasks.
- `Utils/`: Dataset creation and other plotting functions.
- `Dataset/`: Datasets used in different tasks.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Install with pip

```bash
pip install requirments.txt
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