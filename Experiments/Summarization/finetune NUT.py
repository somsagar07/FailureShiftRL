import config as cfg
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

dataset = load_dataset("openai/summarize_from_feedback", 'axis')

# Initialize the model
def model_init():
    return cfg.MODEL

# Load the tokenizer
tokenizer = cfg.TOKENIZER

def preprocess_data(examples):
    # Process articles and ensure they are strings
    # Replace the action with your failure node
    articles = [str(cfg.ACTION_LIST[8](example['article'])) if example['article'] is not None else "" for example in examples['info']]
    # Process summaries and ensure they are strings
    summaries = [str(summary['text']) if summary['text'] is not None else "" for summary in examples['summary']]

    model_inputs = tokenizer(articles, max_length=1024, padding="max_length", truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(summaries, max_length=128, padding="max_length", truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_data, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./Experiments/Summarization/results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=3
)

# Initialize the Trainer
trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=tokenized_datasets["test"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("./Experiments/Summarization/finetuned_model/model")
tokenizer.save_pretrained("./Experiments/Summarization/finetuned_model/tokenizer")