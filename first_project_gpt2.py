# !pip install transformers datasets torch

from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
import torch

# 1. Custom dataset (AI/ML/Robotics-themed sentences)
texts = [
    "Artificial intelligence is transforming industries worldwide.",
    "Robots are now capable of performing complex surgical procedures.",
    "Machine learning models learn patterns from historical data.",
    "Ethical considerations in AI development are crucial for the future.",
    "The integration of AI in education is creating personalized learning.",
    "Generative AI can produce art, music, and human-like conversations.",
    "AI-powered robots are revolutionizing manufacturing and logistics.",
    "Training deep learning models requires powerful GPUs and large datasets.",
    "Explainable AI helps users understand model predictions.",
    "Human-AI collaboration will shape the next era of innovation."
]

# 2. Create Hugging Face Dataset object
dataset = Dataset.from_dict({"text": texts})

# 3. Load tokenizer and model (GPT-2)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 4. Fix padding issue
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# 5. Tokenize and prepare the dataset
def tokenize_function(examples):
    encoding = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=50,
        return_tensors="pt"
    )
    encoding["labels"] = encoding["input_ids"].clone()
    return {k: v.squeeze() for k, v in encoding.items()}

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# 6. Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    logging_dir='./logs',
    no_cuda=False  # Set True if you don’t have GPU
)

# 7. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# 8. Train the model
trainer.train()

# 9. Take input from user
user_prompt = input("Enter your prompt: ")
input_ids = tokenizer.encode(user_prompt, return_tensors='pt')

# 10. Generate text
output = model.generate(
    input_ids,
    max_length=200,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id
)

# 11. Show output
print("\nGenerated Text:\n")
print(tokenizer.decode(output[0], skip_special_tokens=True))
