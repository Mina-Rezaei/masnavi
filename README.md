# Masnavi Manavi Text Generation Project

This repository contains the code and steps for training a text generation model to generate the completion of any prompts from **Masnavi Manavi** verses by **Rumi** using **Hugging Face's GPT-2 model**.

## Project Overview
The goal of this project is to fine-tune the **GPT-2** language model on approximately **25,000 verses** from Rumi's **Masnavi Manavi** to generate coherent verse completions.

### Dataset
The dataset used contains **25,664 verses** from the Masnavi Manavi collection.
- Data file: `masnavi.csv`
- Structure:
  - `prompt`: Initial lines of the verse
  - `completion`: Corresponding completion of the verse

## Tools and Libraries Used
- **Hugging Face Transformers**
- **PyTorch**
- **W&B (Weights & Biases)** for experiment tracking
- **Kaggle Free Tier Notebook** for initial prototyping
- **Lambda Labs** GPU instance for model training

## Model Architecture
- Base Model: `gpt2`
- Max Sequence Length: **128 tokens**
- Learning Rate: **5e-4**
- Optimizer: **AdamW**
- Scheduler: **Linear Warmup**
- Epochs: **10**
- Batch Size: **128** (training) / **4** (validation)

## Installation
To set up the project, install the following dependencies:

```bash
pip install 'transformers[torch]' datasets wandb
```

## Dataset Preparation
The dataset is loaded into a **Pandas DataFrame** and tokenized with the **GPT-2 Tokenizer**.

```python
import pandas as pd
from transformers import GPT2Tokenizer

df = pd.read_csv('./masnavi.csv')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
```

### Custom Dataset Class
A custom **PyTorch Dataset** was implemented to prepare the data for training:

```python
class GPT2Dataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.input_ids = []
        self.attn_masks = []
        for _, row in df.iterrows():
            encodings = tokenizer('<|startoftext|>'+row['prompt']+'<|tab|>'+row['completion']+'<|endoftext|>',
                                  truncation=True, max_length=max_length, padding='max_length')
            self.input_ids.append(torch.tensor(encodings['input_ids']))
            self.attn_masks.append(torch.tensor(encodings['attention_mask']))
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]
```

## Training
The training loop uses **W&B** for logging and tracks the following metrics:
- Training Loss
- Validation Loss

```python
wandb.init(project="masnavi_gpt2")
for epoch in range(epochs):
    model.train()
    for step, batch in enumerate(train_dataloader):
        input_ids, attention_masks = batch
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        outputs = model(input_ids, attention_mask=attention_masks, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        wandb.log({"train_loss": loss.item()})
```

## Results
Training Results:
- Training Loss: ~**2.1** (after 10 epochs)
- Validation Loss: ~**2.4**

  <img width="742" alt="Screenshot 2025-03-08 at 8 09 54 AM" src="https://github.com/user-attachments/assets/dd13ab70-0946-4e10-9099-1db0e07480bd" />

W&B Dashboard: [View Project Logs](https://wandb.ai/masnavi_gpt2)

## Inference
To generate text completions:

```python
prompt = "این جهان چون کوه تصویر و صداست"
input_ids = tokenizer('<|startoftext|>' + prompt, return_tensors='pt').input_ids.to(device)
output = model.generate(input_ids, max_length=128, temperature=0.8, top_k=50)
print(tokenizer.decode(output[0]))
```

## Future Improvements
- Fine-tuning larger models (GPT-2 Medium, GPT-2 Large)
- Implementing **Beam Search** for better generations
- Hyperparameter tuning
- Model Deployment using **Gradio**

## Acknowledgements
Special thanks to:
- **Hugging Face Community**
- **Lambda Labs** for providing free GPUs
- **Weights & Biases** for experiment tracking

## License
This project is licensed under the **MIT License**.
