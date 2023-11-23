import torch
from transformers import DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from transformers import LineByLineTextDataset
from datasets import load_dataset

def preprocess(dataset, tokenizer):
    lines = [line for line in dataset['text'] if (len(line) > 0 and not line.isspace())]
    batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=512)
    examples = batch_encoding["input_ids"]
    examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in examples]
    return examples

def train(model_name,
          output_dir,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
          save_steps):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir + "tokenizer")
    tokenizer.pad_token = tokenizer.eos_token

    wikitext_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    wikitext_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    train_dataset = preprocess(wikitext_train, tokenizer)
    eval_dataset = preprocess(wikitext_test, tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.save_pretrained(output_dir + "pretrained")

    training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=overwrite_output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            num_train_epochs=num_train_epochs,
            save_steps=save_steps,
        )

    trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            eval_dataset=eval_dataset,
            train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model()
    eval_results = trainer.evaluate()
    print(f"Perplexity: {torch.exp(eval_results['eval_loss'])}")

model_name = 'gpt2'
output_dir = './ckpts/'
overwrite_output_dir = False
per_device_train_batch_size = 8
num_train_epochs = 5
save_steps = 1000

train(
    model_name=model_name,
    output_dir=output_dir,
    overwrite_output_dir=overwrite_output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    num_train_epochs=num_train_epochs,
    save_steps=save_steps
)