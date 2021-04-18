from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

def train_model(dataset):
    model_checkpoint = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    def tokenize_function(examples):
        return tokenizer(examples["password"])
    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=7, remove_columns=["password"])

    block_size = 128

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=7,
    )

    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

    training_args = TrainingArguments(
        "test-clm",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        no_cuda=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"]
    )

    trainer.train()

    import math
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    return trainer.model

