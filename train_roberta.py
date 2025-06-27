import os
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)

import torch
import wandb
from transformers.trainer_utils import get_last_checkpoint

from sparse_roberta import get_custom_model  # Import the custom model
from alpha_scheduler import AlphaScheduler  # Import the alpha scheduler
import evaluate

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="roberta-base",
        metadata={"help": "The model checkpoint for weights initialization."}
    )
    use_triton_entmax: bool = field(
        default=False,
        metadata={"help": "Flag to use Triton's entmax implementation."}
    )
    train_from_scratch: bool = field(
        default=False,
        metadata={"help": "Flag to train from scratch."}
    )

@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default="allenai/c4", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_subset: Optional[str] = field(
        default="en", metadata={"help": "The subset of the dataset to use (e.g., 'en' for English)."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    streaming: bool = field(default=True, metadata={"help": "Enable streaming mode for large datasets."})
    max_eval_samples: Optional[int] = field(
        default=50000, metadata={"help": "Maximum number of examples to use for evaluation to save memory."}
    )

@dataclass
class CustomTrainingArguments(TrainingArguments):
    max_steps: int = field(
        default=100000,
        metadata={"help": "Set the maximum number of training steps."}
    )
    resume_training: bool = field(default=False, metadata={"help": "Flag to resume training from the last checkpoint."})
    report_to: str = field(
        default="wandb",  # Ensure WandB is used for logging
        metadata={"help": "The integrations to report the logs to."}
    )

@dataclass
class AlphaSchedulerArguments:
    initial_alpha: float = field(default=1.0000001, metadata={"help": "Starting alpha value"})
    final_alpha: float = field(default=2.0, metadata={"help": "Final alpha value"})
    max_alpha_steps: int = field(default=10000, metadata={"help": "Total steps for alpha annealing"})
    strategy: str = field(
        default="linear",
        metadata={"help": "Strategy for alpha scheduling", "choices": ["linear", "exponential", "cosine", "polynomial", "stepwise", "sigmoid"]}
    )
    power: int = field(default=2, metadata={"help": "Power for polynomial annealing"})
    step_size: int = field(default=1000, metadata={"help": "Step size for stepwise annealing"})
    increment: float = field(default=0.1, metadata={"help": "Increment for stepwise annealing"})
    k: float = field(default=0.1, metadata={"help": "Steepness factor for sigmoid annealing"})

def main():
    # Argument parsing
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments, AlphaSchedulerArguments))
    model_args, data_args, training_args, alpha_args = parser.parse_args_into_dataclasses()

    # Print arguments
    logger.info(f"Model arguments: {model_args}")
    logger.info(f"Data arguments: {data_args}")
    logger.info(f"Training arguments: {training_args}")
    logger.info(f"Alpha scheduler arguments: {alpha_args}")

    # Initialize WandB for logging
    wandb.init(project="sparse_pretraining", config={**vars(model_args), **vars(data_args), **vars(training_args), **vars(alpha_args)})
    logger.info("Initialized WandB for logging")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Load dataset splits
    train_dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_subset,
        split='train',
        streaming=data_args.streaming,
    )

    validation_dataset = load_dataset(
        #"cerebras/SlimPajama-627B",
        "allenai/c4",
        "en",
        split='validation',
        streaming=data_args.streaming,
    )

    # Limit evaluation samples if max_eval_samples is set
    if data_args.max_eval_samples is not None:
        validation_dataset = validation_dataset.take(data_args.max_eval_samples)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = get_custom_model(
        model_args.model_name_or_path,
        initial_alpha=alpha_args.initial_alpha,
        use_triton_entmax=model_args.use_triton_entmax,
        from_scratch=model_args.train_from_scratch,
    )

    # Define the tokenization function with filtering for empty texts
    def tokenize_function(examples):
        # Remove any empty or None texts before tokenizing
        examples["text"] = [text for text in examples["text"] if text is not None and text.strip() != ""]
        # return tokenizer(examples["text"], truncation=True, padding=False, return_special_tokens_mask=True)
        return tokenizer(
            examples["text"],
            truncation=True,  # Truncate sequences exceeding `max_length`
            padding="max_length",  # Pad sequences to `max_length`
            max_length=512,  # Ensure consistent sequence length
            return_attention_mask=True
        )

    # Apply tokenization with filter for empty samples
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    validation_dataset = validation_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Initialize alpha scheduler
    alpha_scheduler = AlphaScheduler(
        initial_alpha=alpha_args.initial_alpha,
        final_alpha=alpha_args.final_alpha,
        max_steps=alpha_args.max_alpha_steps,
        strategy=alpha_args.strategy,
        power=alpha_args.power,
        step_size=alpha_args.step_size,
        increment=alpha_args.increment,
        k=alpha_args.k,
    )

    # Define data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        # pad_to_multiple_of=8
    )

    # Define metric for evaluation
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = logits.argmax(dim=-1)
        labels = labels.reshape(-1)
        predictions = predictions.reshape(-1)
        mask = labels != -100
        return metric.compute(predictions=predictions[mask], references=labels[mask])

    # Custom callback to calculate and log perplexity
    class PerplexityLoggingCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics, **kwargs):
            if "eval_loss" in metrics:
                perplexity = math.exp(metrics["eval_loss"])
                metrics["perplexity"] = perplexity
                wandb.log({"perplexity": perplexity, "global_step": state.global_step})

    # Custom callback for logging token count
    class TokenCountingCallback(TrainerCallback):
        def __init__(self):
            super().__init__()
            self.total_tokens = 0

        def on_train_batch_end(self, args, state, control, **kwargs):
            # Retrieve the trainer and batch from kwargs
            batch = kwargs.get("inputs", None)
            if batch and "attention_mask" in batch:
                # Use the attention mask to count non-padding tokens
                non_padding_tokens = batch["attention_mask"].sum().item()
                self.total_tokens += non_padding_tokens
            elif batch and "input_ids" in batch:
                # Use the input IDs to count non-padding tokens
                non_padding_tokens = (batch["input_ids"] != tokenizer.pad_token_id).sum().item()
                self.total_tokens += non_padding_tokens

            # Log the total tokens so far to WandB
            wandb.log(
                {
                    "total_tokens": self.total_tokens,
                    "global_step": state.global_step,
                }
            )

        def on_train_end(self, args, state, control, **kwargs):
            # Log the final token count
            logger.info(f"Total tokens trained on: {self.total_tokens}")
            wandb.log({"total_tokens": self.total_tokens})

    class LossLoggingCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None and "loss" in logs:
                # Log the loss to WandB
                wandb.log({"training_loss": logs["loss"], "global_step": state.global_step})

    # Initialize Trainer with WandB callback for logging and validation dataset
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            PerplexityLoggingCallback(),
            TokenCountingCallback(),
            LossLoggingCallback(),
        ],
    )

    # Define a custom callback for logging sparsity and alpha
    class SparsityLoggingCallback(TrainerCallback):
        def __init__(self, model, alpha_scheduler):
            super().__init__()
            self.model = model
            self.alpha_scheduler = alpha_scheduler
            self.n_tokens = 0

        def on_step_end(self, args, state, control, **kwargs):
            # Gather and log sparsity per head
            sparsity_per_head_item = {}

            for layer_index, layer in enumerate(trainer.model.roberta.encoder.layer):
                if layer.attention.self.sparsity_per_head is not None:
                    sph = layer.attention.self.sparsity_per_head.mean(0)  # average over all batches
                    for head in range(sph.size(0)):  # iterate over all heads
                        sparsity_per_head_item[f"sparsity_l{layer_index:02d}_h{head:02d}"] = sph[head].item()
                    sparsity_per_head_item[f"sparsity_l{layer_index:02d}_avg"] = sph.mean().item()

            # Update total token count
            self.n_tokens += trainer.model.roberta.encoder.layer[0].attention.self.n_tokens.item()

            # Update alpha and log it
            old_alpha = self.alpha_scheduler.alpha
            new_alpha = self.alpha_scheduler.step()
            for layer in trainer.model.roberta.encoder.layer:
                layer.attention.self.alpha = new_alpha

            log_dict = {
                "alpha": old_alpha,
                "global_step": state.global_step,
                "n_tokens_so_far": self.n_tokens,
            }
            wandb.log({**log_dict, **sparsity_per_head_item})
    trainer.add_callback(SparsityLoggingCallback(model, alpha_scheduler))

    # Detect the last checkpoint
    checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.resume_training:
        # Look for the last checkpoint if the directory exists
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")
            checkpoint = last_checkpoint
        else:
            logger.info(f"No previous checkpoint found in {training_args.output_dir}, starting fresh.")

    # Train and save
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    wandb.finish()  # Finalize WandB session

if __name__ == "__main__":
    main()
