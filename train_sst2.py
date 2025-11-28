"""
Fine-tune T5-small on SST-2 Dataset

Trains T5-small to classify sentiment (positive/negative) using the SST-2 dataset.
Uses Hugging Face Trainer for efficient training.
"""

import argparse
import os
from typing import Dict
import torch
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def preprocess_function(examples, tokenizer, max_input_length=512, max_target_length=10):
    """
    Preprocess data for T5:
    Input: "sst2 sentence: {sentence}"
    Target: "positive" or "negative"
    """
    # T5 expects a task prefix
    inputs = [f"sst2 sentence: {sentence}" for sentence in examples["sentence"]]
    
    # Map labels (0/1) to text ("negative"/"positive")
    label_map = {0: "negative", 1: "positive"}
    targets = [label_map[label] for label in examples["label"]]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs, 
        max_length=max_input_length, 
        truncation=True,
        padding=False  # Padding handled by data collator
    )
    
    # Tokenize targets
    labels = tokenizer(
        targets, 
        max_length=max_target_length, 
        truncation=True,
        padding=False
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_pred, tokenizer):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Clean up text
    decoded_preds = [pred.strip().lower() for pred in decoded_preds]
    decoded_labels = [label.strip().lower() for label in decoded_labels]
    
    # Convert back to binary labels for metrics
    # Note: T5 might generate text that isn't exactly "positive" or "negative"
    # We'll map based on containment or default to negative (0)
    
    def text_to_label(text):
        if "positive" in text: return 1
        if "negative" in text: return 0
        return 0 # Default fallback
        
    pred_labels = [text_to_label(p) for p in decoded_preds]
    true_labels = [text_to_label(l) for l in decoded_labels]
    
    # Compute metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, 
        pred_labels, 
        average='binary'
    )
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main():
    parser = argparse.ArgumentParser(description="Fine-tune T5-small on SST-2")
    parser.add_argument("--output_dir", type=str, default="./t5-sst2-finetuned", help="Output directory")
    parser.add_argument("--epochs", type=float, default=3.0, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (overrides epochs if > 0)")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    model_name = "t5-small"
    print(f"Loading {model_name}...")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Load dataset
    print("Loading SST-2 dataset...")
    dataset = load_dataset("glue", "sst2")
    
    # Preprocess dataset
    print("Preprocessing dataset...")
    # Only process train and validation, ignore test (has -1 labels)
    splits_to_process = ["train", "validation"]
    tokenized_datasets = {}
    
    for split in splits_to_process:
        tokenized_datasets[split] = dataset[split].map(
            lambda x: preprocess_function(x, tokenizer),
            batched=True,
            remove_columns=dataset[split].column_names
        )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        weight_decay=0.01,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        load_best_model_at_end=True,
    )
    
    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, tokenizer),
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print(f"Saving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Evaluate
    print("Evaluating final model...")
    metrics = trainer.evaluate()
    print("Final Metrics:", metrics)

if __name__ == "__main__":
    main()
