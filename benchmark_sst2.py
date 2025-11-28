"""
T5-Small Benchmark on SST-2 Dataset (Local Mode)

Evaluates T5-small model on the SST-2 sentiment analysis dataset.
Computes classification metrics and performance benchmarks.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm


class T5SentimentBenchmark:
    """Benchmark T5-small on SST-2 sentiment analysis"""
    
    def __init__(self, model_name: str = "t5-small"):
        """Initialize model and tokenizer"""
        print(f"Loading model: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully\n")
    
    def load_sst2_dataset(self, split: str = "validation", num_samples: int = None) -> List[Dict]:
        """
        Load SST-2 dataset
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            num_samples: Number of samples to load (None for all)
        
        Returns:
            List of examples with 'sentence' and 'label' keys
        """
        print(f"Loading SST-2 dataset ({split} split)...")
        dataset = load_dataset("glue", "sst2", split=split)
        
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        print(f"Loaded {len(dataset)} examples\n")
        return dataset
    
    def predict_sentiment(self, text: str) -> str:
        """
        Predict sentiment for a single text
        
        Args:
            text: Input text
        
        Returns:
            Predicted sentiment ('positive' or 'negative')
        """
        # Format input for T5 (sentiment classification task)
        input_text = f"sst2 sentence: {text}"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate prediction
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs['input_ids'],
                max_length=10,  # Short output for sentiment
                num_beams=1,    # Greedy decoding for speed
                early_stopping=True
            )
        
        # Decode output
        prediction = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip().lower()
        
        # Map to positive/negative (handle various outputs)
        if "positive" in prediction or prediction == "1":
            return "positive"
        elif "negative" in prediction or prediction == "0":
            return "negative"
        else:
            # Default to positive if unclear
            return "positive" if "pos" in prediction else "negative"
    
    def evaluate(self, dataset, verbose: bool = True) -> Dict:
        """
        Evaluate model on dataset
        
        Args:
            dataset: SST-2 dataset
            verbose: Show progress bar
        
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = []
        ground_truth = []
        inference_times = []
        sample_predictions = []
        
        # Label mapping (SST-2: 0=negative, 1=positive)
        label_map = {0: "negative", 1: "positive"}
        
        print("Running evaluation...")
        iterator = tqdm(dataset, desc="Evaluating") if verbose else dataset
        
        for i, example in enumerate(iterator):
            text = example['sentence']
            true_label = label_map[example['label']]
            
            # Measure inference time
            start_time = time.time()
            pred_label = self.predict_sentiment(text)
            inference_time = time.time() - start_time
            
            predictions.append(pred_label)
            ground_truth.append(true_label)
            inference_times.append(inference_time)
            
            # Store sample predictions for analysis
            if i < 10:  # Keep first 10 examples
                sample_predictions.append({
                    'text': text,
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'correct': pred_label == true_label,
                    'inference_time': inference_time
                })
        
        # Compute metrics
        metrics = self._compute_metrics(
            predictions,
            ground_truth,
            inference_times,
            sample_predictions
        )
        
        return metrics
    
    def _compute_metrics(
        self,
        predictions: List[str],
        ground_truth: List[str],
        inference_times: List[float],
        sample_predictions: List[Dict]
    ) -> Dict:
        """Compute evaluation metrics"""
        
        # Classification metrics
        accuracy = accuracy_score(ground_truth, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            ground_truth,
            predictions,
            average='binary',
            pos_label='positive'
        )
        
        # Confusion matrix
        cm = confusion_matrix(ground_truth, predictions, labels=['negative', 'positive'])
        
        # Performance metrics
        total_time = sum(inference_times)
        avg_inference_time = total_time / len(inference_times)
        throughput = len(predictions) / total_time
        
        return {
            'classification_metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            },
            'confusion_matrix': {
                'true_negative': int(cm[0][0]),
                'false_positive': int(cm[0][1]),
                'false_negative': int(cm[1][0]),
                'true_positive': int(cm[1][1])
            },
            'performance_metrics': {
                'total_samples': len(predictions),
                'total_time_seconds': total_time,
                'avg_inference_time_ms': avg_inference_time * 1000,
                'throughput_samples_per_sec': throughput
            },
            'sample_predictions': sample_predictions
        }
    
    def print_report(self, metrics: Dict):
        """Print formatted benchmark report"""
        print("\n" + "="*70)
        print("T5-SMALL SST-2 BENCHMARK REPORT")
        print("="*70)
        
        # Classification Metrics
        cm = metrics['classification_metrics']
        print("\n📊 CLASSIFICATION METRICS")
        print("-" * 70)
        print(f"  Accuracy:  {cm['accuracy']:.4f} ({cm['accuracy']*100:.2f}%)")
        print(f"  Precision: {cm['precision']:.4f}")
        print(f"  Recall:    {cm['recall']:.4f}")
        print(f"  F1 Score:  {cm['f1_score']:.4f}")
        
        # Confusion Matrix
        conf = metrics['confusion_matrix']
        print("\n📈 CONFUSION MATRIX")
        print("-" * 70)
        print(f"                    Predicted")
        print(f"                Negative  Positive")
        print(f"  Actual Negative    {conf['true_negative']:4d}      {conf['false_positive']:4d}")
        print(f"         Positive    {conf['false_negative']:4d}      {conf['true_positive']:4d}")
        
        # Performance Metrics
        perf = metrics['performance_metrics']
        print("\n⚡ PERFORMANCE METRICS")
        print("-" * 70)
        print(f"  Total Samples:        {perf['total_samples']}")
        print(f"  Total Time:           {perf['total_time_seconds']:.2f} seconds")
        print(f"  Avg Inference Time:   {perf['avg_inference_time_ms']:.2f} ms/sample")
        print(f"  Throughput:           {perf['throughput_samples_per_sec']:.2f} samples/sec")
        
        # Sample Predictions
        print("\n🔍 SAMPLE PREDICTIONS")
        print("-" * 70)
        for i, sample in enumerate(metrics['sample_predictions'][:5], 1):
            status = "✓" if sample['correct'] else "✗"
            print(f"\n  {i}. {status} [{sample['inference_time']*1000:.1f}ms]")
            print(f"     Text: {sample['text'][:80]}...")
            print(f"     True: {sample['true_label']:8s} | Predicted: {sample['predicted_label']}")
        
        print("\n" + "="*70)
    
    def save_report(self, metrics: Dict, output_path: str):
        """Save metrics to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n💾 Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark T5-small on SST-2 dataset")
    parser.add_argument(
        '--samples',
        type=int,
        default=None,
        help='Number of samples to evaluate (default: all validation set)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='validation',
        choices=['train', 'validation', 'test'],
        help='Dataset split to use (default: validation)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='benchmark_results.json',
        help='Output file for results (default: benchmark_results.json)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='t5-small',
        help='Model name or path (default: t5-small)'
    )
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = T5SentimentBenchmark(model_name=args.model)
    
    # Load dataset
    dataset = benchmark.load_sst2_dataset(split=args.split, num_samples=args.samples)
    
    # Run evaluation
    metrics = benchmark.evaluate(dataset, verbose=True)
    
    # Print report
    benchmark.print_report(metrics)
    
    # Save results
    benchmark.save_report(metrics, args.output)


if __name__ == "__main__":
    main()
