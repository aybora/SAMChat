import argparse
import os
import glob
import json
import re
import gc
from tqdm import tqdm
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ==========================================
# 1. Configuration & Argument Parsing
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM Inference and Measure Performance")
    parser.add_argument('--model', type=str, required=True, 
                        help='Path to local model or HuggingFace model ID')
    parser.add_argument('--test_folder', type=str, required=True, 
                        help='Directory containing the test images')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Directory to save the log files and results')
    parser.add_argument('--num_gpus', type=int, default=1, 
                        help='Number of GPUs to use for tensor parallelism')
    parser.add_argument('--keywords', nargs='+', default=["military", "missile", "silo"], 
                        help='List of keywords to trigger a positive detection')
    return parser.parse_args()

# ==========================================
# 2. Evaluation Logic
# ==========================================
def calculate_metrics(results, keywords):
    """
    Calculates Precision, Recall, and F1.
    Logic:
    1. If <answer>...</answer> tags exist, check ONLY that content.
    2. Otherwise, check the full text.
    3. '0'/'1' filenames = Positive Ground Truth.
    4. '2' filenames = Negative Ground Truth.
    """
    
    tp = 0  # True Positive
    fp = 0  # False Positive
    tn = 0  # True Negative
    fn = 0  # False Negative

    cat_stats = {
        '0': {'total': 0, 'correct': 0},
        '1': {'total': 0, 'correct': 0},
        '2': {'total': 0, 'correct': 0}
    }

    for item in results:
        image_name = os.path.basename(item['image'])
        raw_text = item['conversations']
        
        # --- NEW LOGIC: Extract content between <answer> tags ---
        # Using DOTALL so . matches newlines, and IGNORECASE to catch <ANSWER> etc.
        match = re.search(r'<answer>(.*?)</answer>', raw_text, re.DOTALL | re.IGNORECASE)
        
        if match:
            # Found tags: use only the content inside
            text_to_analyze = match.group(1).lower()
            used_full_text = False
        else:
            # No tags: use the whole text
            text_to_analyze = raw_text.lower()
            used_full_text = True
        # --------------------------------------------------------

        # Determine Ground Truth based on filename
        category = image_name[0]
        if category not in ['0', '1', '2']:
            continue 

        is_ground_truth_positive = category in ['0', '1']
        
        # Determine Prediction
        is_prediction_positive = any(k in text_to_analyze for k in keywords)

        # Update Category Stats
        cat_stats[category]['total'] += 1

        # Calculate Confusion Matrix
        is_correct = False

        if is_ground_truth_positive and is_prediction_positive:
            tp += 1
            is_correct = True
        elif not is_ground_truth_positive and not is_prediction_positive:
            tn += 1
            is_correct = True
        elif not is_ground_truth_positive and is_prediction_positive:
            fp += 1
        elif is_ground_truth_positive and not is_prediction_positive:
            fn += 1

        # Log detailed info for JSON output
        item['evaluation'] = {
            'ground_truth_positive': is_ground_truth_positive,
            'prediction_positive': is_prediction_positive,
            'is_correct': is_correct,
            'category': category,
            'used_full_text_fallback': used_full_text,
            'analyzed_snippet': text_to_analyze[:100] + "..." if len(text_to_analyze) > 100 else text_to_analyze
        }
        
        if is_correct:
            cat_stats[category]['correct'] += 1

    # Calculate Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    return {
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
        'Precision': precision, 'Recall': recall, 'F1': f1, 'Accuracy': accuracy,
        'Category_Stats': cat_stats
    }

# ==========================================
# 3. Main Inference Pipeline
# ==========================================
def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_slug = args.model.strip('/').split('/')[-1]
    output_json_path = os.path.join(args.output_dir, f'{model_slug}_full_log.json')
    output_txt_path = os.path.join(args.output_dir, f'{model_slug}_metrics.txt')

    print(f"--- Starting Inference with {args.model} on {args.num_gpus} GPUs ---")

    # Initialize vLLM
    model = LLM(
        model=args.model, 
        tensor_parallel_size=args.num_gpus, 
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1} 
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens if tokenizer.convert_tokens_to_ids(i) is not None]

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=32768,
        stop_token_ids=stop_token_ids
    )

    # Load Images
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    files = []
    for ext in image_extensions:
        files.extend(glob.glob(os.path.join(args.test_folder, ext)))
    files = sorted(files)
    
    if not files:
        print("No images found in the test folder!")
        return

    data = []
    print(f"Processing {len(files)} images...")
    
    for file in tqdm(files, desc='Running Inference'):
        try:
            image = Image.open(file).convert('RGB')
            width, height = image.size

            question = 'Explain this image in detail within one paragraph, using 4-6 sentences.'
            placeholder = "<|image_pad|>"
            
            prompt = (f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                      f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
                      f"{question}<|im_end|>\n"
                      f"<|im_start|>assistant\n")

            outputs = model.generate({
                "prompt": prompt,
                "multi_modal_data": {"image": image},
            }, sampling_params=sampling_params, use_tqdm=False)

            generated_text = outputs[0].outputs[0].text

            new_entry = {
                "image": file,
                "width": width,
                "height": height,
                "conversations": generated_text
            }
            data.append(new_entry)

        except Exception as e:
            print(f"Error processing {file}: {e}")

    # Clear Memory
    del model 
    gc.collect()

    # ==========================================
    # 4. Performance Calculation
    # ==========================================
    print("\n--- Calculating Performance Metrics ---")
    metrics = calculate_metrics(data, args.keywords)

    report = []
    report.append(f"Model: {args.model}")
    report.append(f"Test Folder: {args.test_folder}")
    report.append("="*50)
    report.append(f"Precision: {metrics['Precision']:.4f}")
    report.append(f"Recall:    {metrics['Recall']:.4f}")
    report.append(f"F1 Score:  {metrics['F1']:.4f}")
    report.append(f"Accuracy:  {metrics['Accuracy']:.4f}")
    report.append("-" * 30)
    report.append(f"Confusion Matrix:")
    report.append(f"TP: {metrics['TP']} | FP: {metrics['FP']} | TN: {metrics['TN']} | FN: {metrics['FN']}")
    report.append("-" * 30)
    report.append("Per-Category Accuracy:")
    for cat, stats in metrics['Category_Stats'].items():
        if stats['total'] > 0:
            acc = (stats['correct'] / stats['total']) * 100
            report.append(f"Category {cat}: {stats['correct']}/{stats['total']} ({acc:.2f}%)")
        else:
            report.append(f"Category {cat}: N/A (0 images)")

    print("\n".join(report))

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))

    print(f"\nSaved full logs to: {output_json_path}")
    print(f"Saved metrics to:   {output_txt_path}")

if __name__ == "__main__":
    main()
