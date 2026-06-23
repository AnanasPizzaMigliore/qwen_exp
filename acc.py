import json
import os

# --- CONFIGURATION ---
GT_PATH = "/home/penghao/qwen/Products-Real/evaluation/annotations.json"
PRED_PATH = "/home/penghao/qwen/qwen_output/all_expirations.json"
#PRED_PATH = "/home/penghao/qwen/glm_output/glm_all_expirations.json"

def calculate_accuracy(gt_file, pred_file):
    if not os.path.exists(pred_file):
        print(f"Error: Prediction file not found at {pred_file}")
        return

    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)

    correct = 0
    total = 0
    
    print(f"{'Filename':<20} | {'GT (Exp)':<12} | {'Pred':<12} | {'Match'}")
    print("-" * 65)

    # We iterate based on GT keys to ensure we evaluate everything required
    for img_name in sorted(gt_data.keys()):
        total += 1
        gt_info = gt_data[img_name]
        
        # 1. Extract the ground truth 'exp' transcription
        gt_exp_date = None
        if "ann" in gt_info:
            for annotation in gt_info["ann"]:
                if annotation.get("cls") == "exp":
                    gt_exp_date = annotation.get("transcription")
                    break
        
        # 2. Get the prediction
        pred_item = pred_data.get(img_name, {})
        pred_date = pred_item.get("date") if isinstance(pred_item, dict) else None

        # 3. Normalize strings for comparison (remove / . - and spaces)
        def normalize(val):
            if val is None: return ""
            return str(val).replace("/", "").replace(".", "").replace("-", "").strip()

        is_match = normalize(gt_exp_date) == normalize(pred_date) and gt_exp_date is not None
        
        if is_match:
            correct += 1
            status = "✅"
        else:
            status = "❌"

            print(f"{img_name:<20} | {str(gt_exp_date):<12} | {str(pred_date):<12} | {status}")

    # Results
    acc = (correct / total * 100) if total > 0 else 0
    print("-" * 65)
    print(f"Total Images: {total}")
    print(f"Correct:      {correct}")
    print(f"Accuracy:     {acc:.2f}%")

if __name__ == "__main__":
    calculate_accuracy(GT_PATH, PRED_PATH)