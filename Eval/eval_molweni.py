import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # Add project root to sys.path

from Eval.utils.rhe_eval import tsinghua_F1

# 关系类型映射: pred -> gold
RELATION_MAPPING = {
    'CLARIF_Q': 'Clarification_question',
    'COMMENT': 'Comment',
    'CONDITIONAL': 'Conditional',
    'CONTINUATION': 'Continuation',
    'CONTRAST': 'Contrast',
    'CORRECTION': 'Correction',
    'ELABORATION': 'Elaboration',
    'EXPLANATION': 'Explanation',
    'NARRATION': 'Narration',
    'PARALLEL': 'Parallel',
    'QA_PAIR': 'QAP',
    'Q_ELABORATION': 'Q-Elab',
    'RESULT': 'Result',
}

# Load predicted data (from processed_dialogues/molweni.jsonl)
pred_data = {}
with open('/share/home/jiahui/Discourse_Analysis_across_dialogues/Data/processed_dialogues/molweni.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            item = json.loads(line)
            pred_data[item['id']] = item

# Load golden data (from Data/molweni/dialogue.json)
gold_data = {}
with open('/share/home/jiahui/Discourse_Analysis_across_dialogues/Data/molweni/dialogue.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    for item in data:
        gold_data[item['id']] = item

# Prepare data for tsinghua_F1
pred_list = []
gold_list = []
edu_num_list = []

# Match by ID
common_ids = set(pred_data.keys()) & set(gold_data.keys())
print(common_ids)
print(f"Total matched dialogues: {len(common_ids)}")

for dialog_id in common_ids:
    pred_item = pred_data[dialog_id]
    gold_item = gold_data[dialog_id]
    
    # Get EDU count
    edu_num = len(pred_item['edus'])
    edu_num_list.append(edu_num)
    
    # Pred: 从 discourse_structure 读取，并映射到 gold 类型
    pred_relations = {}
    for ds in pred_item.get('discourse_structure', []):
        if not ds:
            continue
        import re
        match = re.match(r'(\w+)\((\d+), (\d+)\)', ds)
        if match:
            rel_type = match.group(1)
            x = int(match.group(2)) - 1
            y = int(match.group(3)) - 1
            # 映射到 gold 类型
            mapped_type = RELATION_MAPPING.get(rel_type, rel_type)
            pred_relations[(x, y)] = mapped_type
    pred_list.append(pred_relations)
    
    # Gold: 从 relations 读取 (格式: {x, y, type})
    gold_relations = {}
    for rel in gold_item.get('relations', []):
        gold_relations[(rel['x'], rel['y'])] = rel['type']
    gold_list.append(gold_relations)

print("top-10 predicted relations (x, y): type:")
for i, (pred_rel, gold_rel) in enumerate(zip(pred_list, gold_list)):
    if i >= 10:
        break
    print(f"  {pred_rel}")
    print(f"  {gold_rel}")

# Run evaluation
f1_bi, f1_multi = tsinghua_F1(pred_list, gold_list, edu_num_list)

print(f"\n=== Results ===")
print(f"Binary F1: {f1_bi:.4f}")
print(f"Multi-class F1: {f1_multi:.4f}")