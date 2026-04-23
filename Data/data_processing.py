import json
import random
import os

def process_doc2dial():
    with open('/share/home/jiahui/Discourse_Analysis_across_dialogues/Raw_Data/doc2dial/doc2dial_dial_train.json', 'r') as f:
        data = json.load(f)
        save_data = []
        for k, v in data["dial_data"].items():
            for dial_type in data["dial_data"][k]:
                for dial in data["dial_data"][k][dial_type]:
                    sample = {}
                    sample["id"] = dial["dial_id"]
                    sample['edus'] = []
                    for turn in dial["turns"]:
                        sample['edus'].append({"text": turn["utterance"], "speaker": turn["role"]})
                    save_data.append(sample)

    random.seed(42)
    random.shuffle(save_data)
    save_data = save_data[:100]

    with open('/share/home/jiahui/Discourse_Analysis_across_dialogues/Data/doc2dial/dialogue.json', 'w') as f:
        json.dump(save_data, f, indent=4)

def process_multiwoz():
    path = "/share/home/jiahui/Discourse_Analysis_across_dialogues/Raw_Data/multiwoz/train"
    for file in os.listdir(path):
        with open(os.path.join(path, file), 'r') as f:
            data = json.load(f)
            save_data = []
            for dial in data:
                sample = {}
                sample["id"] = dial["dialogue_id"]
                sample['edus'] = []
                for turn in dial["turns"]:
                    sample['edus'].append({"text": turn["utterance"], "speaker": turn["speaker"]})
                save_data.append(sample)


    random.seed(42)
    random.shuffle(save_data)
    save_data = save_data[:100]

    with open('/share/home/jiahui/Discourse_Analysis_across_dialogues/Data/multiwoz/dialogue.json', 'w') as f:
        json.dump(save_data, f, indent=4)

def process_molweni():
    with open('/share/home/jiahui/Discourse_Analysis_across_dialogues/Raw_Data/molweni/train.json', 'r') as f:
        data = json.load(f)

    random.seed(42)
    random.shuffle(data)
    data = data[:100]

    with open('/share/home/jiahui/Discourse_Analysis_across_dialogues/Data/molweni/dialogue.json', 'w') as f:
        json.dump(data, f, indent=4)

def process_topicalchat():
    save_data = []
    with open('/share/home/jiahui/Discourse_Analysis_across_dialogues/Raw_Data/topical-chat/train.json', 'r') as f:
        data = json.load(f)
        for dialog_id, item in data.items():
            sample = {}
            sample["id"] = dialog_id
            sample['edus'] = []
            for turn in item["content"]:
                sample['edus'].append({"text": turn["message"], "speaker": turn["agent"]})
            save_data.append(sample)

    random.seed(42)
    random.shuffle(save_data)
    save_data = save_data[:100]

    with open('/share/home/jiahui/Discourse_Analysis_across_dialogues/Data/topical-chat/dialogue.json', 'w') as f:
        json.dump(save_data, f, indent=4)

if __name__ == "__main__":
    # process_doc2dial()
    # process_multiwoz()
    # process_molweni()
    process_topicalchat()