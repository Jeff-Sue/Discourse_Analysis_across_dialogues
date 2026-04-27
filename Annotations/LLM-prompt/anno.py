import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from api import parallel_inference


def load_prompts():
    """Load BeDiscovER and user prompts"""
    
    with open("BeDiscovER.prompt", "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()
    
    with open("user.prompt", "r", encoding="utf-8") as f:
        user_template = f.read().strip()
    
    return system_prompt, user_template


def build_context_and_structure(edus, discourse_results, turn_idx):
    """
    Build context and structure for the current turn
    
    Args:
        edus: List of all EDUs
        discourse_results: List of discourse analysis results for previous turns
        turn_idx: Current turn index (0-based)
    
    Returns:
        context_str: Formatted context string
        structure_str: Formatted structure string
    """
    context_lines = []
    structure_lines = []
    
    # Add all previous turns as context
    for i in range(turn_idx):
        edu = edus[i]
        turn_id = i + 1
        speaker = edu["speaker"]
        text = edu["text"]
        context_lines.append(f"Turn {turn_id} ({speaker}): {text}")
        
        # Add discourse structure for this turn if available
        if i < len(discourse_results) and discourse_results[i]:
            structure_lines.append(f"Turn {turn_id}: {discourse_results[i]}")
    
    context_str = "\n".join(context_lines) if context_lines else "No previous turns"
    structure_str = "\n".join(structure_lines) if structure_lines else "No previous structure"
    
    return context_str, structure_str


def create_discourse_prompt(context, structure, new_turn_text, new_turn_id, user_template):
    """
    Create the user prompt for discourse analysis
    
    Args:
        context: Context text
        structure: Previous structure text
        new_turn_text: Text of the new turn
        new_turn_id: ID of the new turn
        user_template: User prompt template
    
    Returns:
        Formatted user prompt
    """
    prompt = user_template.replace("<CONTEXT>", context)
    prompt = prompt.replace("<STRUCTURE>", structure)
    prompt = prompt.replace("<NEW TURN>", f"Turn {new_turn_id}: {new_turn_text}")
    
    return prompt


def validator_discourse(response):
    """
    Validate discourse structure response
    
    Args:
        response: Response from LLM
    
    Returns:
        (is_valid, parsed_result or error_message)
    """
    response = response.strip()
    
    # Check if response contains valid discourse relations
    valid_relations = [
        "ACKNOWLEDGEMENT", "ALTERNATION", "BACKGROUND", "CLARIF_Q", "COMMENT",
        "CONDITIONAL", "CONTINUATION", "CONTRAST", "CORRECTION", "ELABORATION",
        "EXPLANATION", "NARRATION", "PARALLEL", "QA_PAIR", "Q_ELABORATION", "RESULT"
    ]
    
    # Check format: RELATION(S_ID1, S_ID2)
    if not response or response.lower() == "none":
        return True, ""
    
    # Validate format roughly
    if "(" in response and ")" in response:
        # Extract relation name
        relation_part = response.split("(")[0].strip()
        if relation_part in valid_relations:
            return True, response
        else:
            return False, f"Unknown relation: {relation_part}"
    
    return False, f"Invalid format: {response}"


def process_dialogue_incremental(dialogue, system_prompt, user_template, max_retry=3):
    """
    Process a single dialogue with incremental discourse analysis
    
    Args:
        dialogue: Dictionary with 'id' and 'edus' (list of EDU dicts)
        system_prompt: System prompt for LLM
        user_template: User prompt template
        max_retry: Max retry attempts
    
    Returns:
        Updated dialogue with discourse results
    """
    dialogue_id = dialogue.get("id")
    edus = dialogue.get("edus", [])
    
    if len(edus) <= 1:
        # Single turn dialogue has no structure relations
        dialogue["discourse_structure"] = [""]
        return dialogue
    
    # Initialize discourse results
    discourse_results = []
    
    # Process turns sequentially to maintain incremental context dependency
    for turn_idx in range(1, len(edus)):  # Start from turn 1 (second EDU)
        new_turn = edus[turn_idx]
        new_turn_id = turn_idx + 1
        
        # Build context and previous structure (now with previously processed results)
        context, structure = build_context_and_structure(edus, discourse_results, turn_idx)
        
        # Create prompt
        user_prompt = create_discourse_prompt(
            context,
            structure,
            new_turn["text"],
            new_turn_id,
            user_template
        )
        
        # Create sample for processing
        sample = {
            "id": f"{dialogue_id}_turn_{new_turn_id}",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
        
        # Process one turn at a time
        result = parallel_inference(
            [sample],
            max_workers=1,
            max_retry=max_retry,
            validator=validator_discourse
        )[0]
        
        if result["success"]:
            discourse_results.append(result["parsed_response"])
        else:
            discourse_results.append(None)
            print(f"Failed to process turn {new_turn_id} in dialogue {dialogue_id}: {result['error']}")
    
    # Add first turn as having no prior context (no structure)
    discourse_results.insert(0, "")
    
    # Add discourse results to dialogue
    dialogue["discourse_structure"] = discourse_results
    
    return dialogue


def process_dialogues_from_file(input_path, output_path, system_prompt, user_template, max_workers=4):
    """
    Process dialogues from input file and save to output file
    
    Args:
        input_path: Path to input JSONL or JSON file
        output_path: Path to output JSONL file
        system_prompt: System prompt
        user_template: User template
        max_workers: Maximum number of concurrent processes for dialogue processing
    """
    # Load dialogues
    dialogues = []
    if input_path.endswith('.jsonl'):
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    dialogues.append(json.loads(line))
    else:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                dialogues = data
            else:
                dialogues = [data]
    
    print(f"Loaded {len(dialogues)} dialogues from {input_path}")
    
    # Process dialogues concurrently
    processed_dialogues = [None] * len(dialogues)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all dialogue processing tasks
        future_to_idx = {
            executor.submit(process_dialogue_incremental, dialogue, system_prompt, user_template): idx
            for idx, dialogue in enumerate(dialogues)
        }
        
        print(f"Processing {len(dialogues)} dialogues with {max_workers} concurrent processes...")
        
        # Collect results as they complete
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                processed_dialogues[idx] = future.result()
                print(f"Completed dialogue {idx + 1}/{len(dialogues)}: {processed_dialogues[idx].get('id', 'unknown')}")
            except Exception as e:
                print(f"Failed to process dialogue {idx + 1}: {str(e)}")
                # Keep original dialogue if processing failed
                processed_dialogues[idx] = dialogues[idx]
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for dialogue in processed_dialogues:
            f.write(json.dumps(dialogue, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(processed_dialogues)} processed dialogues to {output_path}")


def main():
    """Main function"""
    # Load prompts
    system_prompt, user_template = load_prompts()
    
    # Process different datasets
    base_data_path = Path(__file__).parent.parent.parent / "Data"
    
    datasets = [
        ("doc2dial", "dialogue.json"),
        ("molweni", "dialogue.json"),
        ("multiwoz", "dialogue.json"),
        ("topical-chat", "dialogue.json")
    ]
    
    output_base_path = Path(__file__).parent / "processed_dialogues"
    
    # Set number of concurrent processes for dialogue processing
    max_workers = 4  # Adjust based on your CPU cores and API rate limits
    
    for dataset_name, filename in datasets:
        input_path = base_data_path / dataset_name / filename
        output_path = output_base_path / f"{dataset_name}_with_discourse.jsonl"
        
        if input_path.exists():
            print(f"\n{'='*60}")
            print(f"Processing {dataset_name}")
            print(f"{'='*60}")
            process_dialogues_from_file(input_path, str(output_path), system_prompt, user_template, max_workers)
        else:
            print(f"Input file not found: {input_path}")


if __name__ == "__main__":
    main()