import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from api import parallel_inference


def load_prompts():
    """Load DIMSUM prompt"""
    
    with open("DIMSUM.prompt", "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()
    
    return system_prompt


def build_dialogue_context(edus):
    """
    Build context string from all EDUs in the dialogue
    
    Args:
        edus: List of all EDUs
    
    Returns:
        Formatted dialogue context
    """
    context_lines = []
    
    for i, edu in enumerate(edus):
        turn_id = i + 1
        speaker = edu.get("speaker", "Unknown")
        text = edu.get("text", "")
        context_lines.append(f"[Utterance {turn_id}] {speaker}: {text}")
    
    return "\n\n".join(context_lines)


def create_dimsun_prompt(dialogue_context, system_prompt):
    """
    Create the full prompt for DIMSUM analysis
    
    Args:
        dialogue_context: Formatted dialogue context
        system_prompt: DIMSUM system prompt
    
    Returns:
        Formatted user prompt
    """
    user_prompt = f"""Please analyze the following dialogue using the structured relational framework:

{dialogue_context}

Provide the analysis in the specified output format."""
    
    return user_prompt


def validator_dimsum(response):
    """
    Validate DIMSUM response format
    
    Args:
        response: Response from LLM
    
    Returns:
        (is_valid, parsed_result or error_message)
    """
    response = response.strip()
    
    if not response or response.lower() == "none":
        return False, "Empty response"
    
    # Check if response contains expected sections
    required_sections = ["Topics:", "Relationships:", "Premises:", "Narrative Structure:", "Rhetorical Relationships:"]
    
    for section in required_sections:
        if section.lower() not in response.lower():
            return False, f"Missing required section: {section}"
    
    return True, response


def process_dialogue_whole(dialogue, system_prompt, max_retry=3):
    """
    Process a single dialogue with whole dialogue analysis (DIMSUM)
    
    Args:
        dialogue: Dictionary with 'id' and 'edus' (list of EDU dicts)
        system_prompt: System prompt for LLM
        max_retry: Max retry attempts
    
    Returns:
        Updated dialogue with DIMSUM analysis results
    """
    dialogue_id = dialogue.get("id")
    edus = dialogue.get("edus", [])
    
    if len(edus) <= 1:
        # Single utterance dialogue has no structure
        dialogue["dimsum_analysis"] = {
            "topics": "N/A - single utterance",
            "relationships": "N/A",
            "premises": "N/A",
            "narrative_structure": "N/A",
            "rhetorical_relationships": "N/A"
        }
        return dialogue
    
    # Build dialogue context
    dialogue_context = build_dialogue_context(edus)
    
    # Create prompt
    user_prompt = create_dimsun_prompt(dialogue_context, system_prompt)
    
    # Create sample for processing
    sample = {
        "id": dialogue_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }
    
    # Process the whole dialogue at once
    result = parallel_inference(
        [sample],
        max_workers=1,
        max_retry=max_retry,
        validator=validator_dimsum
    )[0]
    
    if result["success"]:
        dialogue["dimsum_analysis"] = {
            "prompt": {
                "system": system_prompt,
                "user": user_prompt
            },
            "raw_response": result["parsed_response"]
        }
    else:
        dialogue["dimsum_analysis"] = {
            "prompt": {
                "system": system_prompt,
                "user": user_prompt
            },
            "error": result["error"]
        }
        print(f"Failed to process dialogue {dialogue_id}: {result['error']}")
    
    return dialogue


def parse_dimsum_response(response):
    """
    Parse DIMSUM response into structured fields
    
    Args:
        response: Raw response from LLM
    
    Returns:
        Dictionary with parsed fields
    """
    result = {
        "topics": "",
        "relationships": "",
        "premises": "",
        "narrative_structure": "",
        "rhetorical_relationships": ""
    }
    
    current_section = None
    lines = response.split('\n')
    
    for line in lines:
        line = line.strip()
        lower_line = line.lower()
        
        if lower_line.startswith("topics:"):
            current_section = "topics"
            result["topics"] = line[7:].strip()  # Remove "Topics:"
        elif lower_line.startswith("relationships:"):
            current_section = "relationships"
            result["relationships"] = line[14:].strip()
        elif lower_line.startswith("premises:"):
            current_section = "premises"
            result["premises"] = line[9:].strip()
        elif lower_line.startswith("narrative structure:"):
            current_section = "narrative_structure"
            result["narrative_structure"] = line[19:].strip()
        elif lower_line.startswith("rhetorical relationships:"):
            current_section = "rhetorical_relationships"
            result["rhetorical_relationships"] = line[24:].strip()
        elif current_section and line:
            # Append to current section
            if current_section == "topics":
                result["topics"] += " " + line
            elif current_section == "relationships":
                result["relationships"] += " " + line
            elif current_section == "premises":
                result["premises"] += " " + line
            elif current_section == "narrative_structure":
                result["narrative_structure"] += " " + line
            elif current_section == "rhetorical_relationships":
                result["rhetorical_relationships"] += " " + line
    
    return result


def process_dialogues_from_file(input_path, output_path, system_prompt, max_workers=4):
    """
    Process dialogues from input file and save to output file
    
    Args:
        input_path: Path to input JSONL or JSON file
        output_path: Path to output JSONL file
        system_prompt: System prompt
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
            executor.submit(process_dialogue_whole, dialogue, system_prompt): idx
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
    system_prompt = load_prompts()
    
    # Process different datasets
    base_data_path = Path(__file__).parent.parent.parent / "Data"
    
    datasets = [
        ("doc2dial", "dialogue.json"),
        ("molweni", "dialogue.json"),
        ("multiwoz", "dialogue.json"),
        ("topical-chat", "dialogue.json")
    ]
    
    output_base_path = Path(__file__).parent / "processed_dialogues" / "DIMSUM"
    
    # Set number of concurrent processes for dialogue processing
    max_workers = 4  # Adjust based on your CPU cores and API rate limits
    
    for dataset_name, filename in datasets:
        input_path = base_data_path / dataset_name / filename
        output_path = output_base_path / f"{dataset_name}.jsonl"
        
        if input_path.exists():
            print(f"\n{'='*60}")
            print(f"Processing {dataset_name} with DIMSUM")
            print(f"{'='*60}")
            process_dialogues_from_file(input_path, str(output_path), system_prompt, max_workers)
        else:
            print(f"Input file not found: {input_path}")


if __name__ == "__main__":
    main()