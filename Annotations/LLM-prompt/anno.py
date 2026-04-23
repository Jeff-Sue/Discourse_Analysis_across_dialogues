import os
import json
import multiprocessing
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import shutil
import random
from abc import ABC, abstractmethod

from tqdm import tqdm
from openai import OpenAI

# Import custom modules for tree edit distance calculation
from GED import parse_tree_data_from_string, multicore_ged_astar_parallel, calculate_tree_ged
from GED2 import build_tree_from_edges, APTED


@dataclass
class ModelConfig:
    """Configuration for AI models and API settings."""
    model_id: str = "deepseek-r1"
    scoring_model_id: str = "deepseek-v3"
    api_key: str = "sk-VAw0SdgEVIouel9D6c0c6f1497B946DeAb03E31e9d606072"
    api_base_url: str = 'https://api.shubiaobiao.cn/v1/'
    temperature: float = 0.1
    processes_num: int = 50


@dataclass
class PathConfig:
    """Configuration for file paths and directories."""
    dir: str = "Annotation/Molweni/ELLA-16epoch"
    prompts_dir: str = "Prompts"
    
    def get_prompt_path(self, prompt_name: str) -> str:
        """Get full path for a prompt file."""
        return os.path.join(self.prompts_dir, f"{prompt_name}.txt")


@dataclass
class PromptTemplates:
    """Container for all prompt templates."""
    task_description: str = ""
    rhe_definition: str = ""
    rhe_definition_refine: str = ""
    rhe_definition_evaluation: str = ""
    rhe_definition_reflection: str = ""
    cot_refine: str = ""
    cot_evaluation: str = ""
    cot_reflection: str = ""
    feedback_refine: str = ""


@dataclass
class ProcessingResult:
    """Result of processing a single sample."""
    index: int
    success: bool
    error_message: Optional[str] = None
    tokens_used: int = 0


class PromptLoader:
    """Handles loading and managing prompt templates."""
    
    def __init__(self, path_config: PathConfig):
        self.path_config = path_config
    
    def load_prompts(self) -> PromptTemplates:
        """Load all prompt templates from files."""
        prompt_files = {
            'task_description': 'discourse_update.txt',
            'rhe_definition': 'Rhe_def.txt',
            "rhe_definition_evaluation": "Rhe_def_evaluation.txt",
            "rhe_definition_reflection": "Rhe_def_reflection.txt",
            # 'rhe_definition_refine': 'Rhe_def_refine.txt',
            # 'cot_refine': 'CoT_refine.txt',
            'cot_evaluation': 'CoT_evaluation.txt',
            'cot_reflection': 'CoT_reflection.txt',
            'feedback_refine': 'Refine.txt'
        }
        
        prompts = {}
        for key, filename in prompt_files.items():
            file_path = os.path.join(self.path_config.prompts_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    prompts[key] = f.read()
            except FileNotFoundError:
                print(f"Warning: Prompt file {file_path} not found. Using empty string.")
                prompts[key] = ""
        
        return PromptTemplates(**prompts)


class BaseProcessor(ABC):
    """Abstract base class for different types of processors."""
    
    def __init__(self, model_config: ModelConfig, output_path: str):
        self.model_config = model_config
        self.output_path = output_path
        self.failed_samples: List[Tuple[int, str]] = []
    
    def _create_client(self) -> OpenAI:
        """Create and return OpenAI client."""
        return OpenAI(
            api_key=self.model_config.api_key,
            base_url=self.model_config.api_base_url,
        )
    
    def _file_exists(self, index: int) -> bool:
        """Check if output file already exists."""
        return os.path.exists(os.path.join(self.output_path, f"{index}.json"))
    
    def _extract_input_from_messages(self, messages: str) -> str:
        """Extract input string from messages."""
        if '</input>' in messages and '<input>' in messages:
            return messages.split('\n</input>')[0].split('<input>')[-1].strip()
        else:
            return messages.split('Below is the Dialogue:\n')[-1]
    
    def _save_output(self, index: int, output: Dict[str, Any]) -> None:
        """Save output to JSON file."""
        output_file = os.path.join(self.output_path, f"{index}.json")
        with open(output_file, mode="w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
    
    @abstractmethod
    def process_sample(self, sample: Tuple[int, str]) -> ProcessingResult:
        """Process a single sample. Must be implemented by subclasses."""
        pass


class StandardProcessor(BaseProcessor):
    """Processor for standard inference tasks."""
    
    def process_sample(self, sample: Tuple[int, str]) -> ProcessingResult:
        """Process a single sample with standard inference."""
        index, messages = sample
        
        if self._file_exists(index):
            print(f"File {index}.json already exists, skipping.")
            return ProcessingResult(index=index, success=True)
        
        client = self._create_client()
        
        try:
            completion = client.chat.completions.create(
                model=self.model_config.model_id,
                messages=[{"role": "user", "content": messages}],
                temperature=self.model_config.temperature,
            )
            
            # Extract reasoning and response
            content = completion.choices[0].message.content
            if '<think>' in content and '</think>' in content:
                reasoning_content = content.split('<think>')[-1].split('</think>')[0]
                response = content.split('</think>')[-1]
            else:
                reasoning_content = ""
                response = content
            
            tokens = completion.usage.total_tokens
            input_str = self._extract_input_from_messages(messages)
            
            output = {
                "messages": [{"role": "user", "content": messages}],
                "input": input_str,
                "reasoning": reasoning_content,
                "response": response,
                "tokens": tokens
            }
            
            self._save_output(index, output)
            return ProcessingResult(index=index, success=True, tokens_used=tokens)
            
        except Exception as e:
            error_msg = f"Error processing sample {index}: {str(e)}"
            print(error_msg)
            self.failed_samples.append(sample)
            return ProcessingResult(index=index, success=False, error_message=error_msg)


class ScoringProcessor(BaseProcessor):
    """Processor for scoring tasks with uncertainty calculation."""
    
    def average_negative_log_probability(self, log_probs: List[float]) -> float:
        """Calculate average negative log probability (cross-entropy per token)."""
        if not log_probs:
            return 0.0
        return -sum(log_probs) / len(log_probs)
    
    def process_sample(self, sample: Tuple[int, str]) -> ProcessingResult:
        """Process a single sample with scoring and uncertainty calculation."""
        index, messages = sample
        if self._file_exists(index):
            print(f"File {index}.json already exists, skipping.")
            return ProcessingResult(index=index, success=True)
        
        client = self._create_client()
        
        try:
            completion = client.chat.completions.create(
                model=self.model_config.scoring_model_id,
                messages=[{"role": "user", "content": messages}],
                temperature=self.model_config.temperature,
                logprobs=True
            )
            
            # Extract log probabilities
            log_probs = []
            for choice in completion.choices:
                if choice.logprobs and choice.logprobs.content:
                    for token in choice.logprobs.content:
                        log_probs.append(token.logprob)
            
            uncertainty = self.average_negative_log_probability(log_probs)
            response = completion.choices[0].message.content
            tokens = completion.usage.total_tokens
            input_str = self._extract_input_from_messages(messages)
            
            output = {
                "messages": [{"role": "user", "content": messages}],
                "input": input_str,
                "uncertainty": uncertainty,
                "response": response,
                "tokens": tokens
            }
            
            self._save_output(index, output)
            return ProcessingResult(index=index, success=True, tokens_used=tokens)
            
        except Exception as e:
            error_msg = f"Error processing sample {index}: {str(e)}"
            print(error_msg)
            self.failed_samples.append(sample)
            return ProcessingResult(index=index, success=False, error_message=error_msg)


class DataManager:
    """Manages data loading, saving, and instance pool operations."""
    
    def __init__(self, initial_output_path: str = ""):
        self.current_output_path = initial_output_path
        self.instance_pool: List[Dict[str, str]] = []
    
    def set_output_path(self, output_path: str) -> None:
        """Set the current output path for operations."""
        self.current_output_path = output_path
    
    def extend_instance_pool(self) -> None:
        """Extend instance pool with data from current output directory."""
        if not self.current_output_path or not os.path.exists(self.current_output_path):
            print(f"Warning: Output path {self.current_output_path} does not exist")
            return
            
        for filename in os.listdir(self.current_output_path):
            if filename.endswith('.json'):
                filepath = os.path.join(self.current_output_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'response' in data and 'input' in data:
                            instance = {
                                "Dialogue": data['input'],
                                "Rhetorical_Structure": data['response']
                            }
                            if instance not in self.instance_pool:
                                self.instance_pool.append(instance)
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error reading {filepath}: {e}")
    
    def extract_instances_by_order(self) -> List[Dict[str, Any]]:
        """Extract instances ordered by filename number from current output directory."""
        if not self.current_output_path or not os.path.exists(self.current_output_path):
            print(f"Warning: Output path {self.current_output_path} does not exist")
            return []
            
        filenames = [f for f in os.listdir(self.current_output_path) if f.endswith('.json')]
        sorted_filenames = sorted(filenames, key=lambda x: int(x.split('.')[0]))
        
        instances = []
        for filename in sorted_filenames:
            filepath = os.path.join(self.current_output_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    instances.append({
                        "input": data.get('input', ''),
                        "reasoning": data.get('reasoning', ''),
                        "response": data.get('response', '')
                    })
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading {filepath}: {e}")
        
        return instances


class SelectionStrategy(ABC):
    """Abstract base class for instance selection strategies."""
    
    @abstractmethod
    def select_indices(self, output_path: str, **kwargs) -> List[int]:
        """Select indices based on the strategy."""
        pass


class DiversitySelector(SelectionStrategy):
    """Selects instances based on diversity scores using tree edit distance."""
    
    def __init__(self, instance_pool: List[Dict[str, str]]):
        self.instance_pool = instance_pool
        self._gold_tree_structures = self._preprocess_gold_structures()
    
    def _preprocess_gold_structures(self) -> List[List[Any]]:
        """Preprocess gold structures from instance pool."""
        gold_structures = []
        for pool_instance in self.instance_pool:
            try:
                rhe_structure = parse_tree_data_from_string(pool_instance['Rhetorical_Structure'])
                gold_structures.append(rhe_structure)
            except Exception as e:
                print(f"Error parsing gold structure: {e}")
                # Skip invalid structures
                continue
        return gold_structures
    
    def _parse_discourse_structure(self, response_string: str) -> Tuple[List[Any], int]:
        """
        Parse discourse structure from response string.
        
        Returns:
            Tuple of (rhetorical_structure, topic_shift_count)
        """
        try:
            pred_discourse_structure = parse_tree_data_from_string(response_string)
            pred_rhe_structure = [pd[:3] for pd in pred_discourse_structure]
            
            # Count topic shifts
            pred_shift_count = sum(1 for d_s in pred_discourse_structure 
                                 if len(d_s) > 3 and d_s[-1] == 'topic shift')
            
            return pred_rhe_structure, pred_shift_count
        except Exception as e:
            print(f"Error parsing discourse structure: {e}")
            return [], 0
    
    def _calculate_tree_edit_distance(self, pred_structure: List[Any], 
                                    gold_structure: List[Any]) -> float:
        """
        Calculate tree edit distance between predicted and gold structures.
        
        Args:
            pred_structure: Predicted rhetorical structure
            gold_structure: Gold rhetorical structure
            
        Returns:
            Tree edit distance score
        """
        try:
            if len(pred_structure) == 0 or len(gold_structure) == 0:
                return 0.0
            
            # Extract rhetorical structure (first 3 elements)
            gold_rhe_structure = [gd[:3] for gd in gold_structure]
            
            # Use APTED to calculate tree edit distance
            apted = APTED()
            distance = float(apted.distance(
                build_tree_from_edges(pred_structure), 
                build_tree_from_edges(gold_rhe_structure)
            ))
            # distance2 = float(calculate_tree_ged(pred_structure, gold_rhe_structure)
            # )
            
            return distance
        except Exception as e:
            print(f"Error calculating tree edit distance: {e}")
            return 0.0
    
    def _calculate_diversity_score(self, pred_structure: List[Any], 
                                 pred_shift_count: int) -> float:
        """
        Calculate diversity score by comparing against all gold structures.
        
        Args:
            pred_structure: Predicted rhetorical structure
            pred_shift_count: Number of topic shifts in prediction
            
        Returns:
            Average diversity score across all gold structures
        """
        if not self._gold_tree_structures:
            return 0.0
        
        total_score = 0.0
        valid_comparisons = 0
        
        for gold_structure in self._gold_tree_structures:
            try:
                # Count topic shifts in gold structure
                gold_shift_count = sum(1 for g_s in gold_structure 
                                     if len(g_s) > 3 and g_s[-1] == 'topic shift')
                
                # Calculate tree edit distance
                tree_distance = self._calculate_tree_edit_distance(
                    pred_structure, gold_structure
                )
                
                # Optional: Add topic shift difference to the score
                # shift_difference = abs(gold_shift_count - pred_shift_count)
                # score = tree_distance + shift_difference
                
                total_score += tree_distance
                valid_comparisons += 1
                
                # Debug information (matching original code)
                if hasattr(self, '_debug_mode') and self._debug_mode:
                    print("#" * 10)
                    print(f"Predicted structure: {pred_structure}")
                    print(f"Gold structure: {[gd[:3] for gd in gold_structure]}")
                    print(f"Shift counts - Gold: {gold_shift_count}, Pred: {pred_shift_count}")
                    print(f"Tree distance: {tree_distance}")
                    print("#" * 10)
                
            except Exception as e:
                print(f"Error in diversity calculation: {e}")
                continue
        
        # Return average score across all valid comparisons
        return total_score / valid_comparisons if valid_comparisons > 0 else 0.0
    
    def select_indices(self, output_path: str, top_k: int = 1, 
                      save_scores: bool = True) -> List[int]:
        """
        Select top-k most diverse instances based on tree edit distance.
        
        Args:
            output_path: Directory containing JSON files to evaluate
            top_k: Number of top diverse instances to select
            save_scores: Whether to save diversity scores back to JSON files
            
        Returns:
            List of indices for the most diverse instances
        """
        scores = []
        filenames = [f for f in os.listdir(output_path) if f.endswith('.json')]
        sorted_filenames = sorted(filenames, key=lambda x: int(x.split('.')[0]))
        
        if not self._gold_tree_structures:
            print("Warning: No valid gold structures found in instance pool")
            return list(range(min(top_k, len(sorted_filenames))))
        
        for filename in tqdm(sorted_filenames, desc="Calculating diversity scores"):
            filepath = os.path.join(output_path, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract response and parse discourse structure
                pred_string = data.get('response', '')
                pred_structure, pred_shift_count = self._parse_discourse_structure(pred_string)
                
                # Calculate diversity score
                diversity_score = self._calculate_diversity_score(
                    pred_structure, pred_shift_count
                )
                
                scores.append(diversity_score)
                
                # Save diversity score back to the file if requested
                if save_scores:
                    data['diversity'] = diversity_score
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=4)
                
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                scores.append(0.0)
        
        # Return indices sorted by diversity score (highest first)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        print(f"Selected {len(top_indices)} most diverse instances with scores: "
              f"{[scores[i] for i in top_indices]}")
        
        return top_indices
    
    def enable_debug_mode(self) -> None:
        """Enable debug mode for detailed logging."""
        self._debug_mode = True
    
    def disable_debug_mode(self) -> None:
        """Disable debug mode."""
        self._debug_mode = False


class UncertaintySelector(SelectionStrategy):
    """Selects instances based on uncertainty scores."""
    
    def select_indices(self, output_path: str, top_k: int = 1) -> List[int]:
        """Select top-k most uncertain instances."""
        uncertainty_list = []
        filenames = [f for f in os.listdir(output_path) if f.endswith('.json')]
        sorted_filenames = sorted(filenames, key=lambda x: int(x.split('.')[0]))
        
        for filename in sorted_filenames:
            filepath = os.path.join(output_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    uncertainty = data.get('uncertainty', 0.0)
                    uncertainty_list.append(float(uncertainty))
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                uncertainty_list.append(0.0)
        
        return sorted(range(len(uncertainty_list)), key=lambda i: uncertainty_list[i], reverse=True)[:top_k]


class ELLA:
    """Enhanced Learning Loop for Annotation - Main orchestrator class."""
    
    def __init__(self, data: List[Dict[str, Any]], test_data: List[Dict[str, Any]]):
        # Configuration
        self.model_config = ModelConfig()
        self.path_config = PathConfig()
        
        # Data
        self.remaining_data = data
        self.test_data = test_data
        
        # Components
        self.prompt_loader = PromptLoader(self.path_config)
        self.prompts = self.prompt_loader.load_prompts()
        self.data_manager = DataManager("")  # Will be set dynamically
        
        # Selection strategies (initialize with empty pool, will be updated later)
        self.diversity_selector = DiversitySelector([])
        self.diversity_golden = ""
        self.uncertainty_selector = UncertaintySelector()
        self.uncertainty_golden = ""
        
        # Processors
        self.standard_processor = None
        self.scoring_processor = None
    
    def _setup_processors(self, output_path: str) -> None:
        """Setup processors with current output path."""
        self.standard_processor = StandardProcessor(self.model_config, output_path)
        self.scoring_processor = ScoringProcessor(self.model_config, output_path)
    
    def _process_samples_parallel(self, samples: List[Tuple[int, str]], 
                                processor: BaseProcessor) -> List[ProcessingResult]:
        """Process samples in parallel using multiprocessing."""
        with multiprocessing.Pool(processes=self.model_config.processes_num) as pool:
            results = [
                pool.apply_async(processor.process_sample, args=(sample,))
                for sample in samples
            ]
            
            processed_results = []
            for r in tqdm(results, desc="Processing samples", unit="sample"):
                r.wait()
                processed_results.append(r.get())
            
            pool.close()
            pool.join()
        
        return processed_results
    
    def _copy_selected_files(self, source_dir: str, target_dir: str, indices: List[int]) -> None:
        """Copy selected files from source to target directory."""
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        
        for idx in indices:
            filename = f"{idx}.json"
            src_path = os.path.join(source_dir, filename)
            dst_path = os.path.join(target_dir, filename)
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
            else:
                print(f"Warning: Source file {src_path} does not exist")
    
    def _update_prompts(self, epoch: int, selected_instances: List[Dict[str, Any]]) -> None:
        """Update prompts based on selected instances and golden data."""
        # Load golden data
        self._create_gloden_data(epoch)
        golden_data = self._load_golden_data(epoch)
        
        # Update rhetorical definition prompt
        self._update_rhetorical_definition_prompt(epoch, golden_data)
        
        # Update task description prompt
        self._update_task_description_prompt(epoch, golden_data)

    def _create_gloden_data(self, epoch):
        diversity_dir = os.path.join(self.path_config.dir, str(epoch), "Select", "Diversity")
        golden_data = {}
        for filename in os.listdir(diversity_dir):
            with open(f"{diversity_dir}/{filename}", 'r') as f:
                diversity_data = json.load(f)
                golden_data = diversity_data
                golden_data['golden'] = self.diversity_golden
        
        golden_dir = os.path.join(self.path_config.dir, str(epoch), "Gold")
        golden_file = os.path.join(self.path_config.dir, str(epoch), "Gold", "0.json")
        Path(golden_dir).mkdir(parents=True, exist_ok=True)
        with open(golden_file, 'w') as f:
            json.dump(golden_data, f, indent=4, ensure_ascii=False)

    def _load_golden_data(self, epoch: int) -> Dict[str, str]:
        """Load golden data for the current epoch."""
        golden_data = {
            'dialogues': '', 'reasoning': '', 'response': '', 'golden': ''
        }
        
        for i in range(1):
            golden_file = os.path.join(self.path_config.dir, str(epoch), "Gold", f"{i}.json")
            try:
                with open(golden_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    golden_data['dialogues'] += f"{data['input'].split('Dialogue: ')[-1]}\n"
                    golden_data['reasoning'] += f"{data['reasoning']}\n"
                    golden_data['response'] += f"{data['response']}\n"
                    golden_data['golden'] += f"{data['golden']}\n"
            except FileNotFoundError:
                print(f"Warning: Golden file {golden_file} not found")
        
        return golden_data
    
    def _update_rhetorical_definition_prompt(self, epoch: int, golden_data: Dict[str, str]) -> None:
        """Update rhetorical definition prompt."""
        # Definition Evaluation
        definition_evaluation_prompt = self.prompts.rhe_definition_evaluation.format(
            Previous_Definitions=self.prompts.rhe_definition,
            dialogue=golden_data['dialogues'],
            reasoning=golden_data['reasoning'],
            response=golden_data['response'],
            golden_response=golden_data['golden']
        )
        
        output_dir = os.path.join(self.path_config.dir, str(epoch), "Refine", "RD", "evaluation")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self._setup_processors(output_dir)
        results = self._process_samples_parallel([(0, definition_evaluation_prompt)], self.standard_processor)

        if results and results[0].success:
            with open(os.path.join(output_dir, "0.json"), 'r', encoding='utf-8') as f:
                refined_data = json.load(f)

        # Definition Reflection
        definition_reflection_prompt = self.prompts.rhe_definition_reflection.format(
            Previous_Definitions=self.prompts.rhe_definition,
            dialogue=golden_data['dialogues'],
            reasoning=golden_data['reasoning'],
            response=golden_data['response'],
            golden_response=golden_data['golden'],
            evaluation=refined_data['response'],
        )
        
        output_dir = os.path.join(self.path_config.dir, str(epoch), "Refine", "RD", "reflection")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self._setup_processors(output_dir)
        results = self._process_samples_parallel([(0, definition_reflection_prompt)], self.standard_processor)

        # Extract refined prompt
        if results and results[0].success:
            with open(os.path.join(output_dir, "0.json"), 'r', encoding='utf-8') as f:
                refined_data = json.load(f)
                # Update the rhetorical definition prompt
                self.prompts.rhe_definition = self._extract_refined_content(
                    refined_data['response'], 'refined_rhetorical_relations_definitions'
                )
    
    def _update_task_description_prompt(self, epoch: int, golden_data: Dict[str, str]) -> None:
        """Update task description prompt."""
        
        # CoT Evaluation
        cot_evaluation_prompt = self.prompts.cot_evaluation.format(
            Previous_Prompt=self.prompts.task_description,
            dialogue=golden_data['dialogues'],
            reasoning=golden_data['reasoning'],
            response=golden_data['response'],
            golden_response=golden_data['golden']
        )
        output_dir = os.path.join(self.path_config.dir, str(epoch), "Refine", "CoT", "evaluation")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self._setup_processors(output_dir)
        results = self._process_samples_parallel([(0, cot_evaluation_prompt)], self.standard_processor)
        if results and results[0].success:
            with open(os.path.join(output_dir, "0.json"), 'r', encoding='utf-8') as f:
                refined_data = json.load(f)

        # CoT Reflection
        cot_reflection_prompt = self.prompts.cot_reflection.format(
            Previous_Prompt=self.prompts.task_description,
            dialogue=golden_data['dialogues'],
            reasoning=golden_data['reasoning'],
            response=golden_data['response'],
            golden_response=golden_data['golden'],
            evaluation=refined_data['response'],
        )
        
        output_dir = os.path.join(self.path_config.dir, str(epoch), "Refine", "CoT", "reflection")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self._setup_processors(output_dir)
        results = self._process_samples_parallel([(0, cot_reflection_prompt)], self.standard_processor)

        
        # Extract refined prompt
        if results and results[0].success:
            with open(os.path.join(output_dir, "0.json"), 'r', encoding='utf-8') as f:
                refined_data = json.load(f)
                # Update the task description prompt
                self.prompts.task_description = self._extract_refined_content(
                    refined_data['response'], 'refined_task_description'
                )
    
    def _extract_refined_content(self, response: str, tag: str) -> str:
        """Extract refined content from response using XML-like tags."""
        start_tag = f'<{tag}>'
        end_tag = f'</{tag}>'
        
        if start_tag in response and end_tag in response:
            return response.split(start_tag)[-1].split(end_tag)[0].strip()
        return response
    
    def run_training_loop(self, num_epochs: int = 1) -> None:
        """Run the main ELLA training loop."""
        # Initialize with sample data
        self._initialize_training()
        
        for epoch in range(num_epochs):
            print(f"\n=== Epoch {epoch}: Starting ===")
            
            # Step 1: Generate candidates and calculate diversity
            self._generate_diversity_candidates(epoch)
            
            # Step 2: Generate candidates and calculate uncertainty
            self._generate_uncertainty_candidates(epoch)
            
            # Step 3: Update prompts based on selected instances
            selected_instances = self.data_manager.extract_instances_by_order()

            self._update_prompts(epoch, selected_instances)

            # Step 4: Test with updated prompts
            if (epoch + 1) % 4 == 0:
                self._test_updated_prompts(epoch)
            
            print(f"=== Epoch {epoch}: Completed ===\n")
    
    def _initialize_training(self) -> None:
        """Initialize training with initial sample data."""
        print("Initializing training with sample data...")
        
        random.seed(42)
        initial_data = random.sample(self.remaining_data, min(5, len(self.remaining_data)))
        self.remaining_data = [d for d in self.remaining_data if d not in initial_data]
        
        # Create initial annotator prompt
        annotator_prompt = f"{self.prompts.task_description}\n\n{self.prompts.rhe_definition}"
        samples = [(idx, f"{annotator_prompt}\nBelow is the Dialogue:\n{item['input']}")
                  for idx, item in enumerate(initial_data)]
        
        # Setup and process initial samples
        output_dir = os.path.join(self.path_config.dir, "Initial")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self._setup_processors(output_dir)
        self._process_samples_parallel(samples, self.standard_processor)
        
        # Update data manager with new output path
        self.data_manager.set_output_path(output_dir)
        self.data_manager.extend_instance_pool()
    
    def _generate_diversity_candidates(self, epoch: int) -> None:
        """Generate candidates for diversity selection."""
        print(f"Epoch {epoch}: Generating diversity candidates...")
        
        # Sample candidate data
        random.seed(42)
        candidate_data = random.sample(self.remaining_data, min(20, len(self.remaining_data)))
        self.remaining_data = [d for d in self.remaining_data if d not in candidate_data]
        
        # Process candidates
        annotator_prompt = f"{self.prompts.task_description}\n\n{self.prompts.rhe_definition}"
        samples = [(idx, f"{annotator_prompt}\nBelow is the Dialogue:\n{item['input']}")
                  for idx, item in enumerate(candidate_data)]
        
        
        output_dir = os.path.join(self.path_config.dir, str(epoch), "Query", "R1")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self._setup_processors(output_dir)
        self._process_samples_parallel(samples, self.standard_processor)
        
        # Update diversity selector with current instance pool
        self.diversity_selector = DiversitySelector(self.data_manager.instance_pool)
        
        # Check if diversity results already exist
        diversity_dir = os.path.join(self.path_config.dir, str(epoch), "Select", "Diversity")
        if os.path.exists(diversity_dir) and os.path.isdir(diversity_dir):
            print(f"Found existing diversity directory: {diversity_dir}")
            filenames = [f for f in os.listdir(diversity_dir) if f.endswith('.json')]
            diverse_indices = sorted([int(filename.split('.')[0]) for filename in filenames])
            print(f"Existing diversity indices: {diverse_indices}")
        else:
            # Select diverse instances
            diverse_indices = self.diversity_selector.select_indices(output_dir, top_k=1)
        
        self.diversity_golden = candidate_data[diverse_indices[0]]['output']
        # Copy selected files
        target_dir = os.path.join(self.path_config.dir, str(epoch), "Select", "Diversity")
        self._copy_selected_files(output_dir, target_dir, diverse_indices)
        
        # Update instance pool with new data
        self.data_manager.set_output_path(target_dir)
        self.data_manager.extend_instance_pool()
    
    def _generate_uncertainty_candidates(self, epoch: int) -> None:
        """Generate candidates for uncertainty selection."""
        print(f"Epoch {epoch}: Generating uncertainty candidates...")
        
        # Use same candidate data as diversity (they should be the same samples)
        annotator_prompt = f"{self.prompts.task_description}\n\n{self.prompts.rhe_definition}"
        
        # Note: In the original code, this would use the same samples as diversity
        # For simplicity, we'll assume the samples are stored somewhere accessible
        
        output_dir = os.path.join(self.path_config.dir, str(epoch), "Query", "V3")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Process with scoring (this would need the actual samples)
        # self._setup_processors(output_dir)
        # self._process_samples_parallel(samples, self.scoring_processor)
        
        # Select uncertain instances
        uncertain_indices = self.uncertainty_selector.select_indices(output_dir, top_k=1)
        
        # Copy selected files
        target_dir = os.path.join(self.path_config.dir, str(epoch), "Select", "Uncertainty")
        self._copy_selected_files(output_dir, target_dir, uncertain_indices)
        
        # Update instance pool with new data
        self.data_manager.set_output_path(target_dir)
        self.data_manager.extend_instance_pool()
    
    def _test_updated_prompts(self, epoch: int) -> None:
        """Test updated prompts on test data."""
        print(f"Epoch {epoch}: Testing updated prompts...")
        
#         output_format = """**Output Format**  
# *AFTER* processing the whole dialogue, extract all discourse structures from output json, output in the format of [index1, index2, relation, topic1, topic2, topic action] within <discourse structure> tag, which index1 from "to" EDU site, index2 from "from" EDU site (index1 < index2), relation from "type", e.g. [0, 1, 'Comment'] and topic1, topic2 from the topic labels for EDU1 and EDU2, topic action from "topic maintain" or "topic shift" to determine whether there exist a abrupt topic shift between EDU1 and EDU2."""
        output_format = """**Output Format**  
*AFTER* processing the whole dialogue, extract all discourse structures from output json, output in the format of [index1, index2, relation] within <discourse structure> tag, which index1 from "to" EDU site, index2 from "from" EDU site (index1 < index2), relation from "type", e.g. [0, 1, 'Comment'] """
        annotator_prompt = f"{self.prompts.task_description}\n\n{self.prompts.rhe_definition}\n\n{output_format}"
        
        # Test on subset of test data
        test_samples = [(idx, f"{annotator_prompt}\nBelow is the Dialogue:\n{item['input']}")
                       for idx, item in enumerate(self.test_data[:200])]
        
        output_dir = os.path.join(self.path_config.dir, str(epoch), "test")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self._setup_processors(output_dir)
        results = self._process_samples_parallel(test_samples, self.standard_processor)
        
        # # Log results
        # successful = sum(1 for r in results if r.success)
        # total_tokens = sum(r.tokens_used for r in results if r.success)
        
        # print(f"Test results: {successful}/{len(results)} successful, {total_tokens} tokens used")

    def Fan_text(self):
        self._initialize_training()
        # Annotator_Prompt = self.task_description_prompt + "\n\n" + self.rhe_def_prompt
        previous_data = []
        with open("Prompts/Baselines/dimsum.txt", 'r', encoding='utf-8') as fr:
            Top_prompt = fr.read()
        output_format = """**Output Format**  
*AFTER* processing the whole dialogue, extract all discourse structures from output json, output in the format of [index1, index2, relation] within <discourse structure> tag, which index1 from "to" EDU site, index2 from "from" EDU site (index1 < index2), relation from "type", e.g. [0, 1, 'Comment'] """
        Annotator_Prompt = Top_prompt + "\n\n" + self.prompts.rhe_definition + "\n\n" + output_format
        # with open("Prompts/mutual.txt", 'r', encoding='utf-8') as fr:
        #     Annotator_Prompt = fr.read()
        annotator_data = [[idx, Annotator_Prompt + "\nBelow is the Dialogue:\n" + i['input']] for idx, i in enumerate(self.test_data[:200])]

        output_dir = f"Annotation/Molweni/dimsum"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self._setup_processors(output_dir)
        results = self._process_samples_parallel(annotator_data, self.standard_processor)


# Example usage
if __name__ == "__main__":
    # Example data structure
    sample_data = [
        {"input": "Sample dialogue 1"},
        {"input": "Sample dialogue 2"},
        # ... more data
    ]
    
    sample_test_data = [
        {"input": "Test dialogue 1"},
        {"input": "Test dialogue 2"},
        # ... more test data
    ]
    
    # Initialize and run ELLA
    ella = ELLA(data=sample_data, test_data=sample_test_data)
    ella.run_training_loop(num_epochs=1)