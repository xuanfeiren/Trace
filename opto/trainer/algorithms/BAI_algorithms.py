# BAI_algorithms.py
# This file contains the algorithms for the Best Candidate Identification (BAI) problem.
# For all BAI algorithm classes, we should have a buffer (candidate-score pairs) as input, and output the best candidate after every evaluation iteration.
# Here are several algorithms. 
# 1. Evenly split
# 2. UCB best candidate identification
# 3. LLM function approximation
import numpy as np
import copy
import time
from typing import Union
from opto import trace
from opto.trainer.algorithms.algorithm import AlgorithmBase
from opto.trainer.loader import DataLoader
from opto.trainer.utils import batch_run, async_run
from opto.optimizers.utils import print_color
from opto.trainer.evaluators import evaluate
from typing import Union, List, Tuple, Dict, Any, Optional
from collections import deque
from opto.utils.llm import LLM # For the selector LLM
from opto.trace.nodes import ParameterNode
import json
import warnings
from black import format_str, FileMode
import random
import math
from opto.trainer.utils import sample_minibatch, retry_with_exponential_backoff, evaluate_agent, construct_update_dict

class BAIAlgorithmBase(AlgorithmBase):
    """We define a base class for all BAI algorithms.
    The input is a list of update_dicts, which is a list of dictionaries that contain the update information.
    The output is the best candidate after every evaluation iteration.
    """
    def __init__(self,agent,num_threads, logger,update_dicts, *args, **kwargs):
        super().__init__(agent,num_threads, logger, *args, **kwargs)
        self.update_dicts = update_dicts
        self.total_samples = 0

    
    def train(self, guide, validate_dataset, test_dataset,num_threads, num_epochs,eval_frequency, **kwargs):
        """The training process of BAI Algorithm."""
        # 1. Initialize the buffer. For each update_dict, evaluate the agent once and add the candidate-score pair to the buffer.
        self.buffer = deque(maxlen=100)
        for update_dict in self.update_dicts:   
            # Evaluate one candidate and update the buffer statistics.
            self.agent._set(update_dict)
            score = evaluate_agent(self.agent, guide, validate_dataset, num_threads=num_threads, num_eval_times=1)
            eval_count = len(validate_dataset['inputs'])
            self.total_samples += eval_count
            candidate_entry = {"params": update_dict, "score_sum": score*eval_count, "eval_count": eval_count}
            self.buffer.append(candidate_entry)
        # 2. Best candidate identification. At each epoch, we will do some evaluation using sample budget, update the buffer statistics, and output the best candidate for test.
        for epoch in range(num_epochs):
            # Evaluate the best candidate at this epoch and update the buffer statistics.
            best_candidate_at_this_epoch, used_sample_budget = self.step(guide, validate_dataset, num_threads, **kwargs)
            self.total_samples += used_sample_budget
            
            if epoch % eval_frequency == 0:
                # Test the current best candidate.
                self.agent._set(best_candidate_at_this_epoch['params'])
                test_score = evaluate_agent(self.agent, guide, test_dataset, num_threads=num_threads, num_eval_times=1)
                self.logger.log("Test score", test_score,epoch+1,color='green')
            return 
        
    def step(self, guide, validate_dataset, num_threads, **kwargs):
        """The best candidate identification step. At each epoch, we will do some evaluation using sample budget, update the buffer statistics, and output the best candidate for test. The output should be the entry of the best candidate in the buffer and the used sample budget."""
        raise NotImplementedError("Subclasses must implement this method")

class EvenlySplitAlgorithm(BAIAlgorithmBase):
    """Evenly split the sample budget for each candidate."""

    def update_buffer_scores(self):
        for candidate_entry in self.buffer:
            candidate_entry['mean_score'] = candidate_entry['score_sum'] / (candidate_entry['eval_count'] or 1E-9)
        return 
    
    def step(self, guide, validate_dataset, num_threads, **kwargs):
        # Evaluate each candidate once on the validation set, and update the buffer statistics.
        for candidate in self.buffer:
            self.agent._set(candidate['params'])
            score = evaluate_agent(self.agent, guide, validate_dataset, num_threads=num_threads, num_eval_times=1)
            candidate['score_sum'] += score*len(validate_dataset['inputs'])
            candidate['eval_count'] += len(validate_dataset['inputs'])
        self.update_buffer_scores()
        # Select the best candidate according to the buffer statistics.
        best_candidate = max(self.buffer, key=lambda x: x['mean_score'])
        return best_candidate, len(self.buffer)*len(validate_dataset['inputs'])
    
class UCBAlgorithm(BAIAlgorithmBase):
    """UCB best candidate identification."""

    def __init__(self,agent,num_threads, logger,update_dicts, *args, **kwargs):
        super().__init__(agent,num_threads, logger,update_dicts, *args, **kwargs)
        self.ucb_exploration_factor = 0.1 # Set the exploration factor for UCB

    def _calculate_ucb(self, candidate_buffer_entry: Dict, total_tracked_evaluations: int) -> float:
        """Calculates UCB score for a candidate in the buffer."""
        if candidate_buffer_entry['eval_count'] == 0:
            return float('inf')  # Explore unvisited states first
        mean_score = candidate_buffer_entry['score_sum'] / candidate_buffer_entry['eval_count']
        exploration_term = self.ucb_exploration_factor * \
                           math.sqrt(math.log(total_tracked_evaluations) / candidate_buffer_entry['eval_count'])
        
        return mean_score + exploration_term
    
    def update_buffer_scores(self):
        total_evaluations_tracker = np.sum([c['eval_count'] for c in self.buffer])
        for candidate_entry in self.buffer:
            candidate_entry['ucb_score'] = self._calculate_ucb(candidate_entry, total_evaluations_tracker)
            candidate_entry['mean_score'] = candidate_entry['score_sum'] / (candidate_entry['eval_count'] or 1E-9)
        return 
    
    def step(self, guide, validate_dataset, num_threads, **kwargs):
        """To make the number of evauation the same for different BAI algorithms, we set ucb horiso to be number of candidates"""
        horizon = len(self.buffer)
        for iteration in range(horizon):
            print_color(f"Iteration {iteration+1}/{horizon}: ", 'blue')
            self.update_buffer_scores()
            selected_candidate = self.ucb_select_candidate(self.buffer)
            self.agent._set(selected_candidate['params'])
            score = evaluate_agent(self.agent, guide, validate_dataset, num_threads=num_threads, num_eval_times=1)
            selected_candidate['score_sum'] += score*len(validate_dataset['inputs'])
            selected_candidate['eval_count'] += len(validate_dataset['inputs'])
        return selected_candidate, len(self.buffer)*len(validate_dataset['inputs'])
    
    def select_candidate(self, buffer):
        """Select the candidate with the highest UCB score."""
        return max(buffer, key=lambda c: c['ucb_score'])


class LLMSelectorAlgorithm(UCBAlgorithm):
    """LLM selector best candidate identification."""
    def __init__(self,agent,num_threads, logger,update_dicts, *args, **kwargs):
        super().__init__(agent,num_threads, logger,update_dicts, *args, **kwargs)
        self.llm_model = "gemini/gemini-2.0-flash"


    def select_candidate(self, buffer):
        """At each select step, we call LLM with the buffer statistics (candidate-score pairs). """
        self.update_buffer_scores()
        return self.llm_generate_candidate(buffer, verbose=True)
        
    def llm_generate_candidate(self, buffer, verbose: bool = False): 
        """
        LLM can either select an existing candidate or generate a new candidate.
        Tries multiple times with exponential backoff until a valid output is parsed.
        Returns a single update_dict.
        """
       
        # Filter buffer to only include candidates with valid UCB scores
        valid_candidates = [c for c in buffer if c.get('mean_score') is not None and c.get('mean_score') != -float('inf') and c.get('mean_score') != float('inf')]
    
        sorted_buffer = sorted(valid_candidates, key=lambda c: c.get('mean_score', -float('inf')), reverse=True)
        prompt_candidates = sorted_buffer
        serializable_candidate_summaries = []
        for idx, cand_entry in enumerate(prompt_candidates):
            summary = {
                "index": idx,
                "parameters":  {p.py_name: copy.deepcopy(p.data) for p in cand_entry['params']},
                "eval_count": cand_entry['eval_count'],
                "mean_score": cand_entry['mean_score']
            }
            serializable_candidate_summaries.append(summary)
        candidate_summaries_json = json.dumps(serializable_candidate_summaries, indent=2)
        
        example_param_schema_json = json.dumps({p.py_name: copy.deepcopy(p.data) for p in self.agent.parameters()}, indent=2)

        prompt_messages = [
            {
                "role": "system",
                "content": """
## Role
You are assisting with best-arm identification using batched exploration. Here are the candidate arms and their statistics. 

## Decision Objective
Choose the next arm to a batched evaluation to maximize the chance of discovering a top arm under the limited budget. Balance exploitation and exploration using UCB-style reasoning (favor higher mean_score and higher uncertainty from lower eval_count).

## What You May Do
- Select an existing arm by its provided index (copy its parameter dictionary exactly), or
- Propose a new arm by providing a parameter dictionary with the same keys and string values.

## Output Requirements
Return ONLY a JSON object with these fields (no prose, no markdown, no extra fields):
- "reasoning": short double-quoted string explaining your choice.
- "existing_arm_index": integer index of an existing arm to evaluate next, or -1 if proposing a new arm.
- "new_update_dict": JSON object mapping parameter names to string values; use {} if selecting an existing arm, otherwise fill with your proposed parameter values when existing_arm_index is -1.

All keys and string values must use double quotes.
""",
            },
            {
                "role": "user",
                "content": f"""
## Context
{candidate_summaries_json}

## Parameter Schema (for proposing a new arm)
Use exactly these parameter keys; values must be double-quoted strings.
{example_param_schema_json}

## Task
Return ONLY the JSON object with fields {{reasoning, existing_arm_index, new_update_dict}}.
If selecting an existing arm, set existing_arm_index to its index and set new_update_dict to {{}}.
If proposing a new arm, set existing_arm_index to -1 and fill new_update_dict with your proposed parameter values.

## Output Format (example shape, not content)
{{
  "reasoning": "...",
  "existing_arm_index": -1,
  "new_update_dict": {{ "param_name": "string_value" }}
}}
""",
            },
        ]
        
        response_format = {"type": "json_object"}
        
        # Single LLM call with internal backoff handled by helper
        def llm_call():
            return self.llm(prompt_messages, response_format=response_format)

        llm_response = retry_with_exponential_backoff(
            llm_call,
            max_retries=10,
            base_delay=1.0,
            operation_name="LLM generation"
        )

        # Default fallback: return the best existing candidate's params
        default_update_dict = max(buffer, key=lambda c: c['mean_score'])['params'] if buffer else {}

        if llm_response is None:
            print_color("LLM call failed after retries. Return the candidate with the highest score.", "yellow")
            return default_update_dict

        llm_response_str = getattr(getattr(llm_response, 'choices', [{}])[0], 'message', None)
        llm_response_str = getattr(llm_response_str, 'content', None)
        if not llm_response_str:
            print_color("LLM returned an empty response.", "yellow")
            return default_update_dict

        cleaned_llm_response_str = llm_response_str.strip()
        try:
            llm_output = json.loads(cleaned_llm_response_str)
        except json.JSONDecodeError:
            print_color("Failed to parse LLM JSON output.", "yellow")
            return default_update_dict

        if not isinstance(llm_output, dict):
            return default_update_dict

        # Extract fields
        existing_arm_index = llm_output.get("existing_arm_index", -1)
        try:
            existing_arm_index = int(existing_arm_index)
        except Exception:
            existing_arm_index = -1
        proposed_new_update_raw = llm_output.get("new_update_dict", {})
        if not isinstance(proposed_new_update_raw, dict):
            proposed_new_update_raw = {}

        # Decide what to return
        if isinstance(existing_arm_index, int) and 0 <= existing_arm_index < len(prompt_candidates):
            # Return the existing arm's update_dict
            selected_entry = prompt_candidates[existing_arm_index]
            if verbose:
                print_color(f"LLM selected existing arm index {existing_arm_index}", "green")
            return selected_entry['params']
        elif existing_arm_index == -1 and len(proposed_new_update_raw) > 0:
            # Construct and return the new arm's update_dict
            try:
                candidate_params_dict = construct_update_dict(proposed_new_update_raw)
                if verbose:
                    print_color("LLM proposed a new arm", "green")
                return candidate_params_dict
            except Exception as e:
                print_color(f"Error constructing new_update_dict: {e}", "yellow")
                return default_update_dict
        else:
            # Fallback to best existing
            if verbose:
                print_color("LLM output incomplete; falling back to best existing arm", "yellow")
            return default_update_dict