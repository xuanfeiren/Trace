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

from typing import Union, List, Tuple, Dict, Any, Optional
from collections import deque
from opto.utils.llm import LLM # For the selector LLM
from opto.trace.nodes import ParameterNode
import json
import warnings
from black import format_str, FileMode
import random
import math
from opto.trainer.utils import sample_minibatch, retry_with_exponential_backoff, evaluate_agent

def set_parameters_for_agent(agent, update_dict):
    """Set the parameters for the agent."""
    for node in agent.parameters():
        if node.trainable and node in update_dict:
            node._set(update_dict[node])
    return agent

class BAIAlgorithmBase(AlgorithmBase):
    """We define a base class for all BAI algorithms.
    The input is a list of update_dicts, which is a list of dictionaries that contain the update information.
    The output is the best candidate after every evaluation iteration.
    """
    def __init__(self,agent,num_threads, logger,update_dicts, *args, **kwargs):
        super().__init__(agent,num_threads, logger, *args, **kwargs)
        self.update_dicts = update_dicts
        self.total_samples = 0

    def print_buffer_statistics(self):
        self.update_buffer_scores()
        for i,candidate_entry in enumerate(self.buffer):
            print_color(f"Candidate {i}. Mean score {candidate_entry['mean_score']}, eval_count {candidate_entry['eval_count']}", "blue")
        return 
    
    def update_buffer_scores(self):
        """Update the buffer statistics."""
        for candidate_entry in self.buffer:
            candidate_entry['mean_score'] = candidate_entry['score_sum'] / (candidate_entry['eval_count'] or 1E-9)
        return 
    
    def train(self, guide, validate_dataset, test_dataset,num_threads, num_epochs,eval_frequency, **kwargs):
        """The training process of BAI Algorithm."""
        # 1. Initialize the buffer. For each update_dict, evaluate the agent once and add the candidate-score pair to the buffer.
        self.buffer = deque(maxlen=100)
        self.validate_dataset = validate_dataset
        print_color(f"Initializing Evaluation", "green")
        for update_dict in self.update_dicts:   
            # Evaluate one candidate and update the buffer statistics.
            set_parameters_for_agent(self.agent, update_dict)
            score = evaluate_agent(self.agent, guide, validate_dataset, num_threads=num_threads, num_eval_times=1)
            eval_count = len(validate_dataset['inputs'])
            self.total_samples += eval_count
            candidate_entry = {"params": update_dict, "score_sum": score*eval_count, "eval_count": eval_count}
            self.buffer.append(candidate_entry)
        # 2. Best candidate identification. At each epoch, we will do some evaluation using sample budget, update the buffer statistics, and output the best candidate for test.
        self.print_buffer_statistics()
        self.num_epochs = num_epochs
        for epoch in range(num_epochs):

            print_color(f"Epoch {epoch+1}/{num_epochs}", "blue")
            # Evaluate the best candidate at this epoch and update the buffer statistics.
            best_candidate_at_this_epoch, used_sample_budget = self.step(guide, validate_dataset, num_threads, **kwargs)
            self.total_samples += used_sample_budget
            
            if epoch % eval_frequency == 0:
                # Test the current best candidate.
                self.print_buffer_statistics()
                set_parameters_for_agent(self.agent, best_candidate_at_this_epoch['params'])
                test_score = evaluate_agent(self.agent, guide, test_dataset, num_threads=num_threads, num_eval_times=5)
                self.logger.log("Test score", test_score,epoch+1,color='green')
        
    def step(self, guide, validate_dataset, num_threads, **kwargs):
        """The best candidate identification step. At each epoch, we will do some evaluation using sample budget, update the buffer statistics, and output the best candidate for test. The output should be the entry of the best candidate in the buffer and the used sample budget."""
        raise NotImplementedError("Subclasses must implement this method")

class EvenlySplitAlgorithm(BAIAlgorithmBase):
    """Evenly split the sample budget for each candidate."""

    def step(self, guide, validate_dataset, num_threads, **kwargs):
        # Evaluate each candidate once on the validation set, and update the buffer statistics.
        for candidate in self.buffer:
            set_parameters_for_agent(self.agent, candidate['params'])
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
        self.horizon = len(self.buffer)
        for iteration in range(self.horizon):
            print_color(f"Iteration {iteration+1}/{self.horizon}: ", 'blue')
            self.update_buffer_scores()
            selected_candidate = self.select_candidate(self.buffer)
            set_parameters_for_agent(self.agent, selected_candidate['params'])
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
        self.llm = LLM(model=self.llm_model)
        # Add tracking for budget management
        self.selection_count = 0

    def select_candidate(self, buffer):
        """At each select step, we call LLM with the buffer statistics (candidate-score pairs). """
        self.update_buffer_scores()
        self.selection_count += 1  # Track how many times this method is called
        selected_entry = self.llm_generate_candidate(buffer, verbose=True) # Modify the verbose value here
        
        # If a new candidate was proposed, add it to the buffer
        
        
        # Return in the expected format with 'params' key
        return selected_entry
        
    def llm_generate_candidate(self, buffer, verbose: bool = False): 
        """
        LLM can either select an existing candidate or generate a new candidate.
        Tries multiple times with exponential backoff until a valid output is parsed.
        Returns a single entry in the buffer.
        """
       
        # Calculate budget information
        total_budget = self.num_epochs * self.horizon
        used_budget = self.selection_count
        remaining_budget = total_budget - used_budget
        
        # Filter buffer to only include candidates with valid UCB scores
        # valid_candidates = [c for c in buffer if c.get('mean_score') is not None and c.get('mean_score') != -float('inf') and c.get('mean_score') != float('inf')]
    
        # sorted_buffer = sorted(valid_candidates, key=lambda c: c.get('mean_score', -float('inf')), reverse=True)
        serializable_candidate_summaries = []
        for idx, cand_entry in enumerate(buffer):
            summary = {
                "index": idx,
                "parameters":  {p.py_name: copy.deepcopy(p.data) for p in cand_entry['params']},
                "eval_count": cand_entry['eval_count'],
                "mean_score": cand_entry['mean_score']
            }
            serializable_candidate_summaries.append(summary)
        candidate_summaries_json = json.dumps(serializable_candidate_summaries, indent=2)
        
        example_param_schema_json = json.dumps({p.py_name: copy.deepcopy(p.data) for p in self.agent.parameters()}, indent=2)

        # Create budget-aware guidance
        budget_guidance = f"""
## Budget Management
You have {remaining_budget} evaluation choices remaining out of {total_budget} total budget. Use this information to balance exploration vs exploitation:

**General Strategy:**
- **Prioritize statistical reliability**: Before proposing new arms, ensure existing arms have sufficient evaluations to trust their scores
- **Early in the process (plenty of budget)**: Mix between getting reliable statistics on existing arms and proposing new arms for discovery
- **Later in the process (limited budget)**: Focus on getting more data on the most promising existing arms, only propose new arms if current candidates are clearly poor
- **Avoid the trap**: Don't constantly propose new arms leaving all candidates with very low eval_count, as this makes all statistics unreliable
- **Balance approach**: Aim for a mix where some arms get multiple evaluations for reliability, while still exploring new parameter combinations"""

        prompt_messages = [
            {
                "role": "system",
                "content": f"""
## Role
You are assisting with best-arm identification for optimizing a retail customer service agent in the tau-bench framework. You will see candidate arms (agent configurations) and their performance statistics. Your task is to choose either **selecting an existing arm** or **proposing a new arm** to be evaluated on the validation dataset next.

## Problem Context
You are optimizing a retail customer service agent by modifying two key parameters:
1. **tools_info**: Descriptions of tools the agent can use (e.g., cancel_pending_order, get_user_details, modify_pending_order_items). Better descriptions help the agent use tools correctly and avoid errors.
2. **additional_instructions**: Strategic guidance and best practices for retail customer service (e.g., authentication procedures, confirmation workflows, error handling).

The agent handles tasks like order cancellations, modifications, returns, exchanges, and user inquiries. Performance is measured by task success rate on retail scenarios.

## Key Retail Domain Guidelines
- Users must be authenticated via email or name+zip before any actions
- Pending orders can be cancelled/modified; delivered orders can be returned/exchanged  
- Consequential actions require explicit user confirmation
- Payment methods include gift cards, PayPal, and credit cards
- Each product has multiple item variants (color, size, etc.)

{budget_guidance}

## Budget Information
- Each unit of budget is one evaluation on the validation dataset, meaning that could lead to {len(self.validate_dataset['inputs'])} more evaluations.
- Total evaluation budget: {total_budget}
- Budget used so far: {used_budget}
- Remaining budget: {remaining_budget}
- Selection number: {self.selection_count}/{total_budget}

## Decision Objective
Choose the next arm to evaluate to maximize the chance of discovering the best agent configuration. You need to balance two important considerations:

## How to Reason Through Your Decision
Before making your choice, you should systematically analyze the situation:

**Step 1: Analyze Current Buffer**
- Look at each arm's mean_score and eval_count
- Identify which arms have reliable statistics (higher eval_count) vs unreliable (low eval_count)
- Identify the current best-performing arms and worst-performing arms

**Step 2: Decide: Existing Arms vs New Arm (Consider Budget)**
- **Choose existing arms if**: Some arms have low eval_count (unreliable statistics) OR promising arms need more data for confirmation
- **Choose new arm if**: All arms have sufficient evaluations AND their performance is consistently poor/mediocre
- **Budget consideration**: With limited budget, prioritize getting reliable data on promising arms. With ample budget, you can afford to explore new approaches.

**Step 3A: If Choosing Existing Arm - Which One?**
- Prioritize arms with promising scores but low eval_count (need reliability)
- Consider arms with moderate scores that might improve with more data
- Explain why this specific arm deserves more evaluation

**Step 3B: If Choosing New Arm - How to Improve?**
- Analyze what's wrong with current arms: Are tool descriptions unclear? Are instructions missing key guidance? Is there a better way to express it?
- Identify specific weaknesses: Authentication issues? Error handling problems? Workflow confusion?
- Propose concrete improvements: Better tool descriptions, clearer instructions, addressing observed failure patterns

**Step 4: Final Risk Assessment**
- What's the risk/reward of your specific choice?
- Is this the best use of one evaluation budget?

Write out your reasoning following these steps before stating your final decision.

**1. Statistical Reliability vs Discovery**
- **Selecting existing arms** adds more evaluation data, making their mean_score statistics more reliable and trustworthy. Arms with low eval_count have unreliable scores that could be misleading due to noise.
- **Proposing new arms** explores new parameter combinations that might perform better than current candidates, but starts with no evaluation data.

**2. Key Trade-off to Consider**
- If existing arms have low eval_count, their scores are unreliable - you should often select them for more evaluations to get trustworthy statistics
- If existing arms have been evaluated many times and show consistently poor performance, you should propose new arms to find better candidates
- If you see promising arms with moderate eval_count, evaluate them more to confirm their quality before proposing new ones

**3. Evaluation Process**
- When you make a choice, the selected candidate will be evaluated on the validation dataset once
- This will increase that candidate's eval_count by {len(self.validate_dataset['inputs'])} and update its mean_score
- More evaluations make the mean_score more reliable and trustworthy

## What You May Do
- **Select an existing arm** by its provided index (copy its parameter dictionary exactly) - this will add one more evaluation to that arm's statistics
- **Propose a new arm** by setting existing_arm_index to -1 and providing new parameter values - this will start evaluating a completely new candidate

## Output Requirements
Return ONLY a JSON object with these fields (no prose, no markdown, no extra fields):
- "reasoning": detailed double-quoted string that follows the 4-step reasoning process above. Analyze the buffer, assess reliability, consider budget, then explain your choice.
- "existing_arm_index": integer index of an existing arm to evaluate next, OR **-1 if proposing a new arm**.
- "new_update_dict": JSON object mapping parameter names to string values; use {{}} if selecting an existing arm, otherwise fill with your proposed parameter values when existing_arm_index is -1.

**IMPORTANT**: To propose a new candidate, you MUST set existing_arm_index to -1 and provide the new parameter values in new_update_dict.

All keys and string values must use double quotes.

## Examples
Here are two examples of choices you can make. Note that the specific numbers and scenarios shown are just examples - you should analyze your actual buffer data and budget situation.

**To select existing arm 2:**
{{
  "reasoning": "Step 1: Buffer shows arm 0 (score 0.6, eval_count 50), arm 1 (score 0.4, eval_count 20), arm 2 (score 0.8, eval_count 10). Step 2: With 25 budget remaining, choose existing arm because arm 2 has promising score but unreliable statistics - need to verify before exploring new approaches. Step 3A: Selecting arm 2 because it has highest score but lowest eval_count - could be consistently good or just lucky. Step 4: This is the best use of one evaluation since confirming the most promising candidate reduces risk.",
  "existing_arm_index": 2,
  "new_update_dict": {{}}
}}

**To propose a new arm:**
{{
  "reasoning": "Step 1: Buffer has 3 arms with scores 0.3, 0.35, 0.32, all with eval_count > 50 - reliable but poor performance. Step 2: With 20 budget remaining (ample), choose new arm because all current arms have sufficient evaluation but consistently poor results - worth exploring new approaches. Step 3B: Current arms seem to fail at user authentication - tool descriptions are vague about verification steps, and instructions don't emphasize the authentication requirement from the wiki. Proposing new arm with: clearer tool descriptions specifying authentication requirements, and explicit instructions about user verification being mandatory before any actions. Step 4: This is good use of evaluation budget since current arms are clearly insufficient and new approach targets identified weakness.",
  "existing_arm_index": -1,
  "new_update_dict": {{ "tools_info": "[detailed tool descriptions with explicit authentication requirements and verification steps]", "additional_instructions": "CRITICAL: Always authenticate users via email or name+zip before any actions. This is mandatory per retail policy." }}
}}
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
        if verbose:
            # Print full system prompt and truncated user messages
            print_color("=== LLM Prompt ===", "cyan")
            for i, msg in enumerate(prompt_messages):
                role = msg["role"]
                if role == "system":
                    # Print full system prompt
                    print_color(f"=== SYSTEM MESSAGE ===", "cyan")
                    print_color(msg["content"], "cyan")
                else:
                    # Truncate user messages (they contain long parameter data)
                    content_preview = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
                    print_color(f"=== USER MESSAGE (truncated) ===", "cyan")
                    print_color(content_preview, "cyan")
        llm_response = retry_with_exponential_backoff(
            llm_call,
            max_retries=10,
            base_delay=1.0,
            operation_name="LLM generation"
        )

        # Default fallback: return the best existing candidate's params
        default_entry = max(buffer, key=lambda c: c['mean_score'])

        if llm_response is None:
            print_color("LLM call failed after retries. Return the candidate with the highest score.", "yellow")
            return default_entry

        llm_response_str = getattr(getattr(llm_response, 'choices', [{}])[0], 'message', None)
        llm_response_str = getattr(llm_response_str, 'content', None)
        if not llm_response_str:
            print_color("LLM returned an empty response.", "yellow")
            return default_entry

        cleaned_llm_response_str = llm_response_str.strip()
    
        self.print_buffer_statistics()
        print_color(f"LLM response: {cleaned_llm_response_str}")
        try:
            llm_output = json.loads(cleaned_llm_response_str)
        except json.JSONDecodeError:
            print_color("Failed to parse LLM JSON output.", "yellow")
            return default_entry

        if not isinstance(llm_output, dict):
            return default_entry

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
        if isinstance(existing_arm_index, int) and 0 <= existing_arm_index < len(buffer):
            # Return the existing arm's update_dict
            selected_entry = buffer[existing_arm_index]
            if verbose:
                print_color(f"LLM selected existing arm index {existing_arm_index}", "green")
            return selected_entry
        elif existing_arm_index == -1 and len(proposed_new_update_raw) > 0:
            # Construct and return the new arm as a buffer entry
            try:
                candidate_params_dict = self.construct_update_dict(proposed_new_update_raw)
                # Create a new buffer entry for the new candidate
                new_candidate_entry = {
                    "params": candidate_params_dict,
                    "score_sum": 0.0,
                    "eval_count": 0
                }
                if new_candidate_entry not in buffer:
                    buffer.append(new_candidate_entry)
                    print_color(f"Added new candidate to buffer. Buffer size: {len(buffer)}", "cyan")
                if verbose:
                    print_color("LLM proposed a new arm", "green")
                return new_candidate_entry
            except Exception as e:
                print_color(f"Error constructing new_update_dict: {e}", "yellow")
                return default_entry
        else:
            # Fallback to best existing
            if verbose:
                print_color("LLM output incomplete; falling back to best existing arm", "yellow")
            return default_entry

    def construct_update_dict(self, suggestion: Dict[str, Any]) -> Dict[ParameterNode, Any]:
        """Convert the suggestion in text into the right data type using this agent's parameter nodes."""
        update_dict: Dict[ParameterNode, Any] = {}
        for node in self.agent.parameters():
            if node.trainable:
                if node.py_name in suggestion:
                    try:
                        formatted_suggestion = suggestion[node.py_name]
                        if isinstance(formatted_suggestion, str) and 'def' in formatted_suggestion:
                            formatted_suggestion = format_str(formatted_suggestion, mode=FileMode())
                        update_dict[node] = type(node.data)(formatted_suggestion)
                    except (ValueError, KeyError) as e:
                        if getattr(self, 'ignore_extraction_error', False):
                            warnings.warn(
                                f"Cannot convert the suggestion '{suggestion[node.py_name]}' for {node.py_name} to the right data type"
                            )
                        else:
                            raise e
                else:
                    update_dict[node] = node.data
        return update_dict
    