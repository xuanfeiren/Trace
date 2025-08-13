# BAI_algorithms.py
# This file contains the algorithms for the Best Candidate Identification (BAI) problem.
# For all BAI algorithm classes, we should have a buffer (candidate-score pairs) as input, and output the best candidate after every evaluation iteration.
# Here are several algorithms. 
# 1. Evenly split
# 2. UCB best candidate identification
# 3. LLM tabular model
# 4. LLM regression model (estimate the score of a candidate)+output the choice
# 5. LLM generator model
   
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
            # for k, v in candidate_entry['params'].items():
            #     # breakpoint()
            #     print_color(f"{k.py_name}", "blue")
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
        self.ucb_exploration_factor = 0.3 # Set the exploration factor for UCB
        self.horizon = None
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
            candidate_entry['lcb_score'] = candidate_entry['mean_score'] - (candidate_entry['ucb_score']-candidate_entry['mean_score'])
        return 
    
    def step(self, guide, validate_dataset, num_threads, **kwargs):
        """To make the number of evauation the same for different BAI algorithms, we set ucb horiso to be number of candidates"""
        if self.horizon is None:
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


class LLMModel(UCBAlgorithm):
    """LLM selector best candidate identification."""
    """Default to be a tabular model"""
    def __init__(self,agent,num_threads, logger,update_dicts, *args, **kwargs):
        super().__init__(agent,num_threads, logger,update_dicts, *args, **kwargs)
        self.llm_model = "gemini/gemini-2.0-flash"
        self.llm = LLM(model=self.llm_model)
        # Add tracking for budget management
        self.selection_count = 0

    def select_candidate(self, buffer):
        """At each select step, we call LLM with the buffer statistics (candidate-score pairs). """
        self.update_buffer_scores()
        selected_entry = self.llm_generate_candidate(buffer, verbose=True) # Modify the verbose value here
        self.selection_count += 1  # Track how many times this method is called.        
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
        # if verbose:
        #     # Print full system prompt and truncated user messages
        #     print_color("=== LLM Prompt ===", "cyan")
        #     for i, msg in enumerate(prompt_messages):
        #         role = msg["role"]
        #         if role == "system":
        #             # Print full system prompt
        #             print_color(f"=== SYSTEM MESSAGE ===", "cyan")
        #             print_color(msg["content"], "cyan")
        #         else:
        #             # Truncate user messages (they contain long parameter data)
        #             content_preview = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
        #             print_color(f"=== USER MESSAGE (truncated) ===", "cyan")
        #             print_color(content_preview, "cyan")
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
    
class LLMRegressionModel(LLMModel):
    """LLM regression model. Could estimate the score of candidates in the buffer or not. Output a choice from the current buffer"""
    def __init__(self, agent, num_threads, logger, update_dicts, enable_estimate_scores=False, *args, **kwargs):
        super().__init__(agent, num_threads, logger, update_dicts, *args, **kwargs)
        self.enable_estimate_scores = enable_estimate_scores
        self.num_tools = None
        
    def llm_generate_candidate(self, buffer, verbose: bool = False):
        "Main function used by the BAI algorithm."
        return self.llm_regressor(buffer, verbose)
        
    def llm_regressor(self, buffer, verbose: bool = False):
        """
        The LLM will be given the buffer statistics and the candidate parameters. In this model, LLM serves as a function approximator/regressor, which means it will take the buffer statistics and the candidate parameters as input, and acts as an estimated score/reward model.

        Input:
        - buffer: list of candidate entries
        - verbose: whether to print verbose output

        Output:
        - one entry **in** the buffer
        """
        
        # Calculate budget information
        total_budget = self.num_epochs * self.horizon
        used_budget = self.selection_count
        remaining_budget = total_budget - used_budget
        
        # Prepare serializable candidate summaries with parameters
        serializable_candidate_summaries = []
        for idx, cand_entry in enumerate(buffer):
            summary = {
                "index": idx,
                "parameters": {p.py_name: copy.deepcopy(p.data) for p in cand_entry['params']},
                "eval_count": cand_entry['eval_count'],
                "mean_score": cand_entry['mean_score'],
                "ucb_score": cand_entry.get('ucb_score', None),
                "lcb_score": cand_entry.get('lcb_score', None)
            }
            serializable_candidate_summaries.append(summary)
        candidate_summaries_json = json.dumps(serializable_candidate_summaries, indent=2)
        
        example_param_schema_json = json.dumps({p.py_name: copy.deepcopy(p.data) for p in self.agent.parameters()}, indent=2)

        # Create conditional example output format
        if self.enable_estimate_scores:
            example_format = '''{{
  "buffer_analysis": "Parameter-Performance Learning: Analyzed 5 candidates (3 evaluated, 2 unevaluated). Pattern Analysis: Candidates with detailed authentication instructions (>500 chars) score 0.15 higher on average. Tool descriptions with specific examples correlate with +0.12 score boost. Error handling emphasis adds +0.08. Statistical Reliability: Candidates 0,1 have reliable data (eval_count 25,30), candidate 2 has moderate data (eval_count 8), candidates 3,4 have no evaluation data. Learned Patterns: Authentication focus + detailed examples + error handling = high performance formula.",
  "score_estimates": {{
    "0": {{"predicted_score": 0.74, "reasoning": "Evaluated candidate with mean_score=0.75, eval_count=25 (reliable). Parameter analysis: comprehensive authentication instructions (650 chars), detailed tool examples, strong error handling. Matches high-performance pattern perfectly. Prediction close to observed due to reliability and excellent parameter quality."}},
    "1": {{"predicted_score": 0.68, "reasoning": "Evaluated candidate with mean_score=0.65, eval_count=30 (very reliable). Parameter analysis: moderate instructions (400 chars), basic tool descriptions, minimal error handling. Missing key high-performance patterns. Reliable statistics support this mid-range performance level."}},
    "2": {{"predicted_score": 0.79, "reasoning": "Evaluated candidate with mean_score=0.82, eval_count=8 (moderate reliability). Parameter analysis: excellent authentication focus (700+ chars), comprehensive examples, strong error handling protocols. Parameters match high-performance pattern strongly. Slight downward adjustment for moderate eval_count but parameters suggest genuine high performance."}},
    "3": {{"predicted_score": 0.71, "reasoning": "UNEVALUATED candidate (eval_count=0). Parameter analysis: good authentication instructions (580 chars), decent examples, some error handling. Similar to candidate 0 but slightly less comprehensive. Predicted score based on similarity to candidate 0 (0.74) with small penalty for less detailed examples. Confident prediction due to clear pattern match."}},
    "4": {{"predicted_score": 0.63, "reasoning": "UNEVALUATED candidate (eval_count=0). Parameter analysis: basic instructions (350 chars), minimal examples, no error handling focus. Similar parameter profile to candidate 1 (scored 0.68) but even less detailed. Predicted slightly lower than candidate 1 due to weaker parameter quality. Pattern suggests below-average performance."}}
  }},
  "selection_reasoning": "Predicted performance ranking: candidate 2 (0.79) > candidate 0 (0.74) > candidate 3 (0.71) > candidate 1 (0.68) > candidate 4 (0.63). Selection: candidate 2. Rationale: (1) Highest predicted score based on excellent parameter-performance match, (2) Moderate evaluation data (eval_count=8) provides some confidence but needs verification, (3) Strong parameter quality suggests genuine high performance rather than noise, (4) High information value - confirming this candidate would validate our parameter-performance learning model.",
  "selected_index": 2
}}'''
        else:
            example_format = '''{{
  "buffer_analysis": "Buffer Statistics: 5 candidates total. Observed scores: [0.75, 0.65, 0.85, 0.45, 0.88], eval_counts: [25, 30, 3, 2, 4]. Confidence Analysis: UCB scores [0.78, 0.68, 1.02, 0.72, 0.98], LCB scores [0.72, 0.62, 0.68, 0.18, 0.78], confidence widths [0.06, 0.06, 0.34, 0.54, 0.20]. Narrow intervals for candidates 0,1 (reliable), wide intervals for candidates 2,3,4 (high uncertainty). Reliability: candidates 0,1 are reliable (high eval_count, narrow confidence intervals), candidates 2,3,4 are unreliable (low eval_count, wide confidence intervals). Parameter Patterns: Candidates with longer and more detailed additional_instructions (>800 chars) tend to score higher. Candidates 0,2 have detailed tool descriptions with specific examples, while candidates 1,3,4 have generic descriptions. Authentication-focused instructions appear in higher-scoring candidates. Content analysis shows candidates 0,2 emphasize user verification and error handling, while candidates 1,3,4 lack specific guidance.",
  "selection_reasoning": "Confidence interval analysis: candidate 2 has wide uncertainty [0.68, 1.02] but excellent parameters, candidate 4 has moderate uncertainty [0.78, 0.98] but poor parameters, candidates 0,1 have narrow intervals indicating reliability. Decision factors: (1) Parameter quality: candidate 2 has excellent parameter patterns with detailed tool descriptions and comprehensive instructions, suggesting high potential, (2) Information value: candidate 2 has very wide confidence interval (0.34 width) indicating high uncertainty - substantial information gain from additional evaluation, (3) Risk assessment: candidate 2's UCB (1.02) shows high upside potential while LCB (0.68) shows acceptable downside, parameter quality supports optimistic outlook, (4) Budget efficiency analysis: With {remaining_budget} evaluations remaining (ample budget), can afford to resolve high-uncertainty, high-potential candidate. If budget were low (<10 remaining), would choose candidate 0 (narrow confidence interval, reliable). Rejected candidate 4 despite high confidence bounds [0.78, 0.98] because parameter analysis suggests disconnect between observed performance and parameter quality. Candidate 2's combination of wide confidence interval (high information value) and excellent parameters (high expected performance) makes it optimal choice.",
  "selected_index": 2
}}'''

        prompt_messages = [
            {
                "role": "system",
                "content": f"""
## Role
You are a score prediction model for retail customer service agent configurations. Your primary task is to learn parameter-performance relationships from the buffer statistics and predict scores for all candidates, including those without evaluation data.

## Core Capabilities
1. **Pattern Learning**: Identify which parameter characteristics correlate with high/low performance
2. **Score Prediction**: Predict scores for candidates based on their parameters
3. **Statistical Analysis**: Account for noise and confidence intervals in existing data
4. **Extrapolation**: Predict scores for new candidates by comparing their parameters to evaluated ones

## Input Data Analysis
You receive candidates with:
- **Parameters**: Configuration settings (tools descriptions, instructions)
- **Statistics**: Some candidates have mean_score, eval_count, UCB/LCB bounds
- **Missing Data**: Some candidates have no evaluation statistics (eval_count=0, mean_score=None)

## Learning Objectives
**Learn from evaluated candidates**:
- Which parameter patterns lead to higher scores?
- What content/style/structure works best?
- How do parameter characteristics correlate with performance?

**Apply to unevaluated candidates**:
- Compare their parameters to successful evaluated candidates
- Predict likely performance based on parameter similarity
- Estimate scores even without evaluation data

## Required Analysis Process

### Step 1: Parameter-Performance Pattern Learning
Analyze evaluated candidates to identify:
- **High-performing patterns**: What makes successful candidates work?
- **Low-performing patterns**: What characteristics lead to poor performance?
- **Content analysis**: Specific words, phrases, structures that correlate with scores
- **Length patterns**: How parameter length affects performance
- **Style patterns**: Detailed vs concise, formal vs conversational, etc.

### Step 2: Score Prediction for All Candidates
For each candidate (both evaluated and unevaluated):
- **Evaluated candidates**: Use statistics + parameter analysis to refine score estimates
- **Unevaluated candidates**: Predict scores based on parameter similarity to evaluated ones
- **Confidence assessment**: How confident are you in each prediction?
- **Reasoning**: Explain your prediction based on learned patterns

### Step 3: Candidate Selection
Choose the candidate most likely to have the highest true performance:
- **Predicted performance**: Which candidate has the highest predicted score?
- **Confidence level**: How reliable is your prediction?
- **Information value**: Which candidate would provide most learning value?

## Prediction Strategy for Unevaluated Candidates
When predicting scores for candidates with no statistics:
1. **Find similar evaluated candidates**: Which evaluated candidates have similar parameters?
2. **Identify key differences**: How do the parameters differ from similar evaluated ones?
3. **Apply learned patterns**: Based on your pattern analysis, would these differences improve or hurt performance?
4. **Predict score**: Estimate a score based on similarity and pattern analysis
5. **Justify prediction**: Explain your reasoning clearly

{'## Output Requirements (With Score Estimation)' if self.enable_estimate_scores else '## Output Requirements (Selection Only)'}
Return ONLY a JSON object with these fields:
- "buffer_analysis": string analyzing parameter-performance patterns and statistical reliability. Candidates with eval_count > {2*len(self.validate_dataset['inputs'])} are considered well-evaluated with reliable statistics.
{'- "score_estimates": object mapping candidate indices to predicted scores with detailed reasoning' if self.enable_estimate_scores else ''}
- "selection_reasoning": string explaining your candidate choice based on predicted performance
- "selected_index": integer index of the candidate you select for next evaluation

## Example Output Format
{example_format}
""",
            },
            {
                "role": "user", 
                "content": f"""
## Candidate Data
{candidate_summaries_json}

## Parameter Schema
{example_param_schema_json}

## Task
Analyze the parameter-performance patterns and select the most promising candidate for evaluation. Focus on data-driven patterns rather than domain assumptions.

Return ONLY the JSON object with your analysis and selection.
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
            operation_name="LLM regression"
        )
        # Default fallback: return the best existing candidate
        # only consider candidates with scores
        default_entry = max([c for c in buffer if c['eval_count'] > 0], key=lambda c: c['mean_score'])

        if llm_response is None:
            if verbose:
                print_color("LLM regression call failed after retries. Returning highest scoring candidate.", "yellow")
            return default_entry

        llm_response_str = getattr(getattr(llm_response, 'choices', [{}])[0], 'message', None)
        llm_response_str = getattr(llm_response_str, 'content', None)
        if not llm_response_str:
            if verbose:
                print_color("LLM returned an empty response.", "yellow")
            return default_entry

        cleaned_llm_response_str = llm_response_str.strip()
        
        if verbose:
            self.print_buffer_statistics()
            print_color(f"LLM Regression response: {cleaned_llm_response_str}", "cyan")
            
        try:
            llm_output = json.loads(cleaned_llm_response_str)
        except json.JSONDecodeError:
            if verbose:
                print_color("Failed to parse LLM regression JSON output.", "yellow")
            return default_entry

        if not isinstance(llm_output, dict):
            return default_entry

        # Extract selection
        selected_index = llm_output.get("selected_index", -1)
        try:
            selected_index = int(selected_index)
        except Exception:
            selected_index = -1

        # Validate and return selection
        if isinstance(selected_index, int) and 0 <= selected_index < len(buffer):
            selected_entry = buffer[selected_index]
            if verbose:
                buffer_analysis = llm_output.get("buffer_analysis", "No analysis provided")
                selection_reasoning = llm_output.get("selection_reasoning", "No reasoning provided")
                
                print_color(f"LLM regression selected candidate {selected_index}", "green")
                # print_color(f"Buffer Analysis: {buffer_analysis}", "cyan")
                # print_color(f"Selection Reasoning: {selection_reasoning}", "cyan")
                
                # if self.enable_estimate_scores:
                #     score_estimates = llm_output.get("score_estimates", {})
                #     print_color(f"Score Estimates: {score_estimates}", "blue")
            return selected_entry
        else:
            if verbose:
                print_color("LLM regression output invalid; falling back to best existing candidate", "yellow")
            return default_entry
        
class LLMGenerator(LLMRegressionModel):
    "Ask LLM to come up with more candidates."
        
    def llm_generator(self, buffer, verbose: bool = False, num_to_generate: int = 1):
        """
        Ask LLM to come up with more candidates. Based on the current buffer with statistics.

        Input:
        - buffer: list of candidate entries
        - verbose: whether to print verbose output
        - num_to_generate: number of candidates to generate

        Output:
        - return a temporary buffer, containing current candidates and newly proposed candidates
        """
        # Delete candidates without scores
        buffer = [c for c in buffer if c['eval_count'] > 0]
        
        temporary_buffer = buffer
        
        # Prepare serializable candidate summaries with parameters and statistics
        serializable_candidate_summaries = []
        self.update_buffer_scores()
        for idx, cand_entry in enumerate(buffer):
            summary = {
                "index": idx,
                "parameters": {p.py_name: copy.deepcopy(p.data) for p in cand_entry['params']},
                "eval_count": cand_entry['eval_count'],
                "mean_score": cand_entry['mean_score'],
                "ucb_score": cand_entry.get('ucb_score', None),
                "lcb_score": cand_entry.get('lcb_score', None)
            }
            serializable_candidate_summaries.append(summary)
        candidate_summaries_json = json.dumps(serializable_candidate_summaries, indent=2)
        
        example_param_schema_json = json.dumps({p.py_name: copy.deepcopy(p.data) for p in self.agent.parameters()}, indent=2)

        prompt_messages = [
            {
                "role": "system",
                "content": f"""
## Role
You are an expert in generating diverse, high-performance retail customer service agent configurations. Based on current candidate performance data, you will generate {num_to_generate} new diverse candidates that could potentially outperform existing ones.

## Task
Analyze the current buffer of candidates and their performance statistics, then generate {num_to_generate} new diverse candidates with different approaches that could achieve better performance.

## Current Buffer Analysis
You have access to:
1. **Candidate parameters**: Configuration settings (tools_info, additional_instructions)
2. **Performance statistics**: mean_score, eval_count, UCB/LCB confidence bounds
3. **Patterns**: What seems to work well vs poorly in current candidates

## Diversity Requirements
**CRITICAL**: Generated candidates MUST be diverse from each other and from existing candidates:
- **Different approaches**: Vary the style, focus, and structure significantly
- **Different strengths**: Target different aspects of customer service (authentication, error handling, workflow efficiency, etc.)
- **Different philosophies**: Some detailed vs concise, some conservative vs aggressive, some structured vs flexible
- **Avoid redundancy**: Don't generate similar candidates

## Generation Strategy
For each new candidate:
1. **Identify gaps**: What weaknesses exist in current candidates?
2. **Propose improvements**: How can this new candidate address those gaps?
3. **Ensure diversity**: How is this candidate meaningfully different from others?
4. **Justify potential**: Why might this candidate perform better?

## Output Requirements
Return ONLY a JSON object with these fields:
- "buffer_analysis": string analyzing current candidates' strengths, weaknesses, and patterns
- "generated_candidates": array of {num_to_generate} objects, each with:
  - "reasoning": string explaining why this candidate might outperform existing ones and how it's diverse
  - "diversity_focus": string describing what makes this candidate unique/different
  - "parameters": object with parameter values (matching the schema)

**CRITICAL REQUIREMENT FOR tools_info**: The tools_info parameter MUST contain descriptions for ALL tools available in the system. You cannot provide partial tool sets or omit any tools. Every tool that exists in the current candidates must be included in your new proposals with updated descriptions. The tools_info should be a complete replacement, not a partial update.

**IMPORTANT CONSTRAINTS**:
- **ONLY modify tool descriptions**: You can only change the "description" field of existing tools
- **CANNOT add new tools**: Do not create tools that don't exist in the current system
- **CANNOT remove tools**: Every existing tool must be present in your proposals
- **CANNOT change tool names**: Tool names (function.name) must remain exactly the same
- **CANNOT change tool parameters**: The parameters schema for each tool must remain unchanged
- **ONLY change descriptions**: Focus on improving how tools are described to the agent

## Example Output Format
{{
  "buffer_analysis": "Current buffer shows candidates focusing heavily on authentication (scores 0.6-0.8) but lacking in error recovery and user guidance. Most candidates have verbose tool descriptions but inconsistent instruction styles. Gap: no candidates emphasize proactive user assistance or streamlined workflows.",
  "generated_candidates": [
    {{
      "reasoning": "Current candidates are verbose and reactive. This candidate focuses on efficiency and proactive assistance, which could reduce interaction time and improve user satisfaction. Addresses the gap in workflow optimization.",
      "diversity_focus": "Efficiency-first approach with proactive user guidance, contrasting with existing reactive verbose style",
      "parameters": {{
        "list0": "Concise, action-focused tool descriptions emphasizing speed and efficiency...",
        "str0": "Prioritize quick resolution and minimal back-and-forth. Always suggest next steps proactively..."
      }}
    }},
    {{
      "reasoning": "Existing candidates lack robust error handling. This candidate specializes in error recovery and provides multiple fallback options, potentially improving success rates in complex scenarios.",
      "diversity_focus": "Error-resilience specialist with comprehensive fallback strategies, unique focus on failure recovery",
      "parameters": {{
        "list0": "Detailed tool descriptions with extensive error handling examples and fallback procedures...",
        "str0": "Comprehensive error recovery protocols. When any tool fails, immediately provide alternatives..."
      }}
    }},
    {{
      "reasoning": "Current candidates assume user expertise. This candidate prioritizes user education and confirmation, potentially improving user satisfaction and reducing misunderstandings in complex transactions.",
      "diversity_focus": "Educational approach with emphasis on user understanding and confirmation, contrasts with assumption-heavy existing candidates",
      "parameters": {{
        "list0": "User-friendly tool descriptions with natural language explanations and examples...",
        "str0": "Explain every action in simple terms. Always confirm understanding before proceeding..."
      }}
    }}
  ]
}}
""",
            },
            {
                "role": "user",
                "content": f"""
## Current Buffer Data
{candidate_summaries_json}

## Parameter Schema
Use exactly these parameter keys; values must be strings:
{example_param_schema_json}

## Task
Generate {num_to_generate} diverse new candidates that could outperform existing ones. Focus on different approaches and address different weaknesses you identify in the current buffer.

Return ONLY the JSON object with your analysis and generated candidates.
""",
            },
        ]

        response_format = {"type": "json_object"}

        # LLM call with retry logic
        def llm_call():
            return self.llm(prompt_messages, response_format=response_format)

        llm_response = retry_with_exponential_backoff(
            llm_call,
            max_retries=10,
            base_delay=1.0,
            operation_name="LLM candidate generation"
        )
        #print the number of tokens in response
        # print(f"Number of tokens in output: {llm_response}")
        
        if llm_response is None:
            if verbose:
                print_color("LLM candidate generation failed after retries. Returning original buffer.", "yellow")
            return temporary_buffer

        llm_response_str = getattr(getattr(llm_response, 'choices', [{}])[0], 'message', None)
        llm_response_str = getattr(llm_response_str, 'content', None)
        if not llm_response_str:
            if verbose:
                print_color("LLM returned empty response for candidate generation.", "yellow")
            return temporary_buffer

        cleaned_llm_response_str = llm_response_str.strip()

        if verbose:
            print_color(f"LLM Generator response: {cleaned_llm_response_str}", "cyan")

        try:
            llm_output = json.loads(cleaned_llm_response_str)
        except json.JSONDecodeError:
            if verbose:
                print_color("Failed to parse LLM generator JSON output.", "yellow")
            return temporary_buffer

        if not isinstance(llm_output, dict):

            return temporary_buffer

        # Extract and process generated candidates
        generated_candidates = llm_output.get("generated_candidates", [])
        buffer_analysis = llm_output.get("buffer_analysis", "No analysis provided")

        if verbose:
            print_color(f"Buffer Analysis: {buffer_analysis}", "cyan")
            print_color(f"Generated {len(generated_candidates)} new candidates", "green")

        # Convert generated candidates to buffer entries
        for i, candidate_data in enumerate(generated_candidates):
            try:
                parameters_raw = candidate_data.get("parameters", {})
                reasoning = candidate_data.get("reasoning", "No reasoning provided")
                diversity_focus = candidate_data.get("diversity_focus", "No diversity focus provided")
                # Validate tools_info
                tools_info_data = None
                # Find any list parameter in parameters_raw
                for key, value in parameters_raw.items():
                    if isinstance(value, list):
                        tools_info_data = value
                        break
                
                if self.num_tools is None:
                    params_dict = buffer[0]['params']
                    for param_node, param_value in params_dict.items():
                        if isinstance(param_value, list):
                            self.num_tools = len(param_value)
                            break
                    
                # check whether tools_info contains the same number of tools as the first candidate in the buffer
                if tools_info_data:
                    if len(tools_info_data) != self.num_tools:
                        if verbose:
                            print_color(f"Skipping candidate {i+1}: tools_info contains {len(tools_info_data)} tools, expected {self.num_tools}", "yellow")
                        continue
                # Convert parameters to the correct format
                candidate_params_dict = self.construct_update_dict(parameters_raw)

                # Create new buffer entry
                new_candidate_entry = {
                    "params": candidate_params_dict,
                    "score_sum": 0.0,
                    "eval_count": 0,
                    "mean_score": None,
                    "ucb_score": None,
                    "lcb_score": None
                }

                temporary_buffer.append(new_candidate_entry)

                if verbose:
                    print_color(f"Generated candidate {i+1}:", "blue")
                    print_color(f"  Reasoning: {reasoning}", "blue")
                    print_color(f"  Diversity Focus: {diversity_focus}", "blue")

            except Exception as e:
                if verbose:
                    print_color(f"Error processing generated candidate {i+1}: {e}", "yellow")
                continue

        if verbose:
            print_color(f"Temporary buffer size: {len(temporary_buffer)} (original: {len(buffer)}, added: {len(temporary_buffer) - len(buffer)})", "green")

        return temporary_buffer
        
    def llm_generate_candidate(self, buffer, verbose: bool = False): 
        # Add new candidates to the buffer
        # Every time we call this function, it will delete candidates without scores first
        self.buffer = self.llm_generator(buffer, verbose = False, num_to_generate=1)

        # Use the LLM regression model to select the best candidate from the buffer
        return self.llm_regressor(self.buffer, verbose)

    