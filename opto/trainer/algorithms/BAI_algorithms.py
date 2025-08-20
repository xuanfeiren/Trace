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
        self.epoch = 0
        for epoch in range(num_epochs):
            self.epoch = epoch
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
                self.logger.log("Validate score", best_candidate_at_this_epoch['mean_score'],epoch+1,color='green')
        
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
        # Select the best candidate according to the buffer statistics.
        self.update_buffer_scores()
        candidate_to_test = max(self.buffer, key=lambda x: x['mean_score'])
        return candidate_to_test, len(self.buffer)*len(validate_dataset['inputs'])
    
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
                "parameters":  {k.py_name: v for k,v in cand_entry['params'].items()},
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
You are assisting with best-arm identification for optimizing tau-bench agents. You will see candidate arms (agent configurations) and their performance statistics. Your task is to choose either **selecting an existing arm** or **proposing a new arm** to be evaluated on the validation dataset next.

## Problem Context
You are optimizing tool-calling agents for tau-bench environments (airline and retail). These agents help users complete complex multi-step tasks like flight bookings, order management, returns, and exchanges. Agent performance is measured by task success rate in realistic user interaction scenarios.

## Key Optimization Areas
- **Tool Usage**: Agents must select appropriate tools and use them with correct parameters
- **User Communication**: Clear, helpful interactions that confirm actions and handle edge cases
- **Domain Compliance**: Following business rules (authentication, verification, policy adherence)
- **Error Recovery**: Graceful handling of failures with alternative solutions
- **Task Completion**: Successfully finishing user requests without unintended actions

## Agent Parameters
Agents have configurable parameters like:
- **tools_info**: Descriptions that help agents understand when and how to use each tool
- **additional_instructions**: Strategic guidance for handling different scenarios and edge cases

Better parameter configurations lead to higher task success rates across diverse user scenarios.

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
  "reasoning": "Step 1: Buffer has 3 arms with scores 0.3, 0.35, 0.32, all with eval_count > 50 - reliable but poor performance. Step 2: With 20 budget remaining (ample), choose new arm because all current arms have sufficient evaluation but consistently poor results - worth exploring new approaches. Step 3B: Current arms seem to fail at key requirements - parameter descriptions are vague about important steps, and instructions don't emphasize critical requirements. Proposing new arm with: clearer parameter descriptions specifying key requirements, and explicit instructions about important procedures. Step 4: This is good use of evaluation budget since current arms are clearly insufficient and new approach targets identified weakness.",
  "existing_arm_index": -1,
  "new_update_dict": {{ "param1": "[detailed parameter descriptions with explicit requirements and verification steps]", "param2": "CRITICAL: Always follow key procedures and requirements. This is mandatory per domain policy." }}
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
    def __init__(self, agent, num_threads, logger, update_dicts, enable_estimate_scores=False, domain_context=None, *args, **kwargs):
        super().__init__(agent, num_threads, logger, update_dicts, *args, **kwargs)
        self.enable_estimate_scores = enable_estimate_scores
        self.num_tools = None
        
        # Set domain context - can be overridden for specific applications
        if domain_context is None:
            self.domain_context = """## Problem Context and Domain Knowledge
You are a score prediction model for tau-bench agent configurations. You are optimizing agents for tool-agent-user interaction in real-world domains (airline and retail environments).

**Core Optimization Task:**
- **Agent Type**: Tool-calling agents that help users complete complex multi-step tasks
- **Parameters**: Agents have configurable parameters like tools_info (tool descriptions) and additional_instructions (strategic guidance)
- **Performance Metric**: Success rate on completing user tasks correctly within the domain constraints
- **Environments**: Airline (flight bookings, cancellations, changes) and Retail (orders, returns, exchanges)

**Key Success Factors:**
- **Tool Usage**: Agents must use the right tools at the right time with correct parameters
- **User Interaction**: Effective communication and confirmation of actions with users
- **Domain Constraints**: Following business rules (authentication requirements, policy compliance)
- **Error Handling**: Graceful recovery from failures and providing alternative solutions
- **Workflow Efficiency**: Completing tasks with minimal back-and-forth while being thorough

**Common Failure Modes:**
- Using wrong tools or incorrect tool parameters
- Missing critical authentication or verification steps
- Poor user communication leading to misunderstandings
- Incomplete task completion or taking unintended actions
- Not following domain-specific business rules and constraints

**Optimization Strategy:**
Better parameter configurations lead to higher task success rates. The goal is to find parameter settings that maximize agent performance across diverse scenarios in the target domain."""
        else:
            self.domain_context = domain_context
        
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
                "parameters": {k.py_name: v for k,v in cand_entry['params'].items()},
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
  "buffer_analysis": "Parameter-Performance Learning: Analyzed 5 candidates with varying evaluation data. Pattern Analysis: Candidates with detailed parameter content (>500 chars) show higher performance trends. Parameter descriptions with specific examples correlate with better outcomes (+0.12 average boost). Comprehensive structure emphasis adds performance value (+0.08). Statistical Patterns: Higher eval_count candidates show more reliable score patterns, but parameter quality remains primary predictor across all candidates.",
  "score_estimates": {{
    "0": {{"reasoning": "Parameter analysis: comprehensive parameter content (650 chars), detailed descriptions, strong structure. Current data shows mean_score=0.75 with eval_count=25. Parameter quality strongly matches high-performance patterns. Predicted score reflects excellent parameter characteristics with confidence from existing data.", "predicted_score": 0.74}},
    "1": {{"reasoning": "Parameter analysis: moderate parameter content (400 chars), basic descriptions, minimal detail. Current data shows mean_score=0.65 with eval_count=30. Parameter patterns suggest mid-range performance, consistent with observed data. Missing key high-performance characteristics.", "predicted_score": 0.68}},
    "2": {{"reasoning": "Parameter analysis: excellent parameter focus (700+ chars), comprehensive examples, strong structure. Current data shows mean_score=0.82 with eval_count=8. Parameter quality strongly indicates high performance potential. Prediction based on strong parameter-performance correlation patterns.", "predicted_score": 0.79}},
    "3": {{"reasoning": "Parameter analysis: good parameter content (580 chars), decent examples, some structure. Limited evaluation data (eval_count=0) but parameter patterns similar to high-performing candidates. Predicted score based on parameter similarity analysis and learned performance patterns.", "predicted_score": 0.71}},
    "4": {{"reasoning": "Parameter analysis: basic parameter content (350 chars), minimal examples, limited structure. No evaluation data yet (eval_count=0) but parameter patterns match lower-performing profiles. Predicted score reflects weaker parameter characteristics based on learned patterns.", "predicted_score": 0.63}}
  }},
  "selection_reasoning": "Predicted performance ranking based on parameter-performance patterns: candidate 2 (0.79) > candidate 0 (0.74) > candidate 3 (0.71) > candidate 1 (0.68) > candidate 4 (0.63). Selection: candidate 2. Rationale: (1) Highest predicted score based on strong parameter-performance correlation, (2) Excellent parameter quality indicators suggest genuine high performance, (3) Additional evaluation would confirm pattern-based prediction, (4) High information value for validating parameter-performance model.",
  "selected_index": 2
}}'''
        else:
            example_format = '''{{
  "buffer_analysis": "Buffer Statistics: 5 candidates total. Observed scores: [0.75, 0.65, 0.85, 0.45, 0.88], eval_counts: [25, 30, 3, 2, 4]. Confidence Analysis: UCB scores [0.78, 0.68, 1.02, 0.72, 0.98], LCB scores [0.72, 0.62, 0.68, 0.18, 0.78], confidence widths [0.06, 0.06, 0.34, 0.54, 0.20]. Narrow intervals for candidates 0,1 (reliable), wide intervals for candidates 2,3,4 (high uncertainty). Reliability: candidates 0,1 are reliable (high eval_count, narrow confidence intervals), candidates 2,3,4 are unreliable (low eval_count, wide confidence intervals). Parameter Patterns: Candidates with longer and more detailed parameter content (>800 chars) tend to score higher. Candidates 0,2 have detailed parameter descriptions with specific examples, while candidates 1,3,4 have generic descriptions. Well-structured parameter content appears in higher-scoring candidates. Content analysis shows candidates 0,2 emphasize comprehensive details and clear structure, while candidates 1,3,4 lack specific guidance.",
  "selection_reasoning": "Confidence interval analysis: candidate 2 has wide uncertainty [0.68, 1.02] but excellent parameters, candidate 4 has moderate uncertainty [0.78, 0.98] but poor parameters, candidates 0,1 have narrow intervals indicating reliability. Decision factors: (1) Parameter quality: candidate 2 has excellent parameter patterns with detailed descriptions and comprehensive content, suggesting high potential, (2) Information value: candidate 2 has very wide confidence interval (0.34 width) indicating high uncertainty - substantial information gain from additional evaluation, (3) Risk assessment: candidate 2's UCB (1.02) shows high upside potential while LCB (0.68) shows acceptable downside, parameter quality supports optimistic outlook, (4) Budget efficiency analysis: With {remaining_budget} evaluations remaining (ample budget), can afford to resolve high-uncertainty, high-potential candidate. If budget were low (<10 remaining), would choose candidate 0 (narrow confidence interval, reliable). Rejected candidate 4 despite high confidence bounds [0.78, 0.98] because parameter analysis suggests disconnect between observed performance and parameter quality. Candidate 2's combination of wide confidence interval (high information value) and excellent parameters (high expected performance) makes it optimal choice.",
  "selected_index": 2
}}'''

        prompt_messages = [
            {
                "role": "system",
                "content": f"""
{self.domain_context}

## Core Capabilities
1. **Pattern Learning**: Identify which parameter characteristics correlate with high/low performance
2. **Score Prediction**: Predict scores for candidates based on their parameters
3. **Statistical Analysis**: Account for noise and confidence intervals in existing data
4. **Extrapolation**: Predict scores for new candidates by comparing their parameters to evaluated ones

## Input Data Analysis
You receive candidates with:
- **Parameters**: Configuration settings and instructions for each candidate
- **Statistics**: Candidates may have varying amounts of evaluation data (eval_count, mean_score, UCB/LCB bounds)
- **Data Variance**: Some candidates have extensive evaluation history, others have limited or no evaluation data

## Learning Objectives
**Learn parameter-performance patterns**:
- Which parameter characteristics correlate with higher performance?
- What content patterns, style, and structure work best?
- How do different parameter approaches affect outcomes?

**Apply patterns to predict scores**:
- Use learned patterns to predict performance for all candidates
- Consider both parameter quality and existing evaluation data
- Make predictions based on parameter-performance correlations

## Required Analysis Process

### Step 1: Parameter-Performance Pattern Learning
Analyze all candidates to identify performance patterns:
- **High-performing patterns**: What parameter characteristics correlate with better performance?
- **Low-performing patterns**: What characteristics correlate with weaker performance?
- **Content analysis**: Specific words, phrases, structures that correlate with scores
- **Length patterns**: How parameter length and detail level affect performance
- **Style patterns**: Detailed vs concise, formal vs conversational, structured vs flexible

### Step 2: Score Prediction for All Candidates
For each candidate, predict performance based on:
- **Parameter analysis**: Evaluate parameter quality using learned patterns
- **Existing data integration**: Incorporate available evaluation statistics when present
- **Pattern matching**: Compare parameters to successful patterns identified
- **Confidence assessment**: How confident are you in each prediction based on pattern strength?
- **Reasoning**: Explain your prediction based on parameter-performance correlations

### Step 3: Candidate Selection
Choose the candidate most likely to have the highest true performance:
- **Predicted performance**: Which candidate has the highest predicted score based on patterns?
- **Prediction confidence**: How reliable is your prediction based on pattern matching?
- **Information value**: Which candidate would provide most learning value for pattern validation?

## Prediction Strategy
When predicting scores for all candidates:
1. **Identify parameter patterns**: What patterns do you see across all candidates?
2. **Correlate with performance**: How do parameter characteristics relate to observed performance?
3. **Apply patterns consistently**: Use learned patterns to predict scores for all candidates
4. **Weight evidence appropriately**: Balance parameter analysis with existing evaluation data
5. **Justify predictions**: Explain reasoning based on parameter-performance patterns

{'## Output Requirements (With Score Estimation)' if self.enable_estimate_scores else '## Output Requirements (Selection Only)'}
Return ONLY a JSON object with these fields:
- "buffer_analysis": string analyzing parameter-performance patterns across all candidates. Focus on identifying what parameter characteristics correlate with performance, considering both parameter quality and available evaluation data.
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
Analyze parameter-performance patterns across all candidates and select the most promising candidate for evaluation. Use pattern-based predictions to estimate scores for all candidates, focusing on parameter characteristics that correlate with performance.

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
    def predict_scores(self, buffer, verbose: bool = False, temperature: float = 0.0):
        """
        Predict scores for all candidates in the buffer.
        This function is almost the same as llm_regressor, but it will not return the selected candidate. 
        
        Args:
            buffer: List of candidate entries with parameters and statistics
            verbose: Whether to print verbose output and debugging information
            temperature: Temperature parameter for LLM sampling (0.0 = deterministic, higher = more random)
            
        Returns:
            np.array: Vector of predicted scores, defaults to mean scores if LLM fails
        """
        
        # Prepare serializable candidate summaries with parameters
        serializable_candidate_summaries = []
        self.update_buffer_scores()
        for idx, cand_entry in enumerate(buffer):
            summary = {
                "index": idx,
                "parameters": {k.py_name: v for k,v in cand_entry['params'].items()},
                "eval_count": cand_entry['eval_count'],
                "mean_score": cand_entry['mean_score'],
                "ucb_score": cand_entry.get('ucb_score', None),
                "lcb_score": cand_entry.get('lcb_score', None)
            }
            serializable_candidate_summaries.append(summary)
        candidate_summaries_json = json.dumps(serializable_candidate_summaries, indent=2)
        
        example_param_schema_json = json.dumps({p.py_name: copy.deepcopy(p.data) for p in self.agent.parameters()}, indent=2)

        # Create the score prediction prompt (always with score estimation enabled)
        example_format = '''{{
  "buffer_analysis": "Parameter-Performance Learning: Analyzed 5 candidates with varying evaluation data. Pattern Analysis: Candidates with detailed parameter content (>500 chars) show higher performance trends. Parameter descriptions with specific examples correlate with better outcomes (+0.12 average boost). Comprehensive structure emphasis adds performance value (+0.08). Statistical Patterns: Higher eval_count candidates show more reliable score patterns, but parameter quality remains primary predictor across all candidates.",
  "score_estimates": {{
    "0": {{"reasoning": "Parameter analysis: comprehensive parameter content (650 chars), detailed descriptions, strong structure. Current data shows mean_score=0.75 with eval_count=25. Parameter quality strongly matches high-performance patterns. Predicted score reflects excellent parameter characteristics with confidence from existing data.", "predicted_score": 0.74}},
    "1": {{"reasoning": "Parameter analysis: moderate parameter content (400 chars), basic descriptions, minimal detail. Current data shows mean_score=0.65 with eval_count=30. Parameter patterns suggest mid-range performance, consistent with observed data. Missing key high-performance characteristics.", "predicted_score": 0.68}},
    "2": {{"reasoning": "Parameter analysis: excellent parameter focus (700+ chars), comprehensive examples, strong structure. Current data shows mean_score=0.82 with eval_count=8. Parameter quality strongly indicates high performance potential. Prediction based on strong parameter-performance correlation patterns.", "predicted_score": 0.79}},
    "3": {{"reasoning": "Parameter analysis: good parameter content (580 chars), decent examples, some structure. Limited evaluation data (eval_count=0) but parameter patterns similar to high-performing candidates. Predicted score based on parameter similarity analysis and learned performance patterns.", "predicted_score": 0.71}},
    "4": {{"reasoning": "Parameter analysis: basic parameter content (350 chars), minimal examples, limited structure. No evaluation data yet (eval_count=0) but parameter patterns match lower-performing profiles. Predicted score reflects weaker parameter characteristics based on learned patterns.", "predicted_score": 0.63}}
  }}
}}'''

        prompt_messages = [
            {
                "role": "system",
                "content": f"""
{self.domain_context}

## Core Capabilities
1. **Pattern Learning**: Identify which parameter characteristics correlate with high/low performance
2. **Score Prediction**: Predict scores for candidates based on their parameters
3. **Statistical Analysis**: Account for noise and confidence intervals in existing data
4. **Extrapolation**: Predict scores for new candidates by comparing their parameters to evaluated ones

## Input Data Analysis
You receive candidates with:
- **Parameters**: Configuration settings and instructions for each candidate
- **Statistics**: Candidates may have varying amounts of evaluation data (eval_count, mean_score, UCB/LCB bounds)
- **Data Variance**: Some candidates have extensive evaluation history, others have limited or no evaluation data

## Learning Objectives
**Learn parameter-performance patterns**:
- Which parameter characteristics correlate with higher performance?
- What content patterns, style, and structure work best?
- How do different parameter approaches affect outcomes?

**Apply patterns to predict scores**:
- Use learned patterns to predict performance for all candidates
- Consider both parameter quality and existing evaluation data
- Make predictions based on parameter-performance correlations

## Required Analysis Process

### Step 1: Parameter-Performance Pattern Learning
Analyze all candidates to identify performance patterns:
- **High-performing patterns**: What parameter characteristics correlate with better performance?
- **Low-performing patterns**: What characteristics correlate with weaker performance?
- **Content analysis**: Specific words, phrases, structures that correlate with scores
- **Length patterns**: How parameter length and detail level affect performance
- **Style patterns**: Detailed vs concise, formal vs conversational, structured vs flexible

### Step 2: Score Prediction for All Candidates
For each candidate, predict performance based on:
- **Parameter analysis**: Evaluate parameter quality using learned patterns
- **Existing data integration**: Incorporate available evaluation statistics when present
- **Pattern matching**: Compare parameters to successful patterns identified
- **Confidence assessment**: How confident are you in each prediction based on pattern strength?
- **Reasoning**: Explain your prediction based on parameter-performance correlations

## Prediction Strategy
When predicting scores for all candidates:
1. **Identify parameter patterns**: What patterns do you see across all candidates?
2. **Correlate with performance**: How do parameter characteristics relate to observed performance?
3. **Apply patterns consistently**: Use learned patterns to predict scores for all candidates
4. **Weight evidence appropriately**: Balance parameter analysis with existing evaluation data
5. **Justify predictions**: Explain reasoning based on parameter-performance patterns

## Output Requirements
Return ONLY a JSON object with these fields:
- "buffer_analysis": string analyzing parameter-performance patterns across all candidates. Focus on identifying what parameter characteristics correlate with performance, considering both parameter quality and available evaluation data.
- "score_estimates": object mapping candidate indices to predicted scores with detailed reasoning

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
Analyze parameter-performance patterns across all candidates and predict scores for each candidate. Use pattern-based predictions to estimate scores for all candidates, focusing on parameter characteristics that correlate with performance.

Return ONLY the JSON object with your analysis and score predictions.
""",
            },
        ]
        
        response_format = {"type": "json_object"}
        
        # Single LLM call with internal backoff handled by helper
        def llm_call():
            return self.llm(prompt_messages, response_format=response_format, temperature=temperature)
            
        llm_response = retry_with_exponential_backoff(
            llm_call,
            max_retries=10,
            base_delay=1.0,
            operation_name="LLM score prediction"
        )
        
        # Default fallback: return mean scores from buffer statistics
        default_scores = np.array([c.get('mean_score', 0.0) for c in buffer])
        
        if llm_response is None:
            if verbose:
                print_color("LLM score prediction call failed after retries. Returning mean scores.", "yellow")
            return default_scores

        llm_response_str = getattr(getattr(llm_response, 'choices', [{}])[0], 'message', None)
        llm_response_str = getattr(llm_response_str, 'content', None)
        if not llm_response_str:
            if verbose:
                print_color("LLM returned empty response for score prediction.", "yellow")
            return default_scores

        cleaned_llm_response_str = llm_response_str.strip()
        
        if verbose:
            self.print_buffer_statistics()
            print_color(f"LLM Score Prediction (temperature={temperature}): {cleaned_llm_response_str}", "cyan")
            
        try:
            llm_output = json.loads(cleaned_llm_response_str)
        except json.JSONDecodeError:
            if verbose:
                print_color("Failed to parse LLM score prediction JSON output.", "yellow")
            return default_scores

        if not isinstance(llm_output, dict):
            return default_scores

        # Extract score estimates
        score_estimates = llm_output.get("score_estimates", {})
        
        if verbose:
            buffer_analysis = llm_output.get("buffer_analysis", "No analysis provided")
            print_color(f"Buffer Analysis: {buffer_analysis}", "cyan")
            print_color(f"Score Estimates: {score_estimates}", "blue")

        # Convert score estimates to numpy array
        predicted_scores = []
        for idx in range(len(buffer)):
            candidate_key = str(idx)
            if candidate_key in score_estimates:
                try:
                    predicted_score = score_estimates[candidate_key].get("predicted_score", buffer[idx].get('mean_score', 0.0))
                    predicted_scores.append(float(predicted_score))
                except (ValueError, TypeError):
                    # Fallback to mean score if prediction is invalid
                    print_color(f"Invalid predicted score for candidate {idx}: {score_estimates[candidate_key]}", "yellow")
                    predicted_scores.append(buffer[idx].get('mean_score', 0.0))
            else:
                # Fallback to mean score if no prediction available
                print_color(f"No predicted score for candidate {idx}", "yellow")
                predicted_scores.append(buffer[idx].get('mean_score', 0.0))
        
        predicted_scores_array = np.array(predicted_scores)
        
        if verbose:
            print_color(f"Predicted scores: {predicted_scores_array}", "green")
            print_color(f"Mean scores (fallback): {default_scores}", "yellow")
            
        return predicted_scores_array

class LLMThompsonSampling(LLMRegressionModel):
    def __init__(self, agent, num_threads, logger, update_dicts, enable_estimate_scores=False, domain_context=None,enable_using_regressor=False, temperature=0.0, *args, **kwargs):
        super().__init__(agent, num_threads, logger, update_dicts, enable_estimate_scores, domain_context,enable_using_regressor, *args, **kwargs)
        self.temperature = temperature
    def llm_regressor(self, buffer, verbose: bool = False):
        """
        1. Given the history, compute the estimate on the candidates in the history.
        2. Choose the candidate with the highest score.
        """
        predicted_scores = self.predict_scores(buffer, verbose=True,temperature=self.temperature)
        print_color(f"Predicted scores with temperature {self.temperature}: {predicted_scores}", "cyan")
        
        # Find the index of the highest predicted score
        best_index = np.argmax(predicted_scores)
        selected_entry = buffer[best_index]

        # The following is for debugging. We want to show what the predicted scores are for different temperatures. Especially, we want to show the effect of temperature on the predicted scores of candidates without any score.
        # num_predict = 5
        # for _ in range(num_predict):
        #     predicted_scores = self.predict_scores(buffer, verbose=False,temperature=0)
        #     print_color(f"Predicted scores with temperature 0: {predicted_scores}", "cyan")
        # for _ in range(num_predict):
        #     predicted_scores = self.predict_scores(buffer, verbose=False,temperature=0.5)
        #     print_color(f"Predicted scores with temperature 0.5: {predicted_scores}", "cyan")
        # for _ in range(num_predict):
        #     predicted_scores = self.predict_scores(buffer, verbose=False,temperature=1)
        #     print_color(f"Predicted scores with temperature 1: {predicted_scores}", "cyan")
        # for _ in range(num_predict):
        #     predicted_scores = self.predict_scores(buffer, verbose=False,temperature=2)
        #     print_color(f"Predicted scores with temperature 2: {predicted_scores}", "cyan")
        return selected_entry
        
class LLMGenerator(LLMThompsonSampling):
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
                "parameters": {k.py_name: v for k,v in cand_entry['params'].items()},
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
{self.domain_context}

## Task
You are an expert agent optimizer generating high-performance configurations. Your goal is to create {num_to_generate} new candidates that will achieve higher task success rates than existing ones by addressing specific performance gaps.

## Success Criteria
Generate candidates that will improve performance by:
- **Higher success rates**: Better task completion and goal achievement
- **Error reduction**: Fewer failures and operational mistakes  
- **Improved reliability**: More consistent and predictable behavior
- **Enhanced effectiveness**: Better alignment with intended objectives

## Critical Analysis Framework
Before generating candidates, you MUST:

### Step 1: Failure Pattern Analysis
- **Identify specific failure modes**: What exactly causes current candidates to fail?
- **Quantify performance gaps**: Which candidates perform worst and why?
- **Analyze parameter weaknesses**: What specific parameter content leads to poor performance?

### Step 2: Success Pattern Extraction  
- **Identify what works**: What specific elements in higher-performing candidates drive success?
- **Extract transferable patterns**: Which successful approaches can be adapted/enhanced?
- **Understand performance drivers**: What parameter characteristics correlate with better scores?

### Step 3: Strategic Diversification
Each new candidate must target a DIFFERENT performance bottleneck:
- **Candidate 1**: Address the #1 failure pattern you identified
- **Candidate 2**: Enhance the #1 success pattern you found
- **Candidate 3**: Target a completely unexplored approach based on domain knowledge

## Mandatory Diversity Requirements
**CRITICAL - EACH CANDIDATE MUST BE FUNDAMENTALLY DIFFERENT**:
- **Different core strategies**: Tool usage philosophy, communication style, error handling approach
- **Different parameter structures**: Vary length, detail level, organization, and emphasis
- **Different performance targets**: Some optimize for accuracy, others for efficiency, others for robustness
- **Different risk profiles**: Conservative vs aggressive, detailed vs streamlined, comprehensive vs focused

**VALIDATION CHECK**: If any two candidates could be described with similar adjectives, they are TOO SIMILAR.

## Output Requirements
Return ONLY a JSON object with these fields:
- "buffer_analysis": string with your Step 1-2 analysis (failure patterns, success patterns, performance gaps)
- "generated_candidates": array of {num_to_generate} objects, each with:
  - "reasoning": string explaining the SPECIFIC performance problem this candidate solves and WHY it will outperform existing ones
  - "diversity_focus": string describing the UNIQUE strategy/approach that makes this candidate different from all others
  - "parameters": object with parameter values (matching the schema exactly)

## NON-NEGOTIABLE CONSTRAINTS
**CRITICAL - VIOLATIONS WILL CAUSE REJECTION**:

### Parameter Structure Requirements
- **Exact schema match**: Use ONLY the parameter keys provided in the schema
- **Correct data types**: All parameter values must be strings (even if they contain structured content)
- **Complete coverage**: Include ALL required parameters, no omissions allowed
- **Consistent formatting**: Follow the same naming and structure conventions as existing candidates

### Content Quality Requirements  
- **Concrete specificity**: Avoid vague phrases like "better handling" or "improved approach"
- **Actionable instructions**: Parameter content must provide clear, executable guidance
- **Practical relevance**: All content must be directly applicable to the optimization domain
- **Length appropriateness**: Match the expected parameter length patterns from existing candidates

### Diversity Enforcement
- **Unique approaches**: Each candidate must solve a DIFFERENT core problem
- **Distinct strategies**: No two candidates should have similar methodologies
- **Varied structures**: Significantly different parameter organization and emphasis

## Example Output Format
{{
  "buffer_analysis": "FAILURE ANALYSIS: Candidate 0 (score 0.45) fails due to vague parameter descriptions lacking concrete examples - causes 40% implementation errors. Candidate 1 (score 0.52) has verbose but unstructured content - leads to execution confusion. SUCCESS ANALYSIS: Candidate 2 (score 0.78) succeeds with structured examples and clear validation steps - drives 25% better performance. PERFORMANCE GAPS: No candidates address edge case handling (major failure mode), none optimize for multi-step processes, missing systematic error recovery patterns.",
  "generated_candidates": [
    {{
      "reasoning": "TARGETS FAILURE MODE: Implementation errors (40% of failures). Current candidates provide vague descriptions. This candidate provides concrete examples and validation steps, directly addressing the #1 cause of failures. Expected improvement: 30-40% reduction in errors based on structured guidance approach.",
      "diversity_focus": "Precision specialist: Exhaustive examples with validation, completely different from existing vague descriptions",
      "parameters": {{
        "param1": "Each operation includes 3 concrete usage examples with exact formats. Always validate inputs before execution. Include step-by-step verification procedures...",
        "param2": "CRITICAL: Before any operation, verify all required inputs are present and correctly formatted. If validation fails, provide specific guidance with concrete examples..."
      }}
    }},
    {{
      "reasoning": "ENHANCES SUCCESS PATTERN: Builds on candidate 2's structured approach but optimizes for multi-step processes. Current candidates handle single operations well but fail in complex sequences. This candidate provides systematic orchestration with checkpoint validation, targeting 20% of remaining failures.",
      "diversity_focus": "Process orchestration expert: Multi-step optimization with checkpoints, unique systematic approach",
      "parameters": {{
        "param1": "For complex processes, break into phases with validation checkpoints. Phase 1: Input gathering and validation. Phase 2: Execution with confirmation. Phase 3: Result verification and output...",
        "param2": "PROCESS PROTOCOL: At each step, confirm previous step completion before proceeding. If any step fails, provide specific recovery options rather than generic error messages..."
      }}
    }},
    {{
      "reasoning": "ADDRESSES UNEXPLORED AREA: Edge case and exception handling (15% of failures). No current candidates handle boundary conditions effectively. This candidate specializes in robust exception handling and edge case management, targeting a completely different failure category.",
      "diversity_focus": "Robustness guardian: Edge case handling with exception management, novel defensive approach",
      "parameters": {{
        "param1": "Before any operation, check for boundary conditions and edge cases. Validate input ranges, handle null/empty values, and provide graceful degradation for unexpected scenarios...",
        "param2": "ROBUSTNESS FIRST: When edge cases occur, provide clear explanations and alternative approaches. Always validate assumptions and handle exceptions gracefully with informative feedback..."
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
Follow the Critical Analysis Framework:
1. **Analyze failures**: Identify specific failure modes and performance gaps in current candidates
2. **Extract successes**: Find what works and can be enhanced
3. **Generate strategically**: Create {num_to_generate} candidates that each target DIFFERENT performance bottlenecks

**MANDATORY**: Each candidate must solve a fundamentally different problem. No similar approaches allowed.

Return ONLY the JSON object following the exact format shown in the example.
""",
            },
        ]

        response_format = {"type": "json_object"}

        # LLM call with retry logic
        def llm_call():
            return self.llm(prompt_messages, response_format=response_format)
        if verbose:
            print("candidats_in_prompt_messages: ", candidate_summaries_json)
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
                # Validate parameter structure consistency
                list_param_data = None
                # Find any list parameter in parameters_raw
                for key, value in parameters_raw.items():
                    if isinstance(value, list):
                        list_param_data = value
                        break
                
                if self.num_tools is None:
                    params_dict = buffer[0]['params']
                    for param_node, param_value in params_dict.items():
                        if isinstance(param_value, list):
                            self.num_tools = len(param_value)
                            break
                    
                # check whether list parameter contains the same number of items as the first candidate
                if list_param_data:
                    if len(list_param_data) != self.num_tools:
                        if verbose:
                            print_color(f"Skipping candidate {i+1}: list parameter contains {len(list_param_data)} items, expected {self.num_tools}", "yellow")
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
        self.buffer = self.llm_generator(buffer, verbose = True, num_to_generate=3)

        # Use the LLM regression model to select the best candidate from the buffer
        return self.llm_regressor(self.buffer, verbose)

class LLMRegressThenGenerate_onecall(LLMRegressionModel):
    """Do the same thing as LLMGenerator, but in one call to the LLM.
    LLMGenerator will call the LLM twice, once to generate the candidates using llm_generator, and once to select the best candidate using llm_regressor.
    LLMRegressThenGenerate_onecall will call the LLM once.
    """
    def __init__(self, agent, num_threads, logger, update_dicts, enable_estimate_scores=False, domain_context=None, *args, **kwargs):
        super().__init__(agent, num_threads, logger, update_dicts, enable_estimate_scores, domain_context, *args, **kwargs)
        # Could add parameters here.

    def llm_generate_candidate(self, buffer, verbose: bool = False):
        """Combined generation and regression in one LLM call.
        
        1. Generate new candidates based on current buffer
        2. Predict scores for all candidates (existing + new)
        3. Select best candidate for evaluation
        4. If new candidate selected, add to buffer
        """
        # Calculate budget information
        total_budget = self.num_epochs * self.horizon
        used_budget = self.selection_count
        remaining_budget = total_budget - used_budget
        
        # Filter buffer to only include candidates with valid scores for analysis
        buffer_with_scores = [c for c in buffer if c['eval_count'] > 0]
        
        # Prepare serializable candidate summaries for existing candidates
        serializable_candidate_summaries = []
        self.update_buffer_scores()
        for idx, cand_entry in enumerate(buffer):
            summary = {
                "index": idx,
                "parameters": {k.py_name: v for k,v in cand_entry['params'].items()},
                "eval_count": cand_entry['eval_count'],
                "mean_score": cand_entry['mean_score'],
                "ucb_score": cand_entry.get('ucb_score', None),
                "lcb_score": cand_entry.get('lcb_score', None)
            }
            serializable_candidate_summaries.append(summary)
        candidate_summaries_json = json.dumps(serializable_candidate_summaries, indent=2)
        
        example_param_schema_json = json.dumps({p.py_name: copy.deepcopy(p.data) for p in self.agent.parameters()}, indent=2)

        # Create the combined prompt
        prompt_messages = [
            {
                "role": "system",
                "content": f"""
{self.domain_context}

## Combined Task: Generation + Regression + Selection

You will perform three tasks in sequence:

### Task 1: Generate New Candidates
Based on current buffer analysis, generate 3 diverse new candidates that could outperform existing ones.

#### Critical Analysis Framework
Before generating candidates, you MUST:

**Step 1: Failure Pattern Analysis**
- Identify specific failure modes in current candidates
- Quantify performance gaps and analyze parameter weaknesses
- Understand what causes poor performance

**Step 2: Success Pattern Extraction**  
- Identify what works in higher-performing candidates
- Extract transferable patterns and understand performance drivers
- Find successful approaches that can be enhanced

**Step 3: Strategic Diversification**
Each new candidate must target a DIFFERENT performance bottleneck:
- Candidate 1: Address the #1 failure pattern identified
- Candidate 2: Enhance the #1 success pattern found
- Candidate 3: Target a completely unexplored approach

#### Mandatory Diversity Requirements
**CRITICAL - EACH CANDIDATE MUST BE FUNDAMENTALLY DIFFERENT**:
- Different core strategies and parameter structures
- Different performance targets (accuracy vs efficiency vs robustness)
- Different risk profiles (conservative vs aggressive, detailed vs streamlined)

**VALIDATION CHECK**: If any two candidates could be described with similar adjectives, they are TOO SIMILAR.

### Task 2: Score Prediction
Using parameter-performance patterns, predict scores for ALL candidates (existing + newly generated):

#### Pattern Learning Process
- Analyze parameter characteristics that correlate with performance
- Learn from existing evaluation data and parameter quality
- Apply patterns consistently to predict scores for all candidates

#### Prediction Strategy
- Use learned patterns to predict performance for all candidates
- Consider both parameter quality and existing evaluation data
- Make predictions based on parameter-performance correlations

### Task 3: Candidate Selection
Choose the candidate most likely to have the highest true performance:
- Select based on predicted performance ranking
- Consider prediction confidence and information value
- Choose the candidate that maximizes expected performance

## Budget Information
- Remaining budget: {remaining_budget} evaluations
- Selection number: {self.selection_count}/{total_budget}

## Output Requirements
Return ONLY a JSON object with these fields:
- "buffer_analysis": string with failure patterns, success patterns, and performance gaps analysis
- "generated_candidates": array of 3 objects, each with:
  - "reasoning": string explaining the SPECIFIC performance problem this candidate solves
  - "diversity_focus": string describing the UNIQUE strategy that makes this candidate different
  - "parameters": object with parameter values (matching schema exactly)
- "score_estimates": object mapping ALL candidate indices (existing + new) to predicted scores with reasoning
- "selection_reasoning": string explaining your candidate choice based on predicted performance
- "selected_candidate": object with:
  - "is_new": boolean (true if selecting a newly generated candidate, false if existing)
  - "index": integer (if is_new=false, index in existing buffer; if is_new=true, index in generated_candidates 0-2)

## NON-NEGOTIABLE CONSTRAINTS
**CRITICAL - VIOLATIONS WILL CAUSE REJECTION**:

### Parameter Structure Requirements
- Use ONLY the parameter keys provided in the schema
- All parameter values must be strings
- Include ALL required parameters, no omissions allowed
- Follow consistent naming and structure conventions

### Content Quality Requirements  
- Concrete specificity: Avoid vague phrases
- Actionable instructions: Provide clear, executable guidance
- Practical relevance: Content must be applicable to the optimization domain
- Length appropriateness: Match expected parameter length patterns

### Diversity Enforcement
- Each candidate must solve a DIFFERENT core problem
- No two candidates should have similar methodologies
- Significantly different parameter organization and emphasis

## Example Output Format
{{
  "buffer_analysis": "FAILURE ANALYSIS: Candidate 0 (score 0.45) fails due to vague parameter descriptions - causes 40% implementation errors. SUCCESS ANALYSIS: Candidate 2 (score 0.78) succeeds with structured examples. PERFORMANCE GAPS: No candidates address edge cases, missing systematic error recovery.",
  "generated_candidates": [
    {{
      "reasoning": "TARGETS FAILURE MODE: Implementation errors (40% of failures). This candidate provides concrete examples and validation steps.",
      "diversity_focus": "Precision specialist: Exhaustive examples with validation",
      "parameters": {{
        "param1": "Each operation includes concrete examples with exact formats...",
        "param2": "CRITICAL: Verify all inputs before execution..."
      }}
    }},
    {{
      "reasoning": "ENHANCES SUCCESS PATTERN: Builds on structured approach but optimizes for multi-step processes.",
      "diversity_focus": "Process orchestration expert: Multi-step optimization with checkpoints",
      "parameters": {{
        "param1": "For complex processes, break into phases with validation...",
        "param2": "PROCESS PROTOCOL: Confirm each step completion..."
      }}
    }},
    {{
      "reasoning": "ADDRESSES UNEXPLORED AREA: Edge case handling (15% of failures).",
      "diversity_focus": "Robustness guardian: Edge case handling with exception management",
      "parameters": {{
        "param1": "Check boundary conditions and edge cases before operations...",
        "param2": "ROBUSTNESS FIRST: Handle exceptions gracefully..."
      }}
    }}
  ],
  "score_estimates": {{
    "0": {{"reasoning": "Parameter analysis shows vague descriptions. Current score 0.45 matches pattern of poor parameter quality.", "predicted_score": 0.47}},
    "1": {{"reasoning": "Structured parameters with good examples. Score 0.78 reflects excellent parameter-performance correlation.", "predicted_score": 0.76}},
    "new_0": {{"reasoning": "Precision-focused approach with concrete examples should address main failure mode. Predicted high performance.", "predicted_score": 0.82}},
    "new_1": {{"reasoning": "Process optimization builds on successful patterns. Expected strong performance.", "predicted_score": 0.79}},
    "new_2": {{"reasoning": "Edge case handling addresses unexplored area. Moderate improvement expected.", "predicted_score": 0.73}}
  }},
  "selection_reasoning": "Predicted ranking: new_0 (0.82) > new_1 (0.79) > existing_1 (0.76) > new_2 (0.73) > existing_0 (0.47). Selecting new_0 because it has highest predicted score and directly addresses the main failure mode.",
  "selected_candidate": {{
    "is_new": true,
    "index": 0
  }}
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
Perform the combined generation + regression + selection process:
1. Analyze current buffer and generate 3 diverse new candidates
2. Predict scores for all candidates (existing + new) using parameter-performance patterns
3. Select the best candidate for evaluation

Return ONLY the JSON object following the exact format shown in the example.
""",
            },
        ]
        
        response_format = {"type": "json_object"}
        
        # Single LLM call
        def llm_call():
            return self.llm(prompt_messages, response_format=response_format)
            
        if verbose:
            print("Combined generation+regression candidates in prompt: ", candidate_summaries_json)
            
        llm_response = retry_with_exponential_backoff(
            llm_call,
            max_retries=10,
            base_delay=1.0,
            operation_name="LLM combined generation+regression"
        )
        
        # Default fallback
        default_entry = max(buffer_with_scores, key=lambda c: c['mean_score'])
        
        if llm_response is None:
            if verbose:
                print_color("LLM combined call failed after retries. Returning highest scoring candidate.", "yellow")
            return default_entry

        llm_response_str = getattr(getattr(llm_response, 'choices', [{}])[0], 'message', None)
        llm_response_str = getattr(llm_response_str, 'content', None)
        if not llm_response_str:
            if verbose:
                print_color("LLM returned empty response.", "yellow")
            return default_entry

        cleaned_llm_response_str = llm_response_str.strip()
        
        if verbose:
            print_color(f"LLM Combined response: {cleaned_llm_response_str}", "cyan")
            
        try:
            llm_output = json.loads(cleaned_llm_response_str)
        except json.JSONDecodeError:
            if verbose:
                print_color("Failed to parse LLM combined JSON output.", "yellow")
            return default_entry

        if not isinstance(llm_output, dict):
            return default_entry

        # Extract components
        generated_candidates = llm_output.get("generated_candidates", [])
        selected_candidate_info = llm_output.get("selected_candidate", {})
        
        if verbose:
            buffer_analysis = llm_output.get("buffer_analysis", "No analysis provided")
            selection_reasoning = llm_output.get("selection_reasoning", "No reasoning provided")
            print_color(f"Buffer Analysis: {buffer_analysis}", "cyan")
            print_color(f"Selection Reasoning: {selection_reasoning}", "cyan")
            print_color(f"Generated {len(generated_candidates)} new candidates", "green")

        # Process selection
        is_new = selected_candidate_info.get("is_new", False)
        selected_index = selected_candidate_info.get("index", 0)
        
        if is_new and 0 <= selected_index < len(generated_candidates):
            # Selected a new candidate - need to create buffer entry
            try:
                selected_generated = generated_candidates[selected_index]
                parameters_raw = selected_generated.get("parameters", {})
                
                # Validate parameter structure for lists (same as in llm_generator)
                list_param_data = None
                for key, value in parameters_raw.items():
                    if isinstance(value, list):
                        list_param_data = value
                        break
                
                if self.num_tools is None:
                    params_dict = buffer[0]['params']
                    for param_node, param_value in params_dict.items():
                        if isinstance(param_value, list):
                            self.num_tools = len(param_value)
                            break
                    
                if list_param_data and len(list_param_data) != self.num_tools:
                    if verbose:
                        print_color(f"Selected new candidate has invalid list parameter length {len(list_param_data)}, expected {self.num_tools}. Falling back to best existing.", "yellow")
                    return default_entry
                
                # Convert parameters to correct format
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
                
                # Add to buffer
                buffer.append(new_candidate_entry)
                
                if verbose:
                    reasoning = selected_generated.get("reasoning", "No reasoning provided")
                    diversity_focus = selected_generated.get("diversity_focus", "No diversity focus provided")
                    print_color(f"Selected NEW candidate {selected_index}:", "green")
                    print_color(f"  Reasoning: {reasoning}", "green")
                    print_color(f"  Diversity Focus: {diversity_focus}", "green")
                
                return new_candidate_entry
                
            except Exception as e:
                if verbose:
                    print_color(f"Error creating new candidate: {e}. Falling back to best existing.", "yellow")
                return default_entry
                
        elif not is_new and 0 <= selected_index < len(buffer):
            # Selected existing candidate
            selected_entry = buffer[selected_index]
            if verbose:
                print_color(f"Selected EXISTING candidate {selected_index}", "green")
            
            return selected_entry
            
        else:
            # Invalid selection
            if verbose:
                print_color("Invalid candidate selection. Falling back to best existing.", "yellow")
            return default_entry
        
class LLMTS_onecall(LLMRegressionModel):
    """Do the same thing as LLMThompsonSampling, but in one call to the LLM.
    """
    def __init__(self, agent, num_threads, logger, update_dicts, enable_estimate_scores=False, domain_context=None,enable_using_regressor=False, temperature=0.0, *args, **kwargs):
        super().__init__(agent, num_threads, logger, update_dicts, enable_estimate_scores, domain_context,enable_using_regressor, *args, **kwargs)
        self.temperature = temperature
    
    
    def llm_generate_candidate(self, buffer, verbose: bool = False):
        """
        Combined generation and score prediction in one LLM call, then select highest scoring candidate.
        
        1. Generate new candidates based on current buffer
        2. Predict scores for all candidates (existing + new)
        3. Programmatically select candidate with highest predicted score
        4. If new candidate selected, add to buffer
        
        Output: selected candidate entry.
        """
        # Calculate budget information
        # total_budget = self.num_epochs * self.horizon
        # used_budget = self.selection_count
        # remaining_budget = total_budget - used_budget
        
        
        # Prepare serializable candidate summaries for existing candidates
        serializable_candidate_summaries = []
        self.update_buffer_scores()
        for idx, cand_entry in enumerate(buffer):
            summary = {
                "index": idx,
                "parameters": {k.py_name: v for k,v in cand_entry['params'].items()},
                "eval_count": cand_entry['eval_count'],
                "mean_score": cand_entry['mean_score'],
                "ucb_score": cand_entry.get('ucb_score', None),
                "lcb_score": cand_entry.get('lcb_score', None)
            }
            serializable_candidate_summaries.append(summary)
        candidate_summaries_json = json.dumps(serializable_candidate_summaries, indent=2)
        
        example_param_schema_json = json.dumps({p.py_name: copy.deepcopy(p.data) for p in self.agent.parameters()}, indent=2)

        # Create the combined prompt
        prompt_messages = [
            {
                "role": "system",
                "content": f"""
{self.domain_context}

## Combined Task: Generation + Score Prediction

You will perform two tasks in sequence:

### Task 1: Generate New Candidates
Based on current buffer analysis, generate 3 diverse new candidates that could outperform existing ones.

#### Critical Analysis Framework
Before generating candidates, you MUST:

**Step 1: Failure Pattern Analysis**
- Identify specific failure modes in current candidates
- Quantify performance gaps and analyze parameter weaknesses
- Understand what causes poor performance

**Step 2: Success Pattern Extraction**  
- Identify what works in higher-performing candidates
- Extract transferable patterns and understand performance drivers
- Find successful approaches that can be enhanced

**Step 3: Strategic Diversification**
Each new candidate must target a DIFFERENT performance bottleneck:
- Candidate 1: Address the #1 failure pattern identified
- Candidate 2: Enhance the #1 success pattern found
- Candidate 3: Target a completely unexplored approach

#### Mandatory Diversity Requirements
**CRITICAL - EACH CANDIDATE MUST BE FUNDAMENTALLY DIFFERENT**:
- Different core strategies and parameter structures
- Different performance targets (accuracy vs efficiency vs robustness)
- Different risk profiles (conservative vs aggressive, detailed vs streamlined)

**VALIDATION CHECK**: If any two candidates could be described with similar adjectives, they are TOO SIMILAR.

### Task 2: Score Prediction
Using parameter-performance patterns, predict scores for ALL candidates (existing + newly generated):

#### Pattern Learning Process
- Analyze parameter characteristics that correlate with performance
- Learn from existing evaluation data and parameter quality
- Apply patterns consistently to predict scores for all candidates

#### Prediction Strategy
- Use learned patterns to predict performance for all candidates
- Consider both parameter quality and existing evaluation data
- Make predictions based on parameter-performance correlations


## Output Requirements
Return ONLY a JSON object with these fields:
- "buffer_analysis": string with failure patterns, success patterns, and performance gaps analysis
- "generated_candidates": array of 3 objects, each with:
  - "reasoning": string explaining the SPECIFIC performance problem this candidate solves
  - "diversity_focus": string describing the UNIQUE strategy that makes this candidate different
  - "parameters": object with parameter values (matching schema exactly)
- "score_estimates": object mapping ALL candidate indices (existing + new) to predicted scores with reasoning

## NON-NEGOTIABLE CONSTRAINTS
**CRITICAL - VIOLATIONS WILL CAUSE REJECTION**:

### Parameter Structure Requirements
- Use ONLY the parameter keys provided in the schema
- All parameter values must be strings
- Include ALL required parameters, no omissions allowed
- Follow consistent naming and structure conventions

### Content Quality Requirements  
- Concrete specificity: Avoid vague phrases
- Actionable instructions: Provide clear, executable guidance
- Practical relevance: Content must be applicable to the optimization domain
- Length appropriateness: Match expected parameter length patterns

### Diversity Enforcement
- Each candidate must solve a DIFFERENT core problem
- No two candidates should have similar methodologies
- Significantly different parameter organization and emphasis

## Example Output Format
{{
  "buffer_analysis": "FAILURE ANALYSIS: Candidate 0 (score 0.45) fails due to vague parameter descriptions - causes 40% implementation errors. SUCCESS ANALYSIS: Candidate 2 (score 0.78) succeeds with structured examples. PERFORMANCE GAPS: No candidates address edge cases, missing systematic error recovery.",
  "generated_candidates": [
    {{
      "reasoning": "TARGETS FAILURE MODE: Implementation errors (40% of failures). This candidate provides concrete examples and validation steps.",
      "diversity_focus": "Precision specialist: Exhaustive examples with validation",
      "parameters": {{
        "param1": "Each operation includes concrete examples with exact formats...",
        "param2": "CRITICAL: Verify all inputs before execution..."
      }}
    }},
    {{
      "reasoning": "ENHANCES SUCCESS PATTERN: Builds on structured approach but optimizes for multi-step processes.",
      "diversity_focus": "Process orchestration expert: Multi-step optimization with checkpoints",
      "parameters": {{
        "param1": "For complex processes, break into phases with validation...",
        "param2": "PROCESS PROTOCOL: Confirm each step completion..."
      }}
    }},
    {{
      "reasoning": "ADDRESSES UNEXPLORED AREA: Edge case handling (15% of failures).",
      "diversity_focus": "Robustness guardian: Edge case handling with exception management",
      "parameters": {{
        "param1": "Check boundary conditions and edge cases before operations...",
        "param2": "ROBUSTNESS FIRST: Handle exceptions gracefully..."
      }}
    }}
  ],
  "score_estimates": {{
    "0": {{"reasoning": "Parameter analysis shows vague descriptions. Current score 0.45 matches pattern of poor parameter quality.", "predicted_score": 0.47}},
    "1": {{"reasoning": "Structured parameters with good examples. Score 0.78 reflects excellent parameter-performance correlation.", "predicted_score": 0.76}},
    "new_0": {{"reasoning": "Precision-focused approach with concrete examples should address main failure mode. Predicted high performance.", "predicted_score": 0.82}},
    "new_1": {{"reasoning": "Process optimization builds on successful patterns. Expected strong performance.", "predicted_score": 0.79}},
    "new_2": {{"reasoning": "Edge case handling addresses unexplored area. Moderate improvement expected.", "predicted_score": 0.73}}
  }}
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
Perform the combined generation + score prediction process:
1. Analyze current buffer and generate 3 diverse new candidates
2. Predict scores for all candidates (existing + new) using parameter-performance patterns

Return ONLY the JSON object following the exact format shown in the example.
""",
            },
        ]
        
        response_format = {"type": "json_object"}
        
        # Single LLM call
        def llm_call():
            return self.llm(prompt_messages, response_format=response_format, temperature=self.temperature)
            
        if verbose:
            print("LLMTS_onecall candidates in prompt: ", candidate_summaries_json)
            
        llm_response = retry_with_exponential_backoff(
            llm_call,
            max_retries=10,
            base_delay=1.0,
            operation_name="LLM Thompson Sampling one call"
        )
        
        # Default fallback
        default_entry = max(buffer, key=lambda c: c['mean_score'])
        
        if llm_response is None:
            if verbose:
                print_color("LLM Thompson Sampling call failed after retries. Returning highest scoring candidate.", "yellow")
            return default_entry

        llm_response_str = getattr(getattr(llm_response, 'choices', [{}])[0], 'message', None)
        llm_response_str = getattr(llm_response_str, 'content', None)
        if not llm_response_str:
            if verbose:
                print_color("LLM returned empty response.", "yellow")
            return default_entry

        cleaned_llm_response_str = llm_response_str.strip()
        
        if verbose:
            print_color(f"LLM Thompson Sampling response (temperature={self.temperature}): {cleaned_llm_response_str}", "cyan")
            
        try:
            llm_output = json.loads(cleaned_llm_response_str)
        except json.JSONDecodeError:
            if verbose:
                print_color("Failed to parse LLM Thompson Sampling JSON output.", "yellow")
            return default_entry

        if not isinstance(llm_output, dict):
            return default_entry

        # Extract components
        generated_candidates = llm_output.get("generated_candidates", [])
        score_estimates = llm_output.get("score_estimates", {})
        
        if verbose:
            buffer_analysis = llm_output.get("buffer_analysis", "No analysis provided")
            print_color(f"Buffer Analysis: {buffer_analysis}", "cyan")
            print_color(f"Generated {len(generated_candidates)} new candidates", "green")
            print_color(f"Score Estimates: {score_estimates}", "blue")

        # Collect all candidates with their predicted scores
        all_candidates = []
        
        # Add existing candidates with their predicted scores
        for idx, candidate_entry in enumerate(buffer):
            candidate_key = str(idx)
            predicted_score = score_estimates.get(candidate_key, {}).get("predicted_score", candidate_entry.get('mean_score', 0.0))
            try:
                predicted_score = float(predicted_score)
            except (ValueError, TypeError):
                predicted_score = candidate_entry.get('mean_score', 0.0)
            
            all_candidates.append({
                "type": "existing",
                "index": idx,
                "entry": candidate_entry,
                "predicted_score": predicted_score
            })
        
        # Process and add new candidates with their predicted scores
        valid_new_candidates = []
        for i, candidate_data in enumerate(generated_candidates):
            try:
                parameters_raw = candidate_data.get("parameters", {})
                reasoning = candidate_data.get("reasoning", "No reasoning provided")
                diversity_focus = candidate_data.get("diversity_focus", "No diversity focus provided")
                
                # Validate parameter structure consistency
                list_param_data = None
                for key, value in parameters_raw.items():
                    if isinstance(value, list):
                        list_param_data = value
                        break
                
                if self.num_tools is None:
                    params_dict = buffer[0]['params']
                    for param_node, param_value in params_dict.items():
                        if isinstance(param_value, list):
                            self.num_tools = len(param_value)
                            break
                    
                # Check whether list parameter contains the same number of items as expected
                if list_param_data:
                    if len(list_param_data) != self.num_tools:
                        if verbose:
                            print_color(f"Skipping new candidate {i}: list parameter contains {len(list_param_data)} items, expected {self.num_tools}", "yellow")
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
                
                # Get predicted score for this new candidate
                new_candidate_key = f"new_{i}"
                predicted_score = score_estimates.get(new_candidate_key, {}).get("predicted_score", 0.0)
                try:
                    predicted_score = float(predicted_score)
                except (ValueError, TypeError):
                    predicted_score = 0.0
                
                valid_new_candidates.append({
                    "type": "new",
                    "index": i,
                    "entry": new_candidate_entry,
                    "predicted_score": predicted_score,
                    "reasoning": reasoning,
                    "diversity_focus": diversity_focus
                })
                
                all_candidates.append({
                    "type": "new",
                    "index": i,
                    "entry": new_candidate_entry,
                    "predicted_score": predicted_score,
                    "reasoning": reasoning,
                    "diversity_focus": diversity_focus
                })

                if verbose:
                    print_color(f"Valid new candidate {i}:", "blue")
                    print_color(f"  Reasoning: {reasoning}", "blue")
                    print_color(f"  Diversity Focus: {diversity_focus}", "blue")
                    print_color(f"  Predicted Score: {predicted_score}", "blue")

            except Exception as e:
                if verbose:
                    print_color(f"Error processing new candidate {i}: {e}", "yellow")
                continue

        # Find the candidate with the highest predicted score
        if not all_candidates:
            if verbose:
                print_color("No valid candidates found. Returning default.", "yellow")
            return default_entry
        
        best_candidate = max(all_candidates, key=lambda x: x['predicted_score'])
        
        predicted_scores = [float(c['predicted_score']) for c in all_candidates]
        buffer_scores = [float(c['mean_score']) if c['mean_score'] is not None else 0.0 for c in buffer]
        if verbose:
            print_color(f"Buffer scores: {buffer_scores}", "cyan")
            print_color(f"Predicted scores with temperature {self.temperature}: {predicted_scores}", "cyan")
            print_color(f"Best candidate type: {best_candidate['type']}, predicted score: {best_candidate['predicted_score']}", "green")
        
        # If the best candidate is new, add it to the buffer
        if best_candidate['type'] == 'new':
            buffer.append(best_candidate['entry'])
            if verbose:
                print_color(f"Selected NEW candidate {best_candidate['index']} with predicted score {best_candidate['predicted_score']}", "green")
                print_color(f"  Reasoning: {best_candidate.get('reasoning', 'N/A')}", "green")
                print_color(f"  Diversity Focus: {best_candidate.get('diversity_focus', 'N/A')}", "green")
        else:
            if verbose:
                print_color(f"Selected EXISTING candidate {best_candidate['index']} with predicted score {best_candidate['predicted_score']}", "green")
        
        return best_candidate['entry']

class LLMSimpleGenerator(LLMRegressionModel):
    """Simple generator that only calls the LLM once to generate candidates.
    """
    def __init__(self, agent, num_threads, logger, update_dicts, enable_estimate_scores=False, domain_context=None,enable_using_regressor=False, *args, **kwargs):
        super().__init__(agent, num_threads, logger, update_dicts, enable_estimate_scores, domain_context, *args, **kwargs)
        # Could add parameters here.
        # If enable_using_regressor is True, the LLM will use the regressor predict scores before making the final selection.
        self.enable_using_regressor = enable_using_regressor

    def llm_generate_candidate(self, buffer, verbose: bool = False):
        """Skip the step of generating new candidates explicitly. Given the history, compute the estimate on the history. At the end, let the LLM select a candidate to evaluate, which might not be within the current buffer.
         To be specific, the LLM reasoning process should be:
         1. Given the history, if enable_using_regressor is True, compute the estimate on the candidates in the history.
         2. Ask the LLM to select a candidate to evaluate, which might not be within the current buffer.
         3. If LLM selects a new candidate, create a new buffer entry for the new candidate.
         4. Return the selected candidate entry.
         """
        # Calculate budget information
        total_budget = self.num_epochs * self.horizon
        used_budget = self.selection_count
        remaining_budget = total_budget - used_budget
        
        # Filter buffer to only include candidates with valid scores for analysis
        buffer_with_scores = [c for c in buffer if c['eval_count'] > 0]
        
        # Prepare serializable candidate summaries for existing candidates
        serializable_candidate_summaries = []
        self.update_buffer_scores()
        for idx, cand_entry in enumerate(buffer):
            summary = {
                "index": idx,
                "parameters": {k.py_name: v for k,v in cand_entry['params'].items()},
                "eval_count": cand_entry['eval_count'],
                "mean_score": cand_entry['mean_score'],
                "ucb_score": cand_entry.get('ucb_score', None),
                "lcb_score": cand_entry.get('lcb_score', None)
            }
            serializable_candidate_summaries.append(summary)
        candidate_summaries_json = json.dumps(serializable_candidate_summaries, indent=2)
        
        example_param_schema_json = json.dumps({p.py_name: copy.deepcopy(p.data) for p in self.agent.parameters()}, indent=2)

        # Create the prompt for regression + optional generation
        prompt_messages = [
            {
                "role": "system",
                "content": f"""
{self.domain_context}

## Task: Regression Analysis + Candidate Selection

You will perform two main tasks:

### Task 1: Parameter-Performance Analysis
Analyze the existing candidates to understand performance patterns:

#### Pattern Learning Process
- **Identify parameter-performance correlations**: What parameter characteristics lead to higher/lower performance?
- **Learn from evaluation data**: Use existing scores and evaluation counts to understand reliability
- **Extract success patterns**: What makes high-performing candidates successful?
- **Identify failure patterns**: What causes poor performance in low-scoring candidates?
- **Understand performance drivers**: Which specific parameter elements correlate with better outcomes?

{'#### Score Prediction' if self.enable_using_regressor else ''}
{'For each existing candidate, predict their true performance based on:' if self.enable_using_regressor else ''}
{'- Parameter quality analysis using learned patterns' if self.enable_using_regressor else ''}
{'- Integration of existing evaluation data when available' if self.enable_using_regressor else ''}
{'- Pattern matching against successful configurations' if self.enable_using_regressor else ''}
{'- Confidence assessment based on parameter-performance correlations' if self.enable_using_regressor else ''}

### Task 2: Candidate Selection
Based on your analysis, choose the best candidate to evaluate next:

#### Selection Options
You have two choices:
1. **Select an existing candidate**: Choose from the current buffer based on predicted performance
2. **Propose a new candidate**: If existing candidates are insufficient, create a new one that addresses identified gaps

#### Decision Criteria
- **Predicted performance**: Which option is most likely to achieve the highest score?
- **Information value**: Which choice provides the most learning value?
- **Gap analysis**: Do existing candidates adequately explore the parameter space, or is a new approach needed?

## Budget Information
- Remaining budget: {remaining_budget} evaluations
- Selection number: {self.selection_count}/{total_budget}

## Output Requirements
Return ONLY a JSON object with these fields:
- "buffer_analysis": string analyzing parameter-performance patterns and candidate reliability
{'- "score_estimates": object mapping candidate indices to predicted scores with detailed reasoning' if self.enable_using_regressor else ''}
- "selection_reasoning": string explaining your choice (existing vs new) and why it's optimal
- "selected_candidate": object with:
  - "is_new": boolean (true if proposing new candidate, false if selecting existing)
  - "index": integer (if is_new=false, index in buffer; ignored if is_new=true)
  - "parameters": object (if is_new=true, provide new parameter values; empty object if is_new=false)

## NON-NEGOTIABLE CONSTRAINTS
**CRITICAL - VIOLATIONS WILL CAUSE REJECTION**:

### Parameter Structure Requirements (for new candidates)
- Use ONLY the parameter keys provided in the schema
- All parameter values must be strings
- Include ALL required parameters, no omissions allowed
- Follow consistent naming and structure conventions

### Content Quality Requirements
- Concrete specificity: Avoid vague phrases like "better handling"
- Actionable instructions: Provide clear, executable guidance
- Practical relevance: Content must be applicable to the optimization domain
- Length appropriateness: Match expected parameter length patterns

## Example Output Format

### Example 1: Selecting Existing Candidate
{{
  "buffer_analysis": "Parameter-Performance Analysis: Analyzed 4 candidates. Pattern Analysis: Candidates with detailed parameter content (>500 chars) show higher performance trends. Candidate 2 (score 0.78, eval_count 15) has excellent parameter structure with concrete examples. Candidates 0,1 have weaker parameter quality correlating with lower scores. Statistical Reliability: Candidate 2 has moderate reliability, others have sufficient data for assessment.",
  {'  "score_estimates": {{' if self.enable_using_regressor else ''}
  {'    "0": {{"reasoning": "Parameter analysis shows basic content (300 chars), minimal examples. Current score 0.45 matches pattern of weak parameter quality. Predicted performance reflects limited parameter effectiveness.", "predicted_score": 0.47}},' if self.enable_using_regressor else ''}
  {'    "1": {{"reasoning": "Moderate parameter content (450 chars) with some structure. Score 0.62 aligns with mid-tier parameter quality. Prediction based on consistent parameter-performance correlation.", "predicted_score": 0.64}},' if self.enable_using_regressor else ''}
  {'    "2": {{"reasoning": "Excellent parameter quality (650+ chars) with concrete examples and clear structure. Score 0.78 reflects strong parameter-performance match. High confidence in continued strong performance.", "predicted_score": 0.76}}' if self.enable_using_regressor else ''}
  {'  }},' if self.enable_using_regressor else ''}
  "selection_reasoning": "{'Predicted ranking: candidate 2 (0.76) > candidate 1 (0.64) > candidate 0 (0.47). Selecting candidate 2 because it has the highest predicted performance based on excellent parameter quality and proven track record. The parameter structure suggests continued strong performance with additional evaluation.' if self.enable_using_regressor else 'Analysis shows candidate 2 has the best combination of current performance (0.78) and parameter quality. The detailed parameter structure with concrete examples makes it the most promising candidate for continued evaluation.'}",
  "selected_candidate": {{
    "is_new": false,
    "index": 2,
    "parameters": {{}}
  }}
}}

### Example 2: Proposing New Candidate
{{
  "buffer_analysis": "Parameter-Performance Analysis: Analyzed 3 candidates with scores 0.35, 0.42, 0.38. Pattern Analysis: All candidates show similar weaknesses - vague parameter descriptions lacking concrete examples and specific guidance. No candidate demonstrates strong parameter-performance patterns. Performance Gap: Missing systematic approach to edge case handling and validation procedures.",
  {'  "score_estimates": {{' if self.enable_using_regressor else ''}
  {'    "0": {{"reasoning": "Basic parameter content with generic descriptions. Score 0.35 reflects weak parameter quality. Limited improvement potential with current approach.", "predicted_score": 0.37}},' if self.enable_using_regressor else ''}
  {'    "1": {{"reasoning": "Slightly better structure but still lacks specificity. Score 0.42 is highest among weak candidates. Marginal improvement expected.", "predicted_score": 0.44}},' if self.enable_using_regressor else ''}
  {'    "2": {{"reasoning": "Similar issues to other candidates - vague content, no concrete examples. Score 0.38 consistent with poor parameter patterns.", "predicted_score": 0.40}}' if self.enable_using_regressor else ''}
  {'  }},' if self.enable_using_regressor else ''}
  "selection_reasoning": "{'All existing candidates show consistently poor performance (0.37-0.44 predicted) due to weak parameter quality. Gap Analysis: No candidate addresses systematic validation or provides concrete operational examples. Proposing new candidate that targets these specific weaknesses with structured approach and concrete examples. Expected significant improvement over existing candidates.' if self.enable_using_regressor else 'All existing candidates show consistently poor performance (0.35-0.42 observed) due to weak parameter quality. Gap Analysis: No candidate addresses systematic validation or provides concrete operational examples. Proposing new candidate that targets these specific weaknesses with structured approach and concrete examples. Expected significant improvement over existing candidates.'}",
  "selected_candidate": {{
    "is_new": true,
    "index": -1,
    "parameters": {{
      "param1": "Each operation includes concrete validation steps with specific examples. Example format: validate_input(data) -> check_format() -> verify_constraints() -> execute_operation(). Always provide step-by-step verification procedures...",
      "param2": "SYSTEMATIC APPROACH: Before any operation, follow validation protocol: 1) Input verification, 2) Constraint checking, 3) Edge case handling, 4) Execution with monitoring. If validation fails, provide specific corrective guidance..."
    }}
  }}
}}
""",
            },
            {
                "role": "user",
                "content": f"""
## Current Buffer Data
{candidate_summaries_json}

## Parameter Schema (for new candidates)
Use exactly these parameter keys; values must be strings:
{example_param_schema_json}

## Task
Perform parameter-performance analysis and select the best candidate to evaluate:
1. Analyze existing candidates{'and predict their scores based on parameter-performance patterns' if self.enable_using_regressor else ' to understand their strengths and weaknesses'}
2. Decide whether to select an existing candidate or propose a new one
3. If proposing new candidate, ensure it addresses gaps in current candidates

Return ONLY the JSON object following the exact format shown in the examples.
""",
            },
        ]
        
        response_format = {"type": "json_object"}
        
        # Single LLM call
        def llm_call():
            return self.llm(prompt_messages, response_format=response_format)
            
        if verbose:
            print("Simple generator candidates in prompt: ", candidate_summaries_json)
        self.print_buffer_statistics()
        llm_response = retry_with_exponential_backoff(
            llm_call,
            max_retries=10,
            base_delay=1.0,
            operation_name="LLM simple generation"
        )
        
        # Default fallback
        default_entry = max(buffer_with_scores, key=lambda c: c['mean_score'])
        
        if llm_response is None:
            if verbose:
                print_color("LLM simple generation call failed after retries. Returning highest scoring candidate.", "yellow")
            return default_entry

        llm_response_str = getattr(getattr(llm_response, 'choices', [{}])[0], 'message', None)
        llm_response_str = getattr(llm_response_str, 'content', None)
        if not llm_response_str:
            if verbose:
                print_color("LLM returned empty response.", "yellow")
            return default_entry

        cleaned_llm_response_str = llm_response_str.strip()
        
        if verbose:
            print_color(f"LLM Simple Generator response: {cleaned_llm_response_str}", "cyan")
            
        try:
            llm_output = json.loads(cleaned_llm_response_str)
        except json.JSONDecodeError:
            if verbose:
                print_color("Failed to parse LLM simple generation JSON output.", "yellow")
            return default_entry

        if not isinstance(llm_output, dict):
            return default_entry

        # Extract components
        selected_candidate_info = llm_output.get("selected_candidate", {})
        
        if verbose:
            buffer_analysis = llm_output.get("buffer_analysis", "No analysis provided")
            selection_reasoning = llm_output.get("selection_reasoning", "No reasoning provided")
            print_color(f"Buffer Analysis: {buffer_analysis}", "cyan")
            print_color(f"Selection Reasoning: {selection_reasoning}", "cyan")

        # Process selection
        is_new = selected_candidate_info.get("is_new", False)
        selected_index = selected_candidate_info.get("index", 0)
        new_parameters = selected_candidate_info.get("parameters", {})
        
        if is_new and len(new_parameters) > 0:
            # Selected a new candidate - need to create buffer entry
            try:
                # Validate parameter structure for lists (same as in other generators)
                list_param_data = None
                for key, value in new_parameters.items():
                    if isinstance(value, list):
                        list_param_data = value
                        break
                
                if self.num_tools is None:
                    params_dict = buffer[0]['params']
                    for param_node, param_value in params_dict.items():
                        if isinstance(param_value, list):
                            self.num_tools = len(param_value)
                            break
                    
                if list_param_data and len(list_param_data) != self.num_tools:
                    if verbose:
                        print_color(f"New candidate has invalid list parameter length {len(list_param_data)}, expected {self.num_tools}. Falling back to best existing.", "yellow")
                    return default_entry
                
                # Convert parameters to correct format
                candidate_params_dict = self.construct_update_dict(new_parameters)
                
                # Create new buffer entry
                new_candidate_entry = {
                    "params": candidate_params_dict,
                    "score_sum": 0.0,
                    "eval_count": 0,
                    "mean_score": None,
                    "ucb_score": None,
                    "lcb_score": None
                }
                
                # Add to buffer
                buffer.append(new_candidate_entry)
                
                if verbose:
                    print_color(f"Selected NEW candidate:", "green")
                    for param_node, param_value in candidate_params_dict.items():
                        if isinstance(param_value, str):
                            print_color(f"  {param_node.py_name}: {param_value}", "green")
                
                return new_candidate_entry
                
            except Exception as e:
                if verbose:
                    print_color(f"Error creating new candidate: {e}. Falling back to best existing.", "yellow")
                return default_entry
                
        elif not is_new and 0 <= selected_index < len(buffer):
            # Selected existing candidate
            selected_entry = buffer[selected_index]
            if verbose:
                print_color(f"Selected EXISTING candidate {selected_index}", "green")
            
            return selected_entry
            
        else:
            # Invalid selection
            if verbose:
                print_color("Invalid candidate selection. Falling back to best existing.", "yellow")
            return default_entry
        