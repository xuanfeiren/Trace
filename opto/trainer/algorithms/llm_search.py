# llm_search.py
# At this stage, we want to extensively use LLM function approximator to search for the best parameters.

import numpy as np
import copy
import time
from typing import Union
from opto import trace
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
from opto.trainer.utils import retry_with_exponential_backoff, sample_minibatch
from opto.trainer.algorithms.baselines import MinibatchAlgorithm , batchify
from opto.trainer.utils import evaluate_agent

DOMAIN_CONTEXT = """## Problem Context and Domain Knowledge
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

class llm_search(MinibatchAlgorithm):
    """
    This algorithm keeps a buffer with arm statistics, and has a LLM function approximator to predict the scores of the arms. It should have the ability to predict the scores of observed and unobserved arms.
    At each epoch
    1. Select an arm with the highest predicted score. (Thompson sampling)
    2. Run a sequential search for several steps (like what MinibatchAlgorithm does) to generate new candidates. Add all the candidates to the buffer.
    3. Do several steps of evaluation on the buffer.
    4. Test the performance of the arm with the highest predicted score, periodically.
    """
    def __init__(self, agent, optimizer, num_threads: int = None, logger=None, *args, **kwargs):
        super().__init__(agent, optimizer, num_threads=num_threads, logger=logger, *args, **kwargs)
        self.buffer = deque(maxlen=200)
        self.llm_model = "gemini/gemini-2.0-flash"
        self.llm = LLM(model=self.llm_model)
        self.min_score = 0
        # initialize the buffer with the initial parameter entry
        initial_update_dict = {p: copy.deepcopy(p.data) for p in self.optimizer.parameters}
        initial_candidate_entry = {
            'params': initial_update_dict,
            'score_sum': 0,
            'eval_count': 0,
            'mean_score': None,
            'predicted_score': None
        }
        self.buffer.append(initial_candidate_entry)
        self.total_samples = 0
        self.total_proposals = 0
        self.domain_context = DOMAIN_CONTEXT

    def print_buffer_statistics(self):
        """print the buffer statistics"""
        print_color("Buffer statistics:", "magenta")
        for i,candidate_entry in enumerate(self.buffer):            
            # print the mean score, predicted score, and evaluation count.
            predicted_score = candidate_entry.get('predicted_score', 'None')
            print_color(f"Candidate {i}. Mean score {candidate_entry['mean_score']}, predicted score {predicted_score}, eval_count {candidate_entry['eval_count']}.", "green")
            # for p in candidate_entry['params']:
            #     print_color(f"Parameter value: {candidate_entry['params'][p]}", "cyan")
        return 
    
    def update_buffer_scores(self):
        """Update the buffer statistics."""
        for candidate_entry in self.buffer:
            candidate_entry['mean_score'] = candidate_entry['score_sum'] / (candidate_entry['eval_count'] or 1E-9)
        return 
    
    def predict_scores(self, buffer, verbose: bool = False, temperature: float = 0.0):
        """
        Predict scores for all candidates in the buffer.
        
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
            }
            serializable_candidate_summaries.append(summary)
        candidate_summaries_json = json.dumps(serializable_candidate_summaries, indent=2)
        
        example_param_schema_json = json.dumps({p.py_name: copy.deepcopy(p.data) for p in self.agent.parameters()}, indent=2)

        # Create the score prediction prompt (function approximation and denoising)
        example_format = '''{{
  "pattern_analysis": "[Provide detailed analysis of what patterns you discovered across all candidates. Analyze parameter characteristics, identify similarities and differences, examine how observed scores relate to parameter features. Discuss your reasoning process for identifying correlations and your confidence in different patterns.]",
  "function_mapping": {{
    "discovered_patterns": ["[List any parameter-performance patterns you identified]"],
    "similarity_groups": ["[Group similar candidates and explain why they are similar]"],
    "uncertainty_notes": "[Discuss what patterns are unclear or uncertain]"
  }},
  "score_estimates": {{
    "0": {{"reasoning": "[Provide thorough analysis: examine parameters in detail, compare to other candidates, explain how you arrived at prediction, discuss confidence level, explain any denoising logic]", "predicted_score": 0.XX}},
    "1": {{"reasoning": "[Detailed reasoning for this candidate...]", "predicted_score": 0.XX}},
    "[...continue for all candidates...]": {{"reasoning": "[Always provide extensive reasoning explaining your analysis process]", "predicted_score": 0.XX}}
  }}
}}'''

        prompt_messages = [
            {
                "role": "system",
                "content": f"""
{self.domain_context}

## Function Approximation Objective
You are a **parameter-to-score function approximator**. Your goal is to learn the underlying mapping from candidate parameters to their true performance scores, using observed data to build this mapping and apply it to all candidates.

## Core Capabilities
1. **Pattern Learning**: Extract parameter-performance correlations from observed data
2. **Function Mapping**: Build a parameter → score mapping function from patterns
3. **Noise Reduction**: Use cross-candidate patterns to denoise observed scores
4. **Score Prediction**: Apply learned function to predict scores for all candidates (observed and unobserved)

## Key Insights for Function Approximation
- **Observed scores contain noise**: Raw scores may not reflect true performance due to evaluation variance
- **Parameters reveal true performance**: Similar parameters should yield similar scores
- **Cross-candidate learning**: Information from one candidate can improve predictions for others
- **Pattern-based denoising**: Use parameter similarities to correct noisy observations

## Analysis Approach

### Step 1: Deep Data Examination
**Thoroughly analyze** all available data:
- **Parameter inspection**: Carefully examine each candidate's parameters in detail
- **Score relationships**: Look for any relationships between parameters and observed scores
- **Cross-candidate comparison**: Compare similar and different candidates
- **Pattern exploration**: Look for potential patterns, but don't force them if unclear

### Step 2: Reasoning-Based Prediction
**Focus on comprehensive reasoning** rather than rigid rules:
- **Detailed analysis**: For each candidate, provide extensive reasoning about parameter quality
- **Similarity assessment**: Compare candidates and explain similarities/differences
- **Uncertainty acknowledgment**: Be honest about what is unclear or uncertain
- **Evidence-based prediction**: Base predictions on thorough analysis, not assumed patterns

### Step 3: Thorough Documentation
**Provide extensive reasoning** for all predictions:
- **Analysis process**: Explain how you examined the parameters
- **Comparison logic**: Describe how you compared candidates
- **Prediction rationale**: Justify your score predictions with detailed reasoning
- **Confidence assessment**: Discuss your confidence level and any uncertainties

## Prediction Methodology
1. **For observed candidates**: Use parameter patterns to denoise raw scores
   - If raw score seems inconsistent with parameter quality, adjust based on similar candidates
   - Consider evaluation count (higher count = more reliable, but still may need correction)
2. **For unobserved candidates**: Use parameter-based function approximation
   - Find candidates with similar parameter profiles
   - Apply learned parameter-performance mappings
   - Predict score based on parameter quality indicators

## Output Requirements
Return ONLY a JSON object with these fields:
- "pattern_analysis": **Provide extensive analysis** of what you observe in the data. Examine parameter characteristics across candidates, discuss how observed scores relate to parameters, explain your reasoning process. Be thorough and detailed in your analysis.
- "function_mapping": Document any patterns you discovered (even if uncertain), group similar candidates, and note areas of uncertainty. Don't force patterns if they're not clear.
- "score_estimates": For each candidate, provide **detailed reasoning** explaining your analysis process, parameter evaluation, cross-candidate comparisons, and how you arrived at your prediction. Reasoning should be comprehensive and thorough.

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
**Function Approximation Challenge**: Analyze the relationship between parameters and performance, then predict scores for ALL candidates through detailed reasoning.

**Your Mission**:
1. **Thoroughly examine** all candidate parameters and any available score data
2. **Provide extensive reasoning** for each prediction based on your detailed analysis
3. **Compare candidates** to identify similarities and differences that might inform predictions
4. **Consider noise** in observed scores and use cross-candidate insights where helpful
5. **Focus on reasoning quality** over discovering specific patterns - be thorough in your analysis

**Key Approach**: Provide comprehensive, detailed reasoning for each prediction. Don't force patterns if they're not clear - focus on thorough analysis and honest assessment of what you observe.

**Critical**: Each candidate's reasoning should be extensive and detailed. Quality of reasoning is more important than finding specific patterns.

Return ONLY the JSON object with your detailed analysis and thoroughly reasoned score predictions.
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
            pattern_analysis = llm_output.get("pattern_analysis", "No pattern analysis provided")
            function_mapping = llm_output.get("function_mapping", "No function mapping provided")
            print_color(f"Pattern Analysis: {pattern_analysis}", "cyan")
            print_color(f"Function Mapping: {function_mapping}", "magenta")
            print_color(f"Score Estimates: {score_estimates}", "blue")

        # Convert score estimates to numpy array and update buffer entries
        predicted_scores = []
        for idx in range(len(buffer)):
            candidate_key = str(idx)
            if candidate_key in score_estimates:
                try:
                    predicted_score = score_estimates[candidate_key].get("predicted_score", buffer[idx].get('mean_score', 0.0))
                    predicted_score_float = float(predicted_score)
                    predicted_scores.append(predicted_score_float)
                    # Update the buffer entry with the predicted score
                    buffer[idx]['predicted_score'] = predicted_score_float
                except (ValueError, TypeError):
                    # Fallback to mean score if prediction is invalid
                    print_color(f"Invalid predicted score for candidate {idx}: {score_estimates[candidate_key]}", "yellow")
                    fallback_score = buffer[idx].get('mean_score', 0.0)
                    predicted_scores.append(fallback_score)
                    # Update the buffer entry with the fallback score
                    buffer[idx]['predicted_score'] = fallback_score
            else:
                # Fallback to mean score if no prediction available
                print_color(f"No predicted score for candidate {idx}", "yellow")
                fallback_score = buffer[idx].get('mean_score', 0.0)
                predicted_scores.append(fallback_score)
                # Update the buffer entry with the fallback score
                buffer[idx]['predicted_score'] = fallback_score
        
        predicted_scores_array = np.array(predicted_scores)
        
        if verbose:
            print_color(f"Predicted scores: {predicted_scores_array}", "green")
            print_color(f"Mean scores (fallback): {default_scores}", "yellow")
            # print_color(f"Ground truth scores: {self.ground_truth_scores}", "blue")
        
        

        return predicted_scores_array
    
    def update(self, outputs, verbose=False, num_threads=None, **kwargs):
        """
        I made some modifications to the original update method.
        return the average score of the minibatch of inputs, and the new update dictionary.
        """

        num_threads = num_threads or self.num_threads  # Use provided num_threads or fall back to self.num_threads

        scores, targets, feedbacks = [], [], []
        # Concatenate the targets and feedbacks into a single string
        for target, score, feedback in outputs:
            scores.append(score)
            targets.append(target)
            feedbacks.append(feedback)
        target = batchify(*targets)
        feedback = batchify(*feedbacks).data  # 
        # old version
        # average_score = np.mean(scores) if all([s is not None for s in scores]) else None
        # new version: using all non-None scores to compute the mean score.
        valid_scores = [s for s in scores if s is not None]
        # If all scores are None, return 0
        average_score = np.mean(valid_scores) if valid_scores else 0

        # Update the agent using the feedback
        self.optimizer.zero_feedback()
        self.optimizer.backward(target, feedback)
        step_kwargs = dict(bypassing=True, verbose='output' if verbose else False)
        step_kwargs = dict(bypassing=True, verbose='output')


        def optimizer_step_func():
            return self.optimizer.step(**step_kwargs)
        
        new_update_dict = retry_with_exponential_backoff(
            optimizer_step_func, 
            operation_name="Optimizer step (UCB parameter generation)"
        )

        return average_score, new_update_dict 
    
    def generate_new_candidates(self, train_batch_size: int = 2, num_steps: int = 5):
        """Generate new candidates. Default to be, select the arm with the highest predicted score, then do a sequential search for several steps (like what MinibatchAlgorithm does) to generate new candidates. Create entries and add all the candidates to the buffer."""
        # select the arm with the highest predicted score, update the agent with the selected arm.
        selected_candidate_entry = max(self.buffer, key=lambda x: x.get('predicted_score', 0.0))
        self.optimizer.update(selected_candidate_entry['params'])

        # do a sequential search for several steps (like what MinibatchAlgorithm does) to generate new candidates.
        for _ in range(num_steps):
            current_update_dict = {p: copy.deepcopy(p.data) for p in self.optimizer.parameters}
            # sample a minibatch from the train dataset
            xs, infos = self._sample_minibatch(self.train_dataset, train_batch_size)
            # forward the agent
            forward = batch_run(max_workers=self.num_threads, description=f"Forward pass (batch size: {len(xs)})")(self.forward)
            outputs = forward(self.agent, xs, self.guide, infos)
            # Update the agent
            _, new_update_dict = self.update(outputs)
            # The new update dict may only contain part of the parameters, we need to merge it with the current update dict
            for p in current_update_dict:
                if p not in new_update_dict:
                    new_update_dict[p] = current_update_dict[p]
            # create a new candidate entry
            new_candidate_entry = {
                'params': new_update_dict,
                'score_sum': 0,
                'eval_count': 0,
                'mean_score': None,
                'predicted_score': None
            }
            self.buffer.append(new_candidate_entry)
            # update the agent
            self.optimizer.update(new_update_dict)
        self.total_proposals += num_steps
        return 
        
    
    def buffer_evaluation(self,validate_batch_size: int = 20):
        """Evaluate several candidates in the buffer. Default to be, evaluating all arms without statistics."""
        # sample a subset of the self.validate_dataset
        xs,infos = self._sample_minibatch(self.validate_dataset, validate_batch_size)
        # create a validate_subset with the same structure as the self.validate_dataset
        validate_subset = {'inputs': xs, 'infos': infos}
        for candidate_entry in self.buffer:
            if candidate_entry['eval_count'] == 0: # evaluate all unobserved arms
                # Update agent with candidate's parameters before evaluation
                self.optimizer.update(candidate_entry['params'])
                score = evaluate_agent(self.agent, self.guide, validate_subset, num_threads=self.num_threads, num_eval_times=1)
                candidate_entry['eval_count'] = validate_batch_size
                candidate_entry['score_sum'] = score*validate_batch_size
                self.total_samples += validate_batch_size
        return

    def train(self,
              guide,
              train_dataset,
              *,
              num_epochs: int = 20,  # number of training epochs
              batch_size: int = 2,  # batch size for updating the agent
              validate_batch_size: int = 20,  # batch size for validating the agent
              validate_dataset = None,  # dataset of (x, info) pairs to validate the agent
              test_dataset = None,  # dataset of (x, info) pairs to evaluate the agent
              eval_frequency: int = 10,  # frequency of evaluation
              num_eval_samples: int = 5,  # number of samples to use to evaluate each input
              num_generation_steps: int = 5,  # number of steps to generate new candidates
              log_frequency: Union[int, None] = None,  # frequency of logging
              save_frequency: Union[int, None] = None,  # frequency of saving the agent
              save_path: str = "checkpoints/agent.pkl",  # path to save the agent
              min_score: Union[int, None] = None,  # minimum score to update the agent
              verbose: Union[bool, str] = False,  # whether to print the output of the agent
              num_threads: int = None,  # maximum number of threads to use (overrides self.num_threads)
              **kwargs
              ):
        """
        At each epoch, the algorithm will:
        1. Select an arm with the highest predicted score. (Thompson sampling)
        2. Run a sequential search for several steps (like what MinibatchAlgorithm does) to generate new candidates. Add all the candidates to the buffer.
        3. Do several steps of evaluation on the buffer.
        4. Test the performance of the arm with the highest predicted score, periodically.
        """
        self.train_dataset = train_dataset
        self.validate_dataset = validate_dataset
        self.guide = guide
        # Initial evaluation and update the initial test score
        if eval_frequency > 0:
            eval_scores = evaluate_agent(self.agent, guide, test_dataset, num_threads=num_threads, num_eval_times=num_eval_samples)
            self.logger.log('Test score', eval_scores, 0, color='green')
            self.logger.log('Total samples', self.total_samples, 0, color='cyan')
            self.logger.log('Total proposals', self.total_proposals, 0, color='magenta')

        for epoch in range(num_epochs):
            print_color(f"Epoch {epoch+1} of {num_epochs}", "magenta")
            self.update_buffer_scores()
            self.print_buffer_statistics()
            self.predict_scores(self.buffer, verbose=verbose)
            self.generate_new_candidates(train_batch_size=batch_size, num_steps=num_generation_steps)
            self.buffer_evaluation(validate_batch_size=validate_batch_size)

            if (epoch+1) % eval_frequency == 0:
                self.update_buffer_scores()
                self.print_buffer_statistics()
                self.predict_scores(self.buffer, verbose=verbose)
                # select the candidate with the highest predicted score
                best_candidate_entry = max(self.buffer, key=lambda x: x.get('predicted_score', 0.0) if x.get('predicted_score') is not None else 0.0)
                self.optimizer.update(best_candidate_entry['params'])
                # evaluate the best candidate on the test dataset
                test_score = evaluate_agent(self.agent, guide, test_dataset, num_threads=num_threads, num_eval_times=num_eval_samples)
                self.logger.log('Selected candidate predicted score', best_candidate_entry['predicted_score'], epoch+1, color='blue')
                self.logger.log('Test score', test_score, epoch+1, color='green')
                self.logger.log('Total samples', self.total_samples, epoch+1, color='cyan')
                self.logger.log('Total proposals', self.total_proposals, epoch+1, color='magenta')
        self.logger.log('Final instruction', best_candidate_entry['params'][0], epoch+1, color='magenta')
        print_color("Training completed.", "green")



    
    