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
    def __init__(self, agent, optimizer, num_threads: int = None, logger=None,select_arm_by_predicted_score: bool = True, num_multiple_generations: int = 1, do_validation: bool = True, *args, **kwargs):
        super().__init__(agent, optimizer, num_threads=num_threads, logger=logger, *args, **kwargs)
        self.buffer = deque(maxlen=500)
        self.llm_model = "gemini/gemini-2.0-flash"
        self.llm = LLM(model=self.llm_model)
        self.min_score = 0
        # initialize the buffer with the initial parameter entry
        initial_update_dict = {p: copy.deepcopy(p.data) for p in self.optimizer.parameters}
        initial_candidate_entry = {
            'params': initial_update_dict,
            'score_sum': 0,
            'eval_count': 0,
            'mean_score': 0,
            'predicted_score': None,
            'will_be_evaluated': True,
            'num_validation':0,
        }
        self.buffer.append(initial_candidate_entry)
        self.total_samples = 0
        self.total_proposals = 0
        self.domain_context = DOMAIN_CONTEXT
        # flags to do ablation study
        # 1. select_arm_by_predicted_score: whether to select the arm by predicted score or mean score.
        # 2. num_multiple_generations: the number of times to call OptoPrime optimizer to generate more candidates. 
        # 3. do_validation: default to be True. If false, the algorithm only using training data to update the buffer statistics.
        self.select_arm_by_predicted_score = select_arm_by_predicted_score
        self.num_multiple_generations = num_multiple_generations
        self.do_validation = do_validation

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
    
    def get_fallback_score(self, entry):
        """
        Get fallback score for an entry based on the priority order:
        1. Use existing positive predicted_score if available
        2. Use non-negative mean_score if available
        3. Use 0 as final fallback
        
        Args:
            entry: Candidate entry dictionary
            
        Returns:
            float: Fallback score
        """
        # First check if there's already a positive predicted score
        existing_predicted = entry.get('predicted_score')
        if existing_predicted is not None and existing_predicted > 0:
            return existing_predicted
        
        # Then check for non-negative mean score
        mean_score = entry.get('mean_score', 0.0)
        if mean_score >= 0:
            return mean_score
        
        # Final fallback to 0
        return 0.0
    
    def predict_scores(self, buffer, verbose: bool = False, temperature: float = 0.0):
        """Default to use XML format."""
        return self.predict_scores_xml(buffer, verbose, temperature)
    
    def predict_scores_json(self, buffer, verbose: bool = False, temperature: float = 0.0):
        """
        Predict scores for all candidates in the buffer.
        
        Args:
            buffer: List of candidate entries with parameters and statistics
            verbose: Whether to print verbose output and debugging information
            temperature: Temperature parameter for LLM sampling (0.0 = deterministic, higher = more random)
            
        Returns:
            np.array: Vector of predicted scores, defaults to fallback scores if LLM fails
        """
        # Create a shuffled copy of buffer for randomized LLM presentation
        import random
        shuffled_buffer_with_original_idx = [(i, entry) for i, entry in enumerate(buffer)]
        random.shuffle(shuffled_buffer_with_original_idx)
        shuffled_buffer = [entry for _, entry in shuffled_buffer_with_original_idx]
        # Create mapping from shuffled index to original index
        shuffled_to_original_idx = {shuffled_idx: original_idx for shuffled_idx, (original_idx, _) in enumerate(shuffled_buffer_with_original_idx)}
        
        # Prepare serializable candidate summaries with parameters using shuffled order
        serializable_candidate_summaries = []
        self.update_buffer_scores()
        for idx, cand_entry in enumerate(shuffled_buffer):
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
        
        # Default fallback: return fallback scores using improved logic
        default_scores = np.array([self.get_fallback_score(c) for c in buffer])

        try:
            llm_response = retry_with_exponential_backoff(
                llm_call,
                max_retries=10,
                base_delay=1.0,
                operation_name="LLM score prediction"
            )
        except Exception as e:
            print_color(f"WARNING: LLM score prediction call failed: {e}, returning fallback scores.", "red")
            # Update buffer entries with fallback scores when LLM fails
            for idx in range(len(buffer)):
                fallback_score = self.get_fallback_score(buffer[idx])
                buffer[idx]['predicted_score'] = fallback_score
            return default_scores
        
        
        llm_response_str = getattr(getattr(llm_response, 'choices', [{}])[0], 'message', None)
        llm_response_str = getattr(llm_response_str, 'content', None)
        if not llm_response_str:
            print_color("WARNING: LLM returned empty response for score prediction. Using fallback scores.", "red")
            # Update buffer entries with fallback scores even when LLM returns empty response
            for idx in range(len(buffer)):
                fallback_score = self.get_fallback_score(buffer[idx])
                buffer[idx]['predicted_score'] = fallback_score
            return default_scores

        cleaned_llm_response_str = llm_response_str.strip()
        
        if verbose:
            self.print_buffer_statistics()
            print_color(f"LLM Score Prediction (temperature={temperature}): {cleaned_llm_response_str}", "cyan")
            
        try:
            llm_output = json.loads(cleaned_llm_response_str)
        except json.JSONDecodeError:
            print_color("WARNING: Failed to parse LLM score prediction JSON output. Using fallback scores.", "red")
            # Update buffer entries with fallback scores even when JSON parsing fails
            for idx in range(len(buffer)):
                fallback_score = self.get_fallback_score(buffer[idx])
                buffer[idx]['predicted_score'] = fallback_score
            return default_scores

        if not isinstance(llm_output, dict):
            print_color("WARNING: LLM output is not a valid dictionary. Using fallback scores.", "red")
            # Update buffer entries with fallback scores even when LLM output is invalid
            for idx in range(len(buffer)):
                fallback_score = self.get_fallback_score(buffer[idx])
                buffer[idx]['predicted_score'] = fallback_score
            return default_scores

        # Extract score estimates
        score_estimates = llm_output.get("score_estimates", {})
        
        if verbose:
            pattern_analysis = llm_output.get("pattern_analysis", "No pattern analysis provided")
            function_mapping = llm_output.get("function_mapping", "No function mapping provided")
            print_color(f"Pattern Analysis: {pattern_analysis}", "cyan")
            print_color(f"Function Mapping: {function_mapping}", "magenta")
            print_color(f"Score Estimates: {score_estimates}", "blue")

        # Process predictions on shuffled_buffer and assign predicted scores
        for idx in range(len(shuffled_buffer)):
            candidate_key = str(idx)
            entry = shuffled_buffer[idx]
            original_idx = shuffled_to_original_idx[idx]
            
            if candidate_key in score_estimates:
                try:
                    predicted_score = score_estimates[candidate_key].get("predicted_score", self.get_fallback_score(entry))
                    predicted_score_float = float(predicted_score)
                    entry['predicted_score'] = predicted_score_float
                except (ValueError, TypeError):
                    print_color(f"WARNING: Invalid predicted score for candidate {idx} (original #{original_idx}): {score_estimates[candidate_key]}, using fallback score.", "red")
                    fallback_score = self.get_fallback_score(entry)
                    entry['predicted_score'] = fallback_score
            else:
                print_color(f"WARNING: No predicted score for candidate {idx} (original #{original_idx}), using fallback score.", "red")
                fallback_score = self.get_fallback_score(entry)
                entry['predicted_score'] = fallback_score
        
        # Return predicted scores in original buffer order
        predicted_scores = [entry.get('predicted_score', 0.0) for entry in buffer]
        
        predicted_scores_array = np.array(predicted_scores)
        
        if verbose:
            print_color(f"Predicted scores: {predicted_scores_array}", "green")
            print_color(f"Mean scores (fallback): {default_scores}", "yellow")
            # print_color(f"Ground truth scores: {self.ground_truth_scores}", "blue")
        
        

        return predicted_scores_array 
    
    def predict_scores_xml(self, buffer, verbose: bool = False, temperature: float = 0.0):
        """
        Predict scores for all candidates in the buffer using XML format.
        This XML version does EXACTLY the same thing as predict_scores but uses XML instead of JSON.
        
        Args:
            buffer: List of candidate entries with parameters and statistics
            verbose: Whether to print verbose output and debugging information
            temperature: Temperature parameter for LLM sampling (0.0 = deterministic, higher = more random)
            
        Returns:
            np.array: Vector of predicted scores, defaults to mean scores if LLM fails
        """
        import xml.etree.ElementTree as ET
        from xml.etree.ElementTree import ParseError
        import re
        
        # Create a shuffled copy of buffer for randomized LLM presentation
        import random
        shuffled_buffer_with_original_idx = [(i, entry) for i, entry in enumerate(buffer)]
        random.shuffle(shuffled_buffer_with_original_idx)
        shuffled_buffer = [entry for _, entry in shuffled_buffer_with_original_idx]
        # Create mapping from shuffled index to original index
        shuffled_to_original_idx = {shuffled_idx: original_idx for shuffled_idx, (original_idx, _) in enumerate(shuffled_buffer_with_original_idx)}
        
        # Prepare serializable candidate summaries with parameters using shuffled order
        # This matches EXACTLY what the JSON version does
        serializable_candidate_summaries = []
        self.update_buffer_scores()
        for idx, cand_entry in enumerate(shuffled_buffer):
            summary = {
                "index": idx,
                "parameters": {k.py_name: v for k,v in cand_entry['params'].items()},
                "eval_count": cand_entry['eval_count'],
                "mean_score": cand_entry['mean_score'],
            }
            serializable_candidate_summaries.append(summary)
        
        # Convert to XML instead of JSON
        candidates_xml = "<candidates>\n"
        for summary in serializable_candidate_summaries:
            candidates_xml += f"  <candidate index='{summary['index']}'>\n"
            candidates_xml += f"    <eval_count>{summary['eval_count']}</eval_count>\n"
            candidates_xml += f"    <mean_score>{summary['mean_score']}</mean_score>\n"
            candidates_xml += "    <parameters>\n"
            for param_name, param_value in summary['parameters'].items():
                # Escape XML special characters
                param_value_escaped = str(param_value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&apos;')
                candidates_xml += f"      <parameter name='{param_name}'><![CDATA[{param_value_escaped}]]></parameter>\n"
            candidates_xml += "    </parameters>\n"
            candidates_xml += "  </candidate>\n"
        candidates_xml += "</candidates>"
        
        # Create example parameter schema XML (matches the JSON version logic)
        example_param_schema_xml = "<parameter_schema>\n"
        example_param_dict = {p.py_name: copy.deepcopy(p.data) for p in self.agent.parameters()}
        for param_name, param_value in example_param_dict.items():
            param_value_escaped = str(param_value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&apos;')
            example_param_schema_xml += f"  <parameter name='{param_name}'><![CDATA[{param_value_escaped}]]></parameter>\n"
        example_param_schema_xml += "</parameter_schema>"

        # Create the score prediction prompt using XML format
        example_format = '''<prediction_result>
    <pattern_analysis>
        [Provide detailed analysis of what patterns you discovered across all candidates. Analyze parameter characteristics, identify similarities and differences, examine how observed scores relate to parameter features. Discuss your reasoning process for identifying correlations and your confidence in different patterns.]
    </pattern_analysis>
    <function_mapping>
        <discovered_patterns>
        <pattern>[List any parameter-performance patterns you identified]</pattern>
        </discovered_patterns>
        <similarity_groups>
        <group>[Group similar candidates and explain why they are similar]</group>
        </similarity_groups>
        <uncertainty_notes>[Discuss what patterns are unclear or uncertain]</uncertainty_notes>
    </function_mapping>
    <score_estimates>
        <candidate index="0">
        <reasoning>[Provide thorough analysis: examine parameters in detail, compare to other candidates, explain how you arrived at prediction, discuss confidence level, explain any denoising logic]</reasoning>
        <predicted_score>0.XX</predicted_score>
        </candidate>
        <candidate index="1">
        <reasoning>[Detailed reasoning for this candidate...]</reasoning>
        <predicted_score>0.XX</predicted_score>
        </candidate>
    </score_estimates>
    </prediction_result>'''

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
    Return ONLY an XML structure with these elements:
    - <pattern_analysis>: **Provide extensive analysis** of what you observe in the data
    - <function_mapping>: Document any patterns you discovered and group similar candidates
    - <score_estimates>: For each candidate, provide **detailed reasoning** and predicted score

    ## Example Output Format
    {example_format}

    **CRITICAL**: Ensure all XML tags are properly closed. If you run out of response space, prioritize completing the current candidate element before stopping.
    """,
            },
            {
                "role": "user", 
                "content": f"""
    ## Candidate Data
    {candidates_xml}

    ## Parameter Schema
    {example_param_schema_xml}

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

    Return ONLY the XML structure with your detailed analysis and thoroughly reasoned score predictions.
    """,
            },
        ]
        # Default fallback: return fallback scores using improved logic (SAME as JSON version)
        default_scores = np.array([self.get_fallback_score(c) for c in buffer])

        # Multi-call LLM function to handle incomplete responses
        def multi_call_llm():
            """
            Call LLM multiple times to get complete response if needed.
            First call uses original prompt, subsequent calls append previous response to continue.
            """
            current_messages = prompt_messages.copy()
            full_response = ""
            max_continuation_calls = 10  # Limit to prevent infinite loops
            
            for call_num in range(max_continuation_calls):
                def single_llm_call():
                    return self.llm(current_messages, temperature=temperature)
                
                try:
                    llm_response = retry_with_exponential_backoff(
                        single_llm_call,
                        max_retries=12,
                        base_delay=1.0,
                        operation_name=f"LLM score prediction (XML) call {call_num + 1}"
                    )
                    
                    response_str = getattr(getattr(llm_response, 'choices', [{}])[0], 'message', None)
                    response_str = getattr(response_str, 'content', None)
                    
                    # print the response_str for debug
                    # print_color(f"Response str: {response_str}", "yellow")

                    if not response_str:
                        if call_num == 0:
                            raise Exception("LLM returned empty response")
                        else:
                            break  # No more content to continue
                    
                    response_str = response_str.strip()
                    full_response += response_str
                    
                    # Check if response is complete by looking for proper XML structure
                    # For Gemini 2.0 Flash, we primarily rely on the presence of closing tag
                    if '</prediction_result>' in full_response:
                        break
                        
                    # Additional check: if this individual response is very short, 
                    # it might indicate the model finished naturally (not truncated)
                    if len(response_str.strip()) < 50:  # Very short response
                        print_color(f"Call {call_num + 1}: Short response received, assuming completion", "yellow")
                        break
                        
                    # Prepare continuation prompt using full accumulated response
                    continuation_prompt = {
                        "role": "user",
                        "content": f"Continue from where you left off. Your response so far was:\n\n{full_response}\n\nPlease continue and complete the remaining candidates' score predictions in the same XML format. Do not repeat what you already provided, just continue from where you stopped."
                    }
                    
                    current_messages = prompt_messages.copy()
                    current_messages.append({"role": "assistant", "content": full_response})
                    current_messages.append(continuation_prompt)

                    # if call_num >0: # print the continuation behavior for debug
                    #     print_color(f"Current messages: {current_messages}", "yellow")

                    
                except Exception as e:
                    if call_num == 0:
                        raise e  # Re-raise original exception for first call
                    else:
                        print_color(f"WARNING: Continuation call {call_num + 1} failed: {e}, using partial response.", "yellow")
                        break
            
            # Create mock response object with combined content
            class MockMessage:
                def __init__(self, content):
                    self.content = content
                    
            class MockChoice:
                def __init__(self, message):
                    self.message = message
                    
            class MockResponse:
                def __init__(self, choices):
                    self.choices = choices
                    
            return MockResponse([MockChoice(MockMessage(full_response))])

        try:
            llm_response = multi_call_llm()
        except Exception as e:
            print_color(f"WARNING: LLM score prediction call failed: {e}, returning fallback scores.", "red")
            # Update buffer entries with fallback scores when LLM fails (SAME as JSON version)
            for idx in range(len(buffer)):
                fallback_score = self.get_fallback_score(buffer[idx])
                buffer[idx]['predicted_score'] = fallback_score
            return default_scores
        
        llm_response_str = getattr(getattr(llm_response, 'choices', [{}])[0], 'message', None)
        llm_response_str = getattr(llm_response_str, 'content', None)
        if not llm_response_str:
            print_color("WARNING: LLM returned empty response for score prediction. Using fallback scores.", "red")
            # Update buffer entries with fallback scores (SAME as JSON version)
            for idx in range(len(buffer)):
                fallback_score = self.get_fallback_score(buffer[idx])
                buffer[idx]['predicted_score'] = fallback_score
            return default_scores

        cleaned_llm_response_str = llm_response_str.strip()
        
        if verbose:
            self.print_buffer_statistics()
            print_color(f"LLM Score Prediction XML (temperature={temperature}): {cleaned_llm_response_str}", "cyan")
        
        # Robust XML parsing with multiple fallback strategies
        def robust_xml_parse(xml_content):
            """Parse XML content robustly, handling truncated outputs"""
            xml_content = xml_content.strip()
            
            # Strategy 1: Try to extract complete prediction_result
            prediction_match = re.search(r'<prediction_result>.*?</prediction_result>', xml_content, re.DOTALL)
            if prediction_match:
                xml_content = prediction_match.group(0)
            elif '<prediction_result>' in xml_content:
                # Strategy 2: Handle truncated XML - find the start and try to fix
                start_idx = xml_content.find('<prediction_result>')
                xml_content = xml_content[start_idx:]
                xml_content = fix_unclosed_xml_tags(xml_content)
            else:
                # Strategy 3: Look for score_estimates section directly
                estimates_match = re.search(r'<score_estimates>.*?</score_estimates>', xml_content, re.DOTALL)
                if estimates_match:
                    xml_content = f"<prediction_result>{estimates_match.group(0)}</prediction_result>"
                elif '<score_estimates>' in xml_content:
                    start_idx = xml_content.find('<score_estimates>')
                    estimates_content = xml_content[start_idx:]
                    estimates_content = fix_unclosed_xml_tags(estimates_content)
                    xml_content = f"<prediction_result>{estimates_content}</prediction_result>"
            
            try:
                root = ET.fromstring(xml_content)
                return parse_prediction_xml(root)
            except ParseError as e:
                if verbose:
                    print_color(f"XML Parse Error, trying partial parsing: {e}", "yellow")
                return parse_partial_xml(xml_content)
        
        def fix_unclosed_xml_tags(xml_content):
            """Fix unclosed tags more comprehensively"""
            # Add closing prediction_result tag if missing
            if '</prediction_result>' not in xml_content and '<prediction_result>' in xml_content:
                xml_content += '</prediction_result>'
            
            # Close unclosed score_estimates section
            if '<score_estimates>' in xml_content and '</score_estimates>' not in xml_content:
                # Count unclosed candidate tags and close them first
                open_candidates = xml_content.count('<candidate') - xml_content.count('</candidate>')
                xml_content += '</candidate>' * open_candidates
                xml_content += '</score_estimates>'
            
            # Close unclosed candidate tags if no score_estimates wrapper
            elif '<candidate' in xml_content:
                open_candidates = xml_content.count('<candidate') - xml_content.count('</candidate>')
                xml_content += '</candidate>' * open_candidates
            
            # Close unclosed reasoning tags
            open_reasoning = xml_content.count('<reasoning>') - xml_content.count('</reasoning>')
            xml_content += '</reasoning>' * open_reasoning
            
            # Close unclosed predicted_score tags  
            open_scores = xml_content.count('<predicted_score>') - xml_content.count('</predicted_score>')
            xml_content += '</predicted_score>' * open_scores
            
            return xml_content
        
        def parse_prediction_xml(root):
            """Parse well-formed prediction XML to match JSON structure exactly"""
            score_estimates = {}
            
            # Extract score estimates
            score_estimates_elem = root.find('score_estimates')
            if score_estimates_elem is not None:
                for candidate_elem in score_estimates_elem.findall('candidate'):
                    index = candidate_elem.get('index')
                    if index is not None:
                        reasoning_elem = candidate_elem.find('reasoning')
                        score_elem = candidate_elem.find('predicted_score')
                        
                        reasoning = reasoning_elem.text if reasoning_elem is not None and reasoning_elem.text else ""
                        try:
                            predicted_score = float(score_elem.text) if score_elem is not None and score_elem.text else 0.0
                        except (ValueError, TypeError):
                            predicted_score = 0.0
                        
                        # Match JSON structure exactly: nested dict with predicted_score and reasoning
                        score_estimates[index] = {
                            'reasoning': reasoning,
                            'predicted_score': predicted_score
                        }
            
            return score_estimates
        
        def parse_partial_xml(xml_content):
            """Extract data from partially formed XML using regex"""
            score_estimates = {}
            
            # Try multiple regex patterns to handle different truncation scenarios
            patterns = [
                # Complete candidate with both reasoning and score
                r'<candidate[^>]*index=["\'](\d+)["\'][^>]*>.*?<reasoning>(.*?)</reasoning>.*?<predicted_score>(.*?)</predicted_score>',
                # Candidate with only reasoning (score truncated)
                r'<candidate[^>]*index=["\'](\d+)["\'][^>]*>.*?<reasoning>(.*?)</reasoning>(?!.*<predicted_score>)',
                # Candidate with only score (reasoning truncated) 
                r'<candidate[^>]*index=["\'](\d+)["\'][^>]*>.*?<predicted_score>(.*?)</predicted_score>(?!.*<reasoning>)',
            ]
            
            for pattern in patterns:
                for match in re.finditer(pattern, xml_content, re.DOTALL):
                    index = match.group(1)
                    if index not in score_estimates:  # Don't overwrite complete matches
                        if len(match.groups()) == 3:  # Complete match
                            reasoning = match.group(2).strip()
                            try:
                                predicted_score = float(match.group(3).strip())
                            except (ValueError, TypeError):
                                predicted_score = 0.0
                        elif 'reasoning' in pattern:  # Only reasoning
                            reasoning = match.group(2).strip()
                            predicted_score = 0.0
                        else:  # Only score
                            reasoning = ""
                            try:
                                predicted_score = float(match.group(2).strip())
                            except (ValueError, TypeError):
                                predicted_score = 0.0
                        
                        score_estimates[index] = {
                            'reasoning': reasoning,
                            'predicted_score': predicted_score
                        }
            
            return score_estimates
        
        # Parse the XML response
        try:
            score_estimates = robust_xml_parse(cleaned_llm_response_str)
        except Exception as e:
            print_color(f"WARNING: Failed to parse LLM XML output: {e}. Using fallback scores.", "red")
            # Update buffer entries with fallback scores when XML parsing fails
            for idx in range(len(buffer)):
                fallback_score = self.get_fallback_score(buffer[idx])
                buffer[idx]['predicted_score'] = fallback_score
            return default_scores

        # EXACT same logic as JSON version for verbose output
        if verbose:
            # Extract additional info for verbose output (if available)
            try:
                root = ET.fromstring(cleaned_llm_response_str)
                pattern_elem = root.find('pattern_analysis')
                mapping_elem = root.find('function_mapping')
                
                pattern_analysis = pattern_elem.text if pattern_elem is not None else "No pattern analysis provided"
                function_mapping = ET.tostring(mapping_elem, encoding='unicode') if mapping_elem is not None else "No function mapping provided"
                
                print_color(f"Pattern Analysis: {pattern_analysis}", "cyan")
                print_color(f"Function Mapping: {function_mapping}", "magenta")
            except:
                pass  # Skip verbose extras if XML parsing fails
            
            print_color(f"Score Estimates: {score_estimates}", "blue")

        # Process predictions on shuffled_buffer and assign predicted scores
        # EXACT same logic as JSON version
        for idx in range(len(shuffled_buffer)):
            candidate_key = str(idx)
            entry = shuffled_buffer[idx]
            original_idx = shuffled_to_original_idx[idx]
            
            if candidate_key in score_estimates:
                try:
                    # FIXED: Use .get() method like JSON version to avoid KeyError
                    predicted_score = score_estimates[candidate_key].get("predicted_score", self.get_fallback_score(entry))
                    predicted_score_float = float(predicted_score)
                    entry['predicted_score'] = predicted_score_float
                except (ValueError, TypeError):
                    print_color(f"WARNING: Invalid predicted score for candidate {idx} (original #{original_idx}): {score_estimates[candidate_key]}, using fallback score.", "red")
                    fallback_score = self.get_fallback_score(entry)
                    entry['predicted_score'] = fallback_score
            else:
                print_color(f"WARNING: No predicted score for candidate {idx} (original #{original_idx}), using fallback score.", "red")
                fallback_score = self.get_fallback_score(entry)
                entry['predicted_score'] = fallback_score
        
        # Return predicted scores in original buffer order (SAME as JSON version)
        predicted_scores = [entry.get('predicted_score', 0.0) for entry in buffer]
        predicted_scores_array = np.array(predicted_scores)
        
        if verbose:
            print_color(f"Predicted scores: {predicted_scores_array}", "green")
            print_color(f"Mean scores (fallback): {default_scores}", "yellow")
        
        return predicted_scores_array

    def update(self, outputs, verbose=False, num_threads=None, **kwargs):
        """
        I made some modifications to the original update method.
        return the average score of the minibatch of inputs, and the new update dictionary/dictionaries.
        If self.num_multiple_generations == 1, returns (average_score, new_update_dict)
        If self.num_multiple_generations > 1, returns (average_score, [update_dict1, update_dict2, ...])
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

        # General case: generate multiple update_dicts with retry logic
        def single_optimizer_step_with_retry():
            def optimizer_step_func():
                return self.optimizer.step(**step_kwargs)
            try:
                return retry_with_exponential_backoff(
                    optimizer_step_func, 
                    operation_name="Optimizer step"
                )
            except Exception as e:
                print(f"Optimizer step failed after retries: {e}. Returning None.")
                return None
        
        # Create list of functions for async_run (works for both single and multiple generations)
        runs = [single_optimizer_step_with_retry] * self.num_multiple_generations
        
        # Run optimizer steps asynchronously (or sequentially if num_multiple_generations=1)
        try:
            update_dicts = async_run(
                runs,
                max_workers=self.num_threads,
                description=f"Generating {self.num_multiple_generations} parameter updates"
            )
            # Filter out None results from failed optimizer steps
            update_dicts = [update_dict for update_dict in update_dicts if update_dict is not None]
            
            # If all optimizer steps failed, return empty list
            if not update_dicts:
                print("All optimizer steps failed. Returning empty update_dicts.")
                update_dicts = []
                
        except Exception as e:
            print(f"Async run for optimizer steps failed: {e}. Returning empty update_dicts.")
            update_dicts = []
        
        return average_score, update_dicts 
    
    def select_starting_point_entry(self):
        """Select the starting point entry. Default to be the arm with the highest predicted score."""
        return 
        
    def generate_new_candidates(self, starting_point_entry, train_batch_size: int = 2, num_steps: int = 4):
        """Generate new candidates. Default to be, select the arm with the highest predicted score, then do a sequential search for several steps (like what MinibatchAlgorithm does) to generate new candidates. Create entries and add all the candidates to the buffer.
        Also use the training data to update the current entry.
        """
        # select the arm with the highest predicted score, update the agent with the selected arm.
        selected_candidate_entry = starting_point_entry
        self.optimizer.update(selected_candidate_entry['params'])

        current_entry = selected_candidate_entry
        # do a sequential search for several steps (like what MinibatchAlgorithm does) to generate new candidates.
        for iter in range(num_steps):
            current_update_dict = current_entry['params']
            # sample a minibatch from the train dataset
            # xs, infos = self._sample_minibatch(self.train_dataset, train_batch_size)
            # another choice: choose the training data sequentially 
            xs = self.train_dataset['inputs'][iter*train_batch_size:(iter+1)*train_batch_size]
            infos = self.train_dataset['infos'][iter*train_batch_size:(iter+1)*train_batch_size]
            # forward the agent
            forward = batch_run(max_workers=self.num_threads, description=f"Forward pass (batch size: {len(xs)})")(self.forward)
            outputs = forward(self.agent, xs, self.guide, infos)
            # Update the agent
            score, update_dicts = self.update(outputs)
            # update the current entry with the new score
            current_entry['score_sum'] += score*len(xs)
            current_entry['eval_count'] += len(xs)
            self.total_samples += len(xs)
            
            # update_dicts is always a list now. update dicts may be empty if all optimizer steps failed.
            for i, new_update_dict in enumerate(update_dicts):
                # The new update dict may only contain part of the parameters, we need to merge it with the current update dict
                for p in current_update_dict:
                    if p not in new_update_dict:
                        new_update_dict[p] = current_update_dict[p]
                # create a new candidate entry
                # First one has will_be_evaluated=True, others have will_be_evaluated=False
                new_candidate_entry = {
                    'params': new_update_dict,
                    'score_sum': 0,
                    'eval_count': 0,
                    'mean_score': 0,
                    'predicted_score': None,
                    'will_be_evaluated': True if i == 0 else False,
                    'num_validation': 0
                }
                self.buffer.append(new_candidate_entry)
                # Update current_entry to the first one for next iteration
                if i == 0:
                    current_entry = new_candidate_entry
                    # update the agent with the first update_dict
                    self.optimizer.update(new_update_dict)

            
        self.total_proposals += num_steps*self.num_multiple_generations
        return 
    
    def single_candidate_evaluation(self, candidate_entry, validate_subset):
        """Evaluate a single candidate. Update the candidate entry with the evaluation result."""
        self.optimizer.update(candidate_entry['params'])
        score = evaluate_agent(self.agent, self.guide, validate_subset, num_threads=self.num_threads, num_eval_times=1)
        candidate_entry['eval_count'] += len(validate_subset['inputs'])
        candidate_entry['score_sum'] += score*len(validate_subset['inputs'])
        candidate_entry['num_validation'] += 1
        self.total_samples += len(validate_subset['inputs'])
    
    def buffer_evaluation(self,starting_point_entry, validate_batch_size: int = 20):
        """Evaluate several candidates in the buffer. Default to be, evaluating all arms without validation. Also evaluate the starting point the algorithm selected."""
        
        # sample a subset of the self.validate_dataset
        xs,infos = self._sample_minibatch(self.validate_dataset, validate_batch_size)
        # create a validate_subset with the same structure as the self.validate_dataset
        validate_subset = {'inputs': xs, 'infos': infos}
        # First evaluate the starting point entry. Then evaluate one generated candidate at each generation step.
        self.single_candidate_evaluation(starting_point_entry, validate_subset)
        # Then evaluate the generated candidates.
        for candidate_entry in self.buffer:
            if candidate_entry['num_validation'] == 0 and candidate_entry['will_be_evaluated']: # evaluate all unvalidated arms that will be evaluated
                self.single_candidate_evaluation(candidate_entry, validate_subset)
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
            # update the buffer scores
            self.update_buffer_scores()
            # Could decide whether to select the arm by predicted score or mean score. If by predicted score, the algorithm would predict the scores for all the candidates in the buffer, and select the arm with the highest predicted score.
            if self.select_arm_by_predicted_score:
                # For all the candidates in the buffer, predict the scores.
                self.predict_scores(self.buffer, verbose=verbose)
                starting_point_entry = max(self.buffer, key=lambda x: x.get('predicted_score', 0.0) if x.get('predicted_score') is not None else 0.0)
            else:
                starting_point_entry = max(self.buffer, key=lambda x: x.get('mean_score', 0.0))
            self.print_buffer_statistics()
            self.generate_new_candidates(starting_point_entry, train_batch_size=batch_size, num_steps=num_generation_steps)

            if self.do_validation: # could do validation or not.
                self.buffer_evaluation(starting_point_entry, validate_batch_size=validate_batch_size)

            if (epoch+1) % eval_frequency == 0:
                self.update_buffer_scores()
                # self.print_buffer_statistics()
                if self.select_arm_by_predicted_score:
                    self.predict_scores(self.buffer, verbose=verbose)
                    best_candidate_entry = max(self.buffer, key=lambda x: x.get('predicted_score', 0.0) if x.get('predicted_score') is not None else 0.0)
                else:
                    best_candidate_entry = max(self.buffer, key=lambda x: x.get('mean_score', 0.0))
                self.optimizer.update(best_candidate_entry['params'])
                # evaluate the best candidate on the test dataset
                test_score = evaluate_agent(self.agent, guide, test_dataset, num_threads=num_threads, num_eval_times=num_eval_samples)
                
                # Log detailed statistics for the selected candidate
                self.logger.log('Selected candidate predicted score', best_candidate_entry.get('predicted_score', 'None'), epoch+1, color='blue')
                self.logger.log('Selected candidate mean score', best_candidate_entry.get('mean_score', 0.0), epoch+1, color='blue')
                self.logger.log('Selected candidate eval count', best_candidate_entry.get('eval_count', 0), epoch+1, color='blue')
                
                
                
                # Log overall performance metrics
                self.logger.log('Test score', test_score, epoch+1, color='green')
                self.logger.log('Total samples', self.total_samples, epoch+1, color='cyan')
                self.logger.log('Total proposals', self.total_proposals, epoch+1, color='magenta')
        # Log final candidate statistics
        param_values = list(best_candidate_entry['params'].values())
        self.logger.log('Final parameters', param_values, epoch+1, color='magenta')
        self.logger.log('Final candidate predicted score', best_candidate_entry.get('predicted_score', 'None'), epoch+1, color='magenta')
        self.logger.log('Final candidate mean score', best_candidate_entry.get('mean_score', 0.0), epoch+1, color='magenta')
        self.logger.log('Final candidate eval count', best_candidate_entry.get('eval_count', 0), epoch+1, color='magenta')
        
        
        
        
        
        print_color("Training completed.", "green")



    
    