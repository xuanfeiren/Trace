# A LLM regressor to predict scores for candidates.
# A simple version: construct a prompt with the buffer statistics (candidate parameters and scores), ask LLM to analyze the pattern and predict scores for all candidates (candidates with noisy scores and new candidates without any statistics).
# Issues for the simple version:
# 1. Output limit is not enough: when the output limit of LLM is reached, the LLM with stop the generation, only part of the candidates have predicted scores. Asking LLM to continue the generation would solve this issue in some sense.
# 2. When we have a large buffer, the prompt is too long. LLM cannot do reasoning properly. It would output nonsenses, like repeating some numbers without any logic. 
# 3. LLM may skip some candidates due to reason 1 or 2.

# The key to fix these issues is to use less candidates in the prompt, and only predict part of the candidates that need predicted scores.
# When we need to predict scores for a batch of candidates, we first divide the batch into smaller batches, and predict scores for each smaller batch. 
# For each smaller batch, we sample a subset of candidates with statistics to construct the prompt, call LLM to make the prediction. To make the predicition more reliable, we repeat this process multiple times (with different subsets in the prompt) and take the average.
# Parallelize the process, then it only take the time of one LLM call the get the predicted scores for a small batch of candidates. Also parallelize the prediction for different batches.
import numpy as np
import copy
from typing import Union
from opto.trainer.loader import DataLoader
from opto.trainer.utils import batch_run, async_run
from opto.optimizers.utils import print_color
# from opto.trainer.evaluators import evaluate
from typing import Union, List, Tuple, Dict, Any, Optional
from collections import deque
from opto.utils.llm import LLM # For the selector LLM
# from opto.trace.nodes import ParameterNode
import json
# import warnings
# from black import format_str, FileMode
import random
# import mathX
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
                    Better parameter configurations lead to higher task success rates. The goal is to find parameter settings that maximize agent performance across diverse scenarios in the target domain.
                 """
class Regressor:
    """
    A LLM regressor to predict scores for a batch of candidates.
    """
    def __init__(self, model_name = "gemini/gemini-2.0-flash", temperature = 0.0, buffer = None, max_candidates_per_prompt = 50, max_candidates_to_predict = 20, num_repetitions = 5):
        self.LLM = LLM(model=model_name)
        self.buffer = buffer
        self.max_candidates_per_prompt = max_candidates_per_prompt
        self.max_candidates_to_predict = max_candidates_to_predict
        self.num_repetitions = num_repetitions

    def predict_scores(self):
        """Predict scores for all candidates in the buffer. It contains the candidates with noisy observed statistics and new candidates without any statistics.
        
        Divide the buffer into smaller batches with at most max_candidates_to_predict candidates.

        For each smaller batch, sample a subset of candidates with statistics to construct the prompt, call LLM to make the prediction. To make the predicition more reliable, we repeat this process multiple times (with different subsets in the prompt) and take the average.

        Parallelize the process, then it only take the time of one LLM call the get the predicted scores for a small batch of candidates. Also parallelize the prediction for different batches.
        """
        # Divide the buffer into smaller batches with at most max_candidates_to_predict candidates.
        # Convert deque to list to support slicing
        buffer_list = list(self.buffer)
        batches = [buffer_list[i:i+self.max_candidates_to_predict] for i in range(0, len(buffer_list), self.max_candidates_to_predict)]



        # For each smaller batch, sample a subset of candidates with statistics to construct the prompt, call LLM to make the prediction. To make the predicition more reliable, we repeat this process multiple times (with different subsets in the prompt) and take the average.
        for batch in batches:
            # TODO: Parallelize the process.
            self.predict_scores_for_batch(batch)
        return 
    
    def sample_minibatch(self):
        """Sample a subset of candidates with statistics to construct the prompt."""
        # Extract all candidates with statistics from the buffer.
        candidates_with_statistics = [candidate for candidate in self.buffer if candidate['eval_count'] > 0]
        batch_size = min(self.max_candidates_per_prompt, len(candidates_with_statistics))
        # Randomly sample a subset of candidates with statistics.
        subset = random.sample(candidates_with_statistics, batch_size)
        return subset

    def call_regressor(self, subset_with_statistics, batch_to_predict):
        """Call the regressor to make the prediction. Randomly shuffle the subset_with_statistics to construct the prompt, then predict the scores for the batch_to_predict. Return a vector of scores for the batch_to_predict."""
        import xml.etree.ElementTree as ET
        from xml.etree.ElementTree import ParseError
        import re
        
        # Randomly shuffle the training subset for randomized LLM presentation
        shuffled_subset = subset_with_statistics.copy()
        random.shuffle(shuffled_subset)
        
        # Update scores for subset_with_statistics
        for candidate_entry in shuffled_subset:
            candidate_entry['mean_score'] = candidate_entry['score_sum'] / (candidate_entry['eval_count'] or 1E-9)
        
        # Default fallback: return zeros for all candidates to predict
        default_scores = np.zeros(len(batch_to_predict))
        
        # Prepare XML for subset_with_statistics (training data)
        if not shuffled_subset:
            # No training data available
            training_candidates_xml = "<training_candidates>\n  <note>No available data</note>\n</training_candidates>"
            serializable_training_summaries = []
        else:
            # Prepare serializable training candidate summaries
            serializable_training_summaries = []
            for idx, cand_entry in enumerate(shuffled_subset):
                summary = {
                    "index": idx,
                    "parameters": {k.py_name if hasattr(k, 'py_name') else str(k): v for k, v in cand_entry['params'].items()},
                    "eval_count": cand_entry['eval_count'],
                    "mean_score": cand_entry['mean_score'],
                }
                serializable_training_summaries.append(summary)
            
            # Build XML from training summaries
            training_candidates_xml = "<training_candidates>\n"
            for summary in serializable_training_summaries:
                training_candidates_xml += f"  <candidate index='{summary['index']}'>\n"
                training_candidates_xml += f"    <eval_count>{summary['eval_count']}</eval_count>\n"
                training_candidates_xml += f"    <mean_score>{summary['mean_score']}</mean_score>\n"
                training_candidates_xml += "    <parameters>\n"
                for param_name, param_value in summary['parameters'].items():
                    # Escape XML special characters
                    param_value_escaped = str(param_value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&apos;')
                    training_candidates_xml += f"      <parameter name='{param_name}'><![CDATA[{param_value_escaped}]]></parameter>\n"
                training_candidates_xml += "    </parameters>\n"
                training_candidates_xml += "  </candidate>\n"
            training_candidates_xml += "</training_candidates>"
        
        # Prepare serializable prediction candidate summaries
        serializable_prediction_summaries = []
        for idx, cand_entry in enumerate(batch_to_predict):
            summary = {
                "index": idx,
                "parameters": {k.py_name if hasattr(k, 'py_name') else str(k): v for k, v in cand_entry['params'].items()},
            }
            serializable_prediction_summaries.append(summary)
        
        # Prepare XML for batch_to_predict (candidates to predict)
        prediction_candidates_xml = "<prediction_candidates>\n"
        for summary in serializable_prediction_summaries:
            prediction_candidates_xml += f"  <candidate index='{summary['index']}'>\n"
            prediction_candidates_xml += "    <parameters>\n"
            for param_name, param_value in summary['parameters'].items():
                # Escape XML special characters
                param_value_escaped = str(param_value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&apos;')
                prediction_candidates_xml += f"      <parameter name='{param_name}'><![CDATA[{param_value_escaped}]]></parameter>\n"
            prediction_candidates_xml += "    </parameters>\n"
            prediction_candidates_xml += "  </candidate>\n"
        prediction_candidates_xml += "</prediction_candidates>"
        
        # Create example parameter schema XML
        if serializable_training_summaries:
            example_param_dict = copy.deepcopy(serializable_training_summaries[0]['parameters'])
        elif serializable_prediction_summaries:
            example_param_dict = copy.deepcopy(serializable_prediction_summaries[0]['parameters'])
        else:
            example_param_dict = {}
        
        example_param_schema_xml = "<parameter_schema>\n"
        for param_name, param_value in example_param_dict.items():
            param_value_escaped = str(param_value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&apos;')
            example_param_schema_xml += f"  <parameter name='{param_name}'><![CDATA[{param_value_escaped}]]></parameter>\n"
        example_param_schema_xml += "</parameter_schema>"

        # Create the score prediction prompt using XML format
        example_format = '''<prediction_result>
        <pattern_analysis>
            [Analyze the training candidates to identify patterns between parameters and observed scores.]
        </pattern_analysis>
        <function_mapping>
            <discovered_patterns>
            <pattern>[List parameter-performance patterns discovered from training data]</pattern>
            </discovered_patterns>
            <similarity_groups>
            <group>[Group similar training candidates and explain relationships]</group>
            </similarity_groups>
            <uncertainty_notes>[Discuss what patterns are unclear or uncertain]</uncertainty_notes>
        </function_mapping>
        <score_estimates>
            <candidate index="0">
            <reasoning>[Provide thorough analysis and prediction reasoning]</reasoning>
            <predicted_score>0.XX</predicted_score>
            </candidate>
        </score_estimates>
        </prediction_result>'''

        prompt_messages = [
            {
                "role": "system",
                "content": f"""
        {DOMAIN_CONTEXT}

        ## Function Approximation Objective
        You are a **parameter-to-score function approximator**. Your goal is to learn the mapping from candidate parameters to their performance scores using training data, then apply this learned function to predict scores for new candidates.

        ## Core Capabilities
        1. **Pattern Learning**: Extract parameter-performance correlations from training candidates with observed scores
        2. **Function Mapping**: Build a parameter → score mapping function from discovered patterns
        3. **Score Prediction**: Apply learned function to predict scores for new candidates without observed scores

        ## Analysis Approach
        1. **Analyze training data**: Examine training candidates' parameters and observed scores to discover patterns
        2. **Learn function mapping**: Build understanding of how parameters influence performance scores
        3. **Apply to new candidates**: Use learned patterns to predict scores for prediction candidates

        ## Output Requirements
        Return ONLY an XML structure with pattern analysis, function mapping, and score estimates for prediction candidates.

        ## Example Output Format
        {example_format}

        **CRITICAL**: Ensure all XML tags are properly closed. Focus on learning from training data to predict scores for new candidates.
        """,
            },
            {
                "role": "user", 
                "content": f"""
        ## Training Data (Candidates with Observed Scores)
        {training_candidates_xml}

        ## Prediction Candidates (Need Score Predictions)
        {prediction_candidates_xml}

        ## Parameter Schema
        {example_param_schema_xml}

        ## Task
        Learn the parameter-performance relationship from training candidates, then predict scores for the prediction candidates.

        Return ONLY the XML structure with your detailed analysis and score predictions for the prediction candidates.
        """,
        },
        ]
        
        # Call LLM with retry logic
        def single_llm_call():
            return self.LLM(prompt_messages, temperature=0.0)
        print_color(prompt_messages, "blue")
        try:
            llm_response = retry_with_exponential_backoff(
                single_llm_call,
                max_retries=10,
                base_delay=1.0,
                operation_name="Regressor LLM call"
            )
        except Exception as e:
            print_color(f"WARNING: Regressor LLM call failed: {e}, returning default scores.", "red")
            return default_scores
        
        llm_response_str = getattr(getattr(llm_response, 'choices', [{}])[0], 'message', None)
        llm_response_str = getattr(llm_response_str, 'content', None)
        print_color(llm_response_str, "green")
        if not llm_response_str:
            print_color("WARNING: Regressor LLM returned empty response. Using default scores.", "red")
            return default_scores

        cleaned_llm_response_str = llm_response_str.strip()
        
        # Parse XML response
        def parse_xml_response(xml_content):
            """Parse XML response to extract predicted scores"""
            score_estimates = {}
            
            # Try to extract score_estimates section
            estimates_match = re.search(r'<score_estimates>.*?</score_estimates>', xml_content, re.DOTALL)
            if estimates_match:
                estimates_section = estimates_match.group(0)
                
                # Extract individual candidate scores
                candidate_pattern = r'<candidate[^>]*index=["\'](\d+)["\'][^>]*>.*?<predicted_score>(.*?)</predicted_score>'
                for match in re.finditer(candidate_pattern, estimates_section, re.DOTALL):
                    index = match.group(1)
                    try:
                        predicted_score = float(match.group(2).strip())
                    except (ValueError, TypeError):
                        predicted_score = 0.0
                    score_estimates[index] = predicted_score
            
            return score_estimates
        
        try:
            score_estimates = parse_xml_response(cleaned_llm_response_str)
        except Exception as e:
            print_color(f"WARNING: Failed to parse regressor XML output: {e}. Using default scores.", "red")
            return default_scores

        # Extract predicted scores in the order they appear in the batch
        predicted_scores = []
        for idx in range(len(batch_to_predict)):
            candidate_key = str(idx)
            
            if candidate_key in score_estimates:
                predicted_score = score_estimates[candidate_key]
            else:
                predicted_score = 0.0
            
            predicted_scores.append(predicted_score)
        
        return np.array(predicted_scores)
        
    def predict_scores_for_batch(self, batch):
        """Predict scores for a batch of candidates. Update the buffer with the predicted scores."""
        # TODO: Parallelize the process.
        predicted_scores_all_rounds = []
        for round in range(self.num_repetitions):
            # Sample a subset of candidates with statistics to construct the prompt.
            subset = self.sample_minibatch()
            # Call LLM to make the prediction.
            predicted_scores_in_this_round = self.call_regressor(subset, batch)
            predicted_scores_all_rounds.append(predicted_scores_in_this_round)
        
        # Calculate the average predicted scores across all rounds
        avg_predicted_scores = np.mean(predicted_scores_all_rounds, axis=0)
        
        # For each candidate in the batch, add the predicted score to the buffer.
        for candidate, predicted_score in zip(batch, avg_predicted_scores):
            candidate['predicted_score'] = predicted_score

