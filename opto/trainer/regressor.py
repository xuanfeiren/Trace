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
import litellm
import time

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
    def __init__(self, model_name = "gemini/gemini-2.0-flash", temperature = 0.0, buffer = None, max_candidates_per_prompt = 50, max_candidates_to_predict = 20, num_repetitions = 5,num_threads = None):
        self.LLM = LLM(model=model_name)
        self.buffer = buffer
        self.max_candidates_per_prompt = max_candidates_per_prompt
        self.max_candidates_to_predict = max_candidates_to_predict
        self.num_repetitions = num_repetitions
        self.num_threads = num_threads

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
        if hasattr(self, 'num_threads') and self.num_threads and self.num_threads > 1:
            # Parallelize batch processing
            batch_functions = [lambda batch=b: self.predict_scores_for_batch(batch) for b in batches]
            async_run(
                batch_functions,
                max_workers=self.num_threads,
                description=f"Processing {len(batches)} candidate batches"
            )
        else:
            # Sequential processing
            for batch in batches:
                self.predict_scores_for_batch(batch)
        # Return the predicted scores for the buffer.
        predicted_scores_for_the_buffer = [candidate['predicted_score'] for candidate in buffer_list]
        return np.array(predicted_scores_for_the_buffer)
    
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
            candidate_entry['squared_score_sum'] = candidate_entry['score_sum']
            candidate_entry['score_variance'] = candidate_entry['squared_score_sum'] / (candidate_entry['eval_count'] or 1E-9) - candidate_entry['mean_score']**2
        
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
                    "score_variance": cand_entry['score_variance'],
                }
                serializable_training_summaries.append(summary)
            
            # Build XML from training summaries
            training_candidates_xml = "<training_candidates>\n"
            for summary in serializable_training_summaries:
                training_candidates_xml += f"  <candidate index='{summary['index']}'>\n"
                training_candidates_xml += f"    <eval_count>{summary['eval_count']}</eval_count>\n"
                training_candidates_xml += f"    <mean_score>{summary['mean_score']}</mean_score>\n"
                training_candidates_xml += f"    <score_variance>{summary['score_variance']}</score_variance>\n"
                training_candidates_xml += "    <parameters>\n"
                for param_name, param_value in summary['parameters'].items():
                    # Escape XML special characters
                    param_value_escaped = str(param_value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&apos;')
                    training_candidates_xml += f"      <parameter name='{param_name}'><![CDATA[{param_value_escaped}]]></parameter>\n"
                training_candidates_xml += "    </parameters>\n"
                training_candidates_xml += "  </candidate>\n"
            training_candidates_xml += "</training_candidates>"
        
        # Randomly shuffle batch_to_predict and keep track of original order
        shuffled_prediction_with_original_idx = [(i, entry) for i, entry in enumerate(batch_to_predict)]
        random.shuffle(shuffled_prediction_with_original_idx)
        shuffled_prediction_batch = [entry for _, entry in shuffled_prediction_with_original_idx]
        # Create mapping from shuffled index to original index
        shuffled_to_original_idx = {shuffled_idx: original_idx for shuffled_idx, (original_idx, _) in enumerate(shuffled_prediction_with_original_idx)}
        
        # Prepare serializable prediction candidate summaries using shuffled order
        serializable_prediction_summaries = []
        for idx, cand_entry in enumerate(shuffled_prediction_batch):
            summary = {
                "index": idx,
                "parameters": {k.py_name if hasattr(k, 'py_name') else str(k): v for k, v in cand_entry['params'].items()},
                "eval_count": cand_entry.get('eval_count', 0),
                "mean_score": cand_entry.get('mean_score', 0.0),
                "score_variance": cand_entry.get('score_variance', 0.0),
            }
            serializable_prediction_summaries.append(summary)
        
        # Prepare XML for batch_to_predict (candidates to predict)
        prediction_candidates_xml = "<prediction_candidates>\n"
        for summary in serializable_prediction_summaries:
            prediction_candidates_xml += f"  <candidate index='{summary['index']}'>\n"
            # prediction_candidates_xml += f"    <eval_count>{summary['eval_count']}</eval_count>\n"
            # prediction_candidates_xml += f"    <mean_score>{summary['mean_score']}</mean_score>\n"
            # prediction_candidates_xml += f"    <score_variance>{summary['score_variance']}</score_variance>\n"
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
        <Deep Data Examination>
            [Comprehensive analysis of ALL training data - examine all training candidates' parameters and scores, identify parameter-performance relationships, understand what makes parameters effective or ineffective, compare similar and different candidates to understand patterns, build overall understanding that will inform all predictions]
        </Deep Data Examination>
        <score_estimates>
            <candidate index="0">
            <reasoning>[Detailed reasoning for this specific candidate - analyze the parameters thoroughly, compare to similar training examples from your analysis above, explain your logic for the predicted score, discuss confidence level and any uncertainty]</reasoning>
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
        **First, thoroughly analyze ALL available training data:**
        - Examine all training candidates' parameters and their observed scores
        - Look for relationships between parameter characteristics and performance
        - Identify what makes parameters effective or ineffective
        - Compare similar and different candidates to understand patterns
        - Build overall understanding of the parameter-performance relationship

        ### Step 2: Individual Candidate Reasoning
        **Then, for each prediction candidate, provide detailed reasoning:**
        - Analyze the candidate's specific parameters thoroughly
        - Compare to similar training examples from your analysis
        - Explain your reasoning for the predicted score
        - Be honest about uncertainty and confidence level

        ## Prediction Methodology
        1. **For training candidates**: Use parameter patterns to denoise raw scores
        - If raw score seems inconsistent with parameter quality, adjust based on similar candidates
        - Consider eval_count (higher count = more reliable, but still may need correction. The mean_score is the empirical success rate on eval_count tasks. So if eval_count is very small, the mean_score may not be reliable.)
        2. **For prediction candidates**: Use parameter-based function approximation
        - Find candidates with similar parameter profiles from training data
        - Apply learned parameter-performance mappings
        - Predict score based on parameter quality indicators

        ## Output Requirements
        Return ONLY an XML structure with these two elements:
        - <Deep Data Examination>: **Comprehensive analysis of ALL training data** - Examine all training candidates' parameters and scores, identify parameter-performance relationships, understand what makes parameters effective, and build foundational insights for predictions.
        - <score_estimates>: **For each prediction candidate, provide detailed reasoning** - Analyze the specific parameters, compare to training examples, explain prediction logic, and assess confidence level. Each candidate needs thorough reasoning.

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
        **Your Mission**: Learn from training data to predict scores for all prediction candidates.

        **Simple Process**:
        1. **Deep Data Examination**: First, thoroughly analyze ALL training data - examine all candidates' parameters and scores, understand what makes parameters effective, identify patterns and relationships.
        2. **Individual Reasoning**: Then, for each prediction candidate, provide detailed reasoning - analyze the specific parameters, compare to training examples, explain your prediction logic.

        **Key Points**: 
        - Start with comprehensive analysis of all training data
        - Provide thorough reasoning for each individual prediction
        - Be honest about uncertainty and confidence levels

        Return ONLY the XML structure with your analysis and reasoned score predictions.
        """,
        },
        ]
        
        # Call LLM with retry logic
        def single_llm_call():
            return self.LLM(prompt_messages, temperature=0.0)
        # print_color(prompt_messages, "blue")
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
        # print_color(llm_response_str, "green")
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

        # Extract predicted scores in original batch order
        predicted_scores = []
        for idx in range(len(shuffled_prediction_batch)):
            candidate_key = str(idx)
            original_idx = shuffled_to_original_idx[idx]
            
            if candidate_key in score_estimates:
                predicted_score = score_estimates[candidate_key]
            else:
                predicted_score = 0.0
            
            predicted_scores.append((original_idx, predicted_score))
        
        # Sort by original index to maintain order
        predicted_scores.sort(key=lambda x: x[0])
        predicted_scores = [score for _, score in predicted_scores]
        
        return np.array(predicted_scores)
        
    def predict_scores_for_batch(self, batch):
        """Predict scores for a batch of candidates. Update the buffer with the predicted scores."""
        if hasattr(self, 'num_threads') and self.num_threads and self.num_threads > 1:
            # Parallelize the repetitions
            def single_round():
                subset = self.sample_minibatch()
                return self.call_regressor(subset, batch)
            
            round_functions = [single_round for _ in range(self.num_repetitions)]
            predicted_scores_all_rounds = async_run(
                round_functions,
                max_workers=self.num_threads,
                description=f"Running {self.num_repetitions} prediction rounds"
            )
        else:
            # Sequential processing
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
        # Return the average predicted scores.
        return avg_predicted_scores

class EmbeddingRegressor:
    """
    Predict scores using embedding logistic regression. 
    Should have two key methods: predict_scores and predict_scores_for_batch. 
    predict_scores has no parameters, it could return predicted scores for all candidates in the buffer. 
    predict_scores_for_batch has one parameter, a batch of candidates, it could return predicted scores for the batch of candidates."""
    def __init__(self, buffer = None, embedding_model="gemini/text-embedding-004", num_threads = None, learning_rate=0.2, regularization_strength=1e-4, max_iterations=20000, tolerance=5e-3):
        # In the regressor, no need for calling LLM to make the prediction. So we could predict the entire buffer at once.
        self.max_candidates_to_predict = 500
        self.buffer = buffer
        self.embedding_model = embedding_model
        self.num_threads = num_threads
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.regularization_strength = regularization_strength  # L2 regularization strength (lambda)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.patience = 20  # Early stopping patience
        self.lr_decay_factor = 0.8   # Learning rate decay factor
        # default linear dimension is 768
        self.linear_dim = 768
        # Initialize weights with larger values for more aggressive learning
        self.weights = np.random.normal(0, 0.1, self.linear_dim)
        self.bias = 0.0
        
    def _sigmoid(self, z):
        """Sigmoid activation function for logistic regression."""
        return 1.0 / (1.0 + np.exp(-z))

    def _get_embedding(self, entry):
        """Get the embedding for an entry."""
        additional_instructions = list(entry["params"].values())[0]
        
        def single_embedding_call():
            return litellm.embedding(
                model=self.embedding_model,
                input=additional_instructions
            )
        
        try:
            response = retry_with_exponential_backoff(
                single_embedding_call,
                max_retries=10,
                base_delay=1.0,
                operation_name="Embedding API call"
            )
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            print_color(f"ERROR: Embedding API call failed after retries: {e}", "red")
            # Return a random embedding as fallback to prevent complete failure
            print_color("Using random embedding as fallback", "yellow")
            fallback_embedding = np.random.normal(0, 0.01, self.linear_dim)
            return fallback_embedding / np.linalg.norm(fallback_embedding)
    
    def _update_buffer_embeddings(self):
        """Update the embeddings for the buffer."""
        for entry in self.buffer:
            if hasattr(entry, "embedding"):
                continue
            entry["embedding"] = self._get_embedding(entry)
    
    def _update_regression_model(self):
        """Update the regression model using the current buffer with logistic regression."""
        start_time = time.time()
        print_color("Updating regression model using the current buffer with logistic regression...", "blue")
        self._update_buffer_embeddings()
        
        # Get training data from buffer (only entries with evaluation data)
        training_entries = [entry for entry in self.buffer if entry.get('eval_count', 0) > 0]
        
        if len(training_entries) == 0:
            print_color("Warning: No training data available for regression model.", "yellow")
            end_time = time.time()
            elapsed_time = end_time - start_time
            print_color(f"_update_regression_model completed in {elapsed_time:.4f} seconds (no training data)", "cyan")
            return
            
        # Extract raw binary training data from each candidate
        X_list = []
        y_list = []
        
        for entry in training_entries:
            embedding = entry["embedding"]
            eval_count = entry['eval_count']
            score_sum = entry['score_sum']
            
            # score_sum directly represents the number of successes
            num_successes = int(score_sum)
            num_failures = eval_count - num_successes
            
            # Create binary training samples: 1 for success, 0 for failure
            for _ in range(num_successes):
                X_list.append(embedding)
                y_list.append(1.0)
            
            for _ in range(num_failures):
                X_list.append(embedding)
                y_list.append(0.0)
        
        if len(X_list) == 0:
            print_color("Warning: No binary training samples generated.", "yellow")
            end_time = time.time()
            elapsed_time = end_time - start_time
            print_color(f"_update_regression_model completed in {elapsed_time:.4f} seconds (no binary samples)", "cyan")
            return
            
        # Convert to numpy arrays
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Ensure X has the right dimensions
        if X.shape[1] != self.linear_dim:
            self.linear_dim = X.shape[1]
            # Initialize weights with larger values for more aggressive learning
            self.weights = np.random.normal(0, 0.1, self.linear_dim)
        
        # Convergence-based regularized logistic regression training using all raw binary data
        m = len(X_list)
        # print_color(f"Training regularized logistic regression with {m} binary samples from {len(training_entries)} candidates until convergence.", "blue")
        # print_color(f"Using L2 regularization strength: {self.regularization_strength}, learning rate: {self.learning_rate}", "blue")
        # print_color(f"Max iterations: {self.max_iterations}, tolerance: {self.tolerance}", "blue")
        
        # Debug: Print initial weight statistics
        initial_weight_norm = np.linalg.norm(self.weights)
        # print_color(f"Initial weight norm: {initial_weight_norm:.6f}", "yellow")
        
        # Debug: Print embedding statistics
        embedding_mean = np.mean(X)
        embedding_std = np.std(X)
        embedding_norm_mean = np.mean([np.linalg.norm(row) for row in X])
        # print_color(f"Embedding stats - mean: {embedding_mean:.6f}, std: {embedding_std:.6f}, avg norm: {embedding_norm_mean:.6f}", "yellow")
        
        # Training loop until convergence with adaptive learning rate and early stopping
        prev_cost = float('inf')
        best_cost = float('inf')
        converged = False
        iteration = 0
        patience_counter = 0
        
        # Reset learning rate
        self.learning_rate = self.initial_learning_rate
        
        for iteration in range(self.max_iterations):
            # Forward pass
            z = X.dot(self.weights) + self.bias
            predictions = self._sigmoid(z)
            
            # Compute cost with L2 regularization
            epsilon = 1e-15  # Small value to prevent log(0)
            predictions_clipped = np.clip(predictions, epsilon, 1 - epsilon)
            log_likelihood = -np.mean(y * np.log(predictions_clipped) + (1 - y) * np.log(1 - predictions_clipped))
            l2_penalty = self.regularization_strength * np.sum(self.weights ** 2)
            total_cost = log_likelihood + l2_penalty
            
            # Check for improvement and early stopping
            cost_change = abs(prev_cost - total_cost)
            if total_cost < best_cost:
                best_cost = total_cost
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Backward pass (compute gradients with L2 regularization)
            dw = (1/m) * X.T.dot(predictions - y) + 2 * self.regularization_strength * self.weights
            db = (1/m) * np.sum(predictions - y)
            gradient_norm = np.linalg.norm(dw)
            
            # Check convergence criteria (stricter)
            if cost_change < self.tolerance and gradient_norm < self.tolerance:
                converged = True
                print_color(f"Converged at iteration {iteration + 1}: cost change {cost_change:.10f}, gradient norm {gradient_norm:.10f}", "green")
                break
            
            # Early stopping if no improvement
            if patience_counter >= self.patience:
                print_color(f"Early stopping at iteration {iteration + 1}: no improvement for {self.patience} iterations", "yellow")
                break
            
            # Adaptive learning rate: decay if no improvement for several iterations
            if patience_counter > 0 and patience_counter % 10 == 0:
                self.learning_rate *= self.lr_decay_factor
                print_color(f"Reducing learning rate to {self.learning_rate:.6f}", "yellow")
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print progress periodically
            # if iteration == 0 or (iteration + 1) % max(1, min(50, self.max_iterations // 20)) == 0:
            #     z_mean, z_std = np.mean(z), np.std(z)
            #     weight_norm = np.linalg.norm(self.weights)
                # print_color(f"Iteration {iteration + 1}: Cost: {total_cost:.6f} (change: {cost_change:.8f}), LR: {self.learning_rate:.6f}, Weight norm: {weight_norm:.6f}, Gradient norm: {gradient_norm:.8f}", "cyan")
                # print_color(f"  Logits - mean: {z_mean:.6f}, std: {z_std:.6f}, range: [{np.min(z):.6f}, {np.max(z):.6f}]", "cyan")
                # print_color(f"  Predictions - range: [{np.min(predictions):.6f}, {np.max(predictions):.6f}], mean: {np.mean(predictions):.6f}", "cyan")
                # print_color(f"  Patience: {patience_counter}/{self.patience}", "cyan")
            
            prev_cost = total_cost
        
        # Final status
        if converged:
            print_color(f"Logistic regression converged after {iteration + 1} iterations. Final cost: {total_cost:.6f} (Log-likelihood: {log_likelihood:.6f}, L2 penalty: {l2_penalty:.6f}), bias: {self.bias:.6f}", "green")
        else:
            print_color(f"Logistic regression reached max iterations ({self.max_iterations}). Final cost: {total_cost:.6f} (Log-likelihood: {log_likelihood:.6f}, L2 penalty: {l2_penalty:.6f}), bias: {self.bias:.6f}", "yellow")
        
        # Print timing information
        end_time = time.time()
        elapsed_time = end_time - start_time
        print_color(f"_update_regression_model completed in {elapsed_time:.4f} seconds", "cyan")
    
    def _predict_single(self, entry):
        """Predict a single score for an entry using the logistic regression model. Using the entire buffer as the training data."""
        self._update_regression_model()
            
        embedding = self._get_embedding(entry)
        z = self.weights.dot(embedding) + self.bias
        predicted_score = self._sigmoid(z)
        return predicted_score
    
    def predict_scores_for_batch(self, batch):
        """Predict scores for a batch of candidates and update the buffer with the predicted scores. Using the entire buffer as the training data."""
        self._update_regression_model()
        
        # Get embeddings for all candidates in batch
        embeddings = []
        for entry in batch:
            if "embedding" not in entry:
                entry["embedding"] = self._get_embedding(entry)
            embeddings.append(entry["embedding"])
        
        # Batch prediction using vectorized operations
        X_batch = np.array(embeddings)
        z = X_batch.dot(self.weights) + self.bias
        predicted_scores = self._sigmoid(z)
        
        # Update each candidate with predicted score
        for entry, predicted_score in zip(batch, predicted_scores):
            entry['predicted_score'] = predicted_score
            
        return predicted_scores
    
    def predict_scores(self):
        """Predict scores for all candidates in the buffer. Using the entire buffer as the training data."""
        buffer_list = list(self.buffer)
        batches = [buffer_list[i:i+self.max_candidates_to_predict] for i in range(0, len(buffer_list), self.max_candidates_to_predict)]
        if hasattr(self, 'num_threads') and self.num_threads and self.num_threads > 1:
            # Parallelize batch processing
            batch_functions = [lambda batch=b: self.predict_scores_for_batch(batch) for b in batches]
            async_run(
                batch_functions,
                max_workers=self.num_threads,
                description=f"Processing {len(batches)} candidate batches"
            )
        else:
            # Sequential processing
            for batch in batches:
                self.predict_scores_for_batch(batch)
        # Return the predicted scores for the buffer.
        predicted_scores_for_the_buffer = [candidate['predicted_score'] for candidate in buffer_list]
        return np.array(predicted_scores_for_the_buffer)
