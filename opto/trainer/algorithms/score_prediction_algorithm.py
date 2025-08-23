# score_prediction_algorithm.py

import numpy as np
import copy
import json
import random
from typing import Union, List, Tuple, Dict, Any, Optional
from collections import deque
from opto.utils.llm import LLM
from opto.optimizers.utils import print_color
from opto.trainer.utils import retry_with_exponential_backoff, evaluate_agent
from opto.trainer.algorithms.BAI_algorithms import BAIAlgorithmBase, set_parameters_for_agent
import litellm

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

class ScorePrediction(BAIAlgorithmBase):
    """In the following experiments, we study whether or not LLM have the ability to predict numerical scores. Two versions of LLM implementations:
    1. Predict scores for known candidates.
    2. Predict scores for new candidates.
    In this class, we have a predict_scores method, that could predict scores for all candidates in a buffer, using history information.
    """
    def __init__(self, agent, num_threads, logger, update_dicts,ground_truth_scores, *args, **kwargs):
        if len(ground_truth_scores) != len(update_dicts): # Check if the length of ground_truth_scores and update_dicts is the same.
            raise ValueError("The length of ground_truth_scores and update_dicts must be the same.")
        super().__init__(agent, num_threads, logger, update_dicts, *args, **kwargs)
        self.domain_context = DOMAIN_CONTEXT
        self.llm_model = "gemini/gemini-2.0-flash"
        self.llm = LLM(model=self.llm_model)
        self.ground_truth_scores = np.array(ground_truth_scores)
        self.temperature = 0.0
        self.update_dicts = update_dicts
        self.raw_data = []
        # Initial buffer construction.
        self.buffer = deque(maxlen=100)
        for i, update_dict in enumerate(self.update_dicts):   
            # Evaluate one candidate and update the buffer statistics.
            set_parameters_for_agent(self.agent, update_dict)
            candidate_entry = {"params": update_dict, "score_sum": 0, "eval_count": 0,"mean_score": None, "ground_truth_score": self.ground_truth_scores[i]  }
            self.buffer.append(candidate_entry)
        print_color(f"Constructed the buffer with {len(self.buffer)} candidates.", "green")

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
                # "ucb_score": cand_entry.get('ucb_score', None),
                # "lcb_score": cand_entry.get('lcb_score', None)
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
            print_color(f"Ground truth scores: {self.ground_truth_scores}", "blue")
            
        return predicted_scores_array
    
    def train(self, guide, validate_dataset, test_dataset,num_threads,num_epochs, eval_frequency,temperature=0.0, *args, **kwargs):
        """A general training method. In each epoch collect some data by evaluating agents. Then calculate the prediction error by comparing the predicted scores and ground truth scores."""
        self.validate_dataset = validate_dataset
        self.test_dataset = test_dataset
        self.num_epochs = num_epochs
        self.eval_frequency = eval_frequency
        self.num_threads = num_threads
        self.total_samples = 0
        self.epoch = None
        self.temperature = temperature
        print_color(f"Start training ScorePrediction model.", "green")
        
        length = len(self.ground_truth_scores)
        # First half: arms that will have stats (get evaluated)
        initial_error_for_arm_with_stats = np.mean((self.ground_truth_scores[:length//2])**2)
        # Second half: arms that won't have stats (never evaluated)
        initial_error_for_arm_without_stats = np.mean((self.ground_truth_scores[length//2:])**2)
        self.logger.log("Training loss", np.sqrt(initial_error_for_arm_with_stats), 0, color="red")
        self.logger.log("LLM_error_for_arm_with_stats", np.sqrt(initial_error_for_arm_with_stats), 0, color="red")
        self.logger.log("LLM_error_for_arm_without_stats", np.sqrt(initial_error_for_arm_without_stats), 0, color="red")
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            print_color(f"Epoch {epoch+1} of {self.num_epochs}", "green")
            # Evaluate the agents in the buffer.
            self.collect_data(self.buffer, guide, validate_dataset, num_threads)
            predicted_scores = self.predict_scores(self.buffer, verbose=True,temperature=self.temperature)
            self.calculate_prediction_error(self.buffer,predicted_scores)
            # print_color(f"Prediction error: {error}", "red")
            # self.logger.log("Prediction_error", error, epoch+1, color="red")
            self.logger.log("Total_samples", self.total_samples, epoch+1, color="blue")

            # Log the regret: ground truth score of the arm with the highest predicted score.
            max_predicted_score_index = np.argmax(predicted_scores)
            
            ground_truth_score_of_arm_with_highest_predicted_score = self.ground_truth_scores[max_predicted_score_index]
            highest_ground_truth_score = max(self.ground_truth_scores)
            self.logger.log("Ground_truth_score_of_arm_with_highest_predicted_score", ground_truth_score_of_arm_with_highest_predicted_score, epoch+1, color="blue")
            self.logger.log("Highest_ground_truth_score", highest_ground_truth_score, epoch+1, color="blue")
            self.logger.log("Regret", highest_ground_truth_score - ground_truth_score_of_arm_with_highest_predicted_score, epoch+1, color="red")
        return 
    
    def collect_data(self, buffer, guide, validate_dataset, num_threads):
        """
        Do evaluations and collect the data. Update the buffer statistics.
        By default, we evaluate the agents in the buffer, and update the buffer statistics.
        """
        for candidate_entry in self.buffer:
            set_parameters_for_agent(self.agent, candidate_entry["params"])
            score = evaluate_agent(self.agent, guide, validate_dataset, num_threads=num_threads, num_eval_times=1)
            eval_count = len(validate_dataset['inputs'])
            self.total_samples += eval_count
            candidate_entry["score_sum"] += score*eval_count
            candidate_entry["eval_count"] += eval_count
        self.update_buffer_scores()
    
    def calculate_prediction_error(self, buffer, predicted_scores):
        """
        Calculate the prediction error by comparing the predicted scores and ground truth scores. 
        By default, we calculate the mean squared error, between the empirical mean scores and the predicted scores.
        """
        mean_scores = [c['mean_score'] for c in buffer]
        ground_truth_scores = [c['ground_truth_score'] for c in buffer]
        prediction_error = np.mean((ground_truth_scores - mean_scores)**2)
        self.logger.log("Prediction_error", np.sqrt(prediction_error), self.epoch+1, color="red")
        return prediction_error

class ScorePrediction_half_buffer(ScorePrediction):
    def collect_data(self, buffer, guide, validate_dataset, num_threads):
        """
        Only collect data for the first half of the buffer. At each time, randomly sample one of the candidates in the first half of the buffer. Do a evaluation.
        """
        # Convert deque to list to enable slicing
        buffer_list = list(self.buffer)
        half_size = len(buffer_list) // 2
        # Randomly sample one of the candidates in the first half of the buffer.
        candidate_entry = random.choice(buffer_list[:half_size])
        set_parameters_for_agent(self.agent, candidate_entry["params"])
        score = evaluate_agent(self.agent, guide, validate_dataset, num_threads=num_threads, num_eval_times=2)
        # Collect the raw data for the embedding regression model.
        self.raw_data.append({"embedding": candidate_entry["embedding"], "score": score})

        eval_count = len(validate_dataset['inputs'])*2
        candidate_entry["score_sum"] += score*eval_count
        candidate_entry["eval_count"] += eval_count
        self.total_samples += eval_count
        self.update_buffer_scores()
    
    

    def calculate_prediction_error(self, buffer, predicted_scores):
        """
            Separately calculate the prediction error of the first half and the second half of the buffer.
        """
        length = len(buffer)
        mean_scores = np.array([c['mean_score'] for c in buffer])
        # For the first half of the buffer, calculate the predicted erorr using LLM predictor and empirical mean scores.
        empirical_error = np.mean((self.ground_truth_scores[:length//2] - mean_scores[:length//2])**2)
        llm_error_for_arm_with_stats = np.mean((self.ground_truth_scores[:length//2] - predicted_scores[:length//2])**2)
        # For the second half of the buffer, calculate the predicted erorr between LLM predicted scores and ground truth scores.
        llm_error_for_arm_without_stats = np.mean((self.ground_truth_scores[length//2:] - predicted_scores[length//2:])**2)
        self.logger.log("Training loss", np.sqrt(empirical_error), self.epoch+1, color="red")
        self.logger.log("LLM_error_for_arm_with_stats", np.sqrt(llm_error_for_arm_with_stats), self.epoch+1, color="red")
        self.logger.log("LLM_error_for_arm_without_stats", np.sqrt(llm_error_for_arm_without_stats), self.epoch+1, color="red")
        return 

class Embedding_Regression(ScorePrediction_half_buffer):
    """
    Instead of using LLM response to predict the scores directly, we train a linear regression model from the embedding to the ground truth scores.
    """
    def __init__(self, agent, num_threads, logger, update_dicts, ground_truth_scores, 
                 embedding_model="gemini/text-embedding-004", learning_rate=0.01, *args, **kwargs):
        super().__init__(agent, num_threads, logger, update_dicts, ground_truth_scores, *args, **kwargs)
        
        # Set embedding model and learning rate from parameters
        self.embedding_model = embedding_model
        self.learning_rate = learning_rate
        self.linear_dim = None
        print_color(f"Computing embeddings for {len(self.buffer)} buffer entries...", "yellow")
        
        for i, agent_entry in enumerate(self.buffer):
            additional_instructions = list(agent_entry["params"].values())[0]
            
            # Use litellm directly for embedding
            response = litellm.embedding(
                model=self.embedding_model,
                input=additional_instructions
            )
            embedding = response.data[0].embedding
            
            agent_entry["embedding"] = embedding
            if self.linear_dim is None:
                self.linear_dim = len(embedding)
                print_color(f"  Embedding dimension: {self.linear_dim}", "cyan")
        
        print_color(f"Embeddings computed successfully!", "green")
        
        # Initialize a linear regression model with all zeros weights
        # Dimension is self.linear_dim. Also add initial bias to be zero.
        self.weights = np.zeros(self.linear_dim)
        self.bias = 0.0
        
        # Track how many data points we've processed for SGD
        self.processed_data_count = 0
        
        print_color(f"Linear regression model initialized:", "cyan")
        print_color(f"  Weights: zeros({self.linear_dim})", "cyan")
        print_color(f"  Bias: 0.0", "cyan")
        
        # Initialize the regression model with current buffer data
        self._update_regression_model()
    
    def _update_regression_model(self):
        """
        Update the embedding linear regression model using SGD.
        Only processes new data points since last update (incremental learning).
        """
        if len(self.raw_data) == 0:
            return
        
        # Check if we have new data to process
        if len(self.raw_data) <= self.processed_data_count:
            return  # No new data
        
        # Process only the new data points (typically just the last one)
        new_data_points = self.raw_data[self.processed_data_count:]
        
        for data_entry in new_data_points:
            embedding = np.array(data_entry["embedding"])
            true_score = data_entry["score"]
            
            # Forward pass: predict score with current weights
            predicted_score = np.dot(self.weights, embedding) + self.bias
            
            # Compute error
            error = predicted_score - true_score
            
            # SGD update: gradients for linear regression
            # Loss = 0.5 * (predicted - true)^2
            # dL/dw = error * embedding
            # dL/db = error
            
            # Update weights and bias
            self.weights -= self.learning_rate * error * embedding
            self.bias -= self.learning_rate * error
        
        # Update the count of processed data points
        self.processed_data_count = len(self.raw_data)
    
    def _predict_single(self, embedding):
        """
        Predict score for a single embedding using linear regression.
        Score = weights^T * embedding + bias
        """
        return np.dot(self.weights, embedding) + self.bias
    
    def predict_scores(self, buffer, verbose: bool = False, temperature: float = 0.0):
        """
        Predict the scores for the buffer using the linear regression model.
        """
        # Update the embedding linear regression model with current buffer statistics
        self._update_regression_model()
        
        # Predict the scores for each candidate in the buffer
        predicted_scores = []
        for agent_entry in buffer:
            embedding = agent_entry["embedding"]
            predicted_score = self._predict_single(embedding)
            predicted_scores.append(predicted_score)
            
            if verbose:
                print(f"Embedding shape: {len(embedding)}, Predicted score: {predicted_score:.4f}")
        
        predicted_scores_array = np.array(predicted_scores)
        
        mean_scores = [c['mean_score'] for c in buffer]
        if verbose:
            print_color(f"SGD training data points: {len(self.raw_data)} (processed: {self.processed_data_count})", "cyan")
            print_color(f"SGD learning rate: {self.learning_rate}", "cyan")
            print_color(f"Regression weights norm: {np.linalg.norm(self.weights):.4f}", "cyan")
            print_color(f"Regression bias: {self.bias:.4f}", "cyan")
            print_color(f"Predicted scores: {predicted_scores_array}", "green")
            print_color(f"Mean scores (fallback): {mean_scores}", "yellow")
            print_color(f"Ground truth scores: {self.ground_truth_scores}", "blue")
            
            # Show training data statistics if available
            if len(self.raw_data) > 0:
                training_scores = [d["score"] for d in self.raw_data]
                print_color(f"Training scores range: [{min(training_scores):.4f}, {max(training_scores):.4f}]", "cyan")
                
                # Show prediction error on training data (last few points)
                if len(self.raw_data) > 0:
                    last_entry = self.raw_data[-1]
                    last_embedding = np.array(last_entry["embedding"])
                    last_true_score = last_entry["score"]
                    last_predicted = np.dot(self.weights, last_embedding) + self.bias
                    print_color(f"Last training point - True: {last_true_score:.4f}, Predicted: {last_predicted:.4f}, Error: {abs(last_predicted - last_true_score):.4f}", "cyan")
        
        return predicted_scores_array
    
class Embedding_Regression_with_true_scores(Embedding_Regression):
    """
    Use the embedding, ground truth scores to do a linear regression. Calculate the misspecification error.
    I can just modify the _update_regression_model function.
    """
    def _update_regression_model(self):
        """
        Not use SGD. Use the embeddings with true scores to do a linear regression.
        Only use the first half of the buffer for training.
        """
        if len(self.buffer) == 0:
            return
            
        # Only use the first half of the buffer for regression training
        buffer_list = list(self.buffer)
        half_size = len(buffer_list) // 2
        
        if half_size == 0:
            print_color("Warning: Buffer too small, using all data", "yellow")
            training_buffer = buffer_list
        else:
            training_buffer = buffer_list[:half_size]
            
        embeddings = np.array([c["embedding"] for c in training_buffer])
        true_scores = np.array([c["ground_truth_score"] for c in training_buffer])
        
        # Add bias column to embeddings for least squares
        X_with_bias = np.column_stack([embeddings, np.ones(embeddings.shape[0])])
        
        # Do a linear regression using least squares
        try:
            coefficients, residuals, rank, s = np.linalg.lstsq(X_with_bias, true_scores, rcond=None)
            self.weights = coefficients[:-1]  # All but last coefficient
            self.bias = coefficients[-1]     # Last coefficient is bias
            
            print_color(f"Linear regression with true scores completed:", "green")
            print_color(f"  Training samples: {len(training_buffer)} (first half of {len(buffer_list)} total)", "green")
            print_color(f"  Weights norm: {np.linalg.norm(self.weights):.4f}", "green")
            print_color(f"  Bias: {self.bias:.4f}", "green")
            if len(residuals) > 0:
                print_color(f"  Residual sum of squares: {residuals[0]:.6f}", "green")
                
        except np.linalg.LinAlgError:
            print_color("Warning: Singular matrix in least squares, using pseudo-inverse", "yellow")
            coefficients = np.linalg.pinv(X_with_bias) @ true_scores
            self.weights = coefficients[:-1]
            self.bias = coefficients[-1] 


