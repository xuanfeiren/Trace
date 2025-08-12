import numpy as np
import copy
from collections import deque
from typing import Union, List, Tuple, Dict, Any, Optional
from opto import trace
from opto.trainer.utils import async_run, batch_run # Assuming print_color is in utils
from opto.optimizers.utils import print_color
from opto.trainer.algorithms.basic_algorithms import batchify # evaluate and batchify might be useful
from opto.trainer.algorithms.UCBsearch import UCBSearchAlgorithm
from opto.utils.llm import LLM # For the selector LLM
from opto.trace.nodes import ParameterNode
import json
import warnings
from black import format_str, FileMode
from opto.trainer.evaluators import evaluate
import time

def auto_retry_with_exponential_backoff(
    func, 
    max_retries=5, 
    base_delay=1.0, 
    operation_name="operation"
):
    """
    Auto retry a function with exponential backoff for transient errors.
    
    Args:
        func: Function to retry (should be a callable with no arguments)
        max_retries: Maximum number of retry attempts
        base_delay: Base delay for exponential backoff
        operation_name: Name of the operation for logging
    
    Returns:
        Result of the function call, or None if all retries failed
    """
    for retry_attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            error_str = str(e).lower()
            error_type = type(e).__name__.lower()
            
            # Check if it's a retryable error
            retryable_errors = [
                'rate limit', 'timeout', 'temporary', 'service unavailable',
                'internal server error', 'bad gateway', 'service temporarily unavailable',
                'too many requests', 'quota', 'overloaded', 'resource has been exhausted',
                'resource_exhausted', 'ratelimiterror', 'quotaexceedederror',
                'connection error', 'network', 'json decode'
            ]
            
            # Also check specific exception types that might be retryable
            retryable_exception_types = [
                'ratelimiterror', 'timeouterror', 'apiconnectionerror', 
                'serviceunavailableerror', 'internalservererror', 'jsondecodeerror'
            ]
            
            is_retryable = (
                any(err in error_str for err in retryable_errors) or
                any(exc_type in error_type for exc_type in retryable_exception_types) or
                'code": 429' in error_str or  # HTTP 429 Too Many Requests
                'code": 503' in error_str or  # HTTP 503 Service Unavailable
                'code": 502' in error_str or  # HTTP 502 Bad Gateway
                'code": 500' in error_str     # HTTP 500 Internal Server Error
            )
            
            if retry_attempt == max_retries - 1:
                # Last attempt failed
                print(f"{operation_name}: Failed after {max_retries} attempts. Error: {e}")
                return None
            elif is_retryable:
                # Special handling for rate limit errors - use longer delays
                is_rate_limit = (
                    'rate limit' in error_str or 'ratelimiterror' in error_type or
                    'quota' in error_str or 'resource has been exhausted' in error_str or
                    'code": 429' in error_str
                )
                
                if is_rate_limit:
                    # Longer delays for rate limits
                    delay = 2 * (retry_attempt + 1) ** 2 + retry_attempt
                else:
                    # Standard exponential backoff for other errors
                    delay = base_delay * (2 ** retry_attempt) + (0.1 * retry_attempt)
                
                error_type_desc = "Rate limit" if is_rate_limit else "Retryable error"
                # print(f"{operation_name}: {error_type_desc} - Retry {retry_attempt + 1}/{max_retries} after {delay:.1f}s. Error: {e}")
                time.sleep(delay)
            else:
                # Non-retryable error
                print(f"{operation_name}: Non-retryable error: {e}")
                return None
    
    return None

class ExploreAlgorithm(UCBSearchAlgorithm):    
    """A phased algorithm that explores the parameter space of the agent, and use UCB to select the best candidate."""
    
    def __init__(self,
                 agent: trace.Module,
                 optimizer,
                 max_buffer_size: int = 1000,
                 ucb_exploration_factor: float = 1.0,
                 logger=None,
                 num_threads: int = None,
                 *args,
                 **kwargs): 
        super().__init__(agent, optimizer, max_buffer_size, ucb_exploration_factor, logger, num_threads, *args, **kwargs)
        
        self.buffer = deque() 
        self.max_buffer_size = max_buffer_size
        self.ucb_exploration_factor = ucb_exploration_factor
        self.logger = logger
        self.num_threads = num_threads
        self.num_eval_times = 5 # number of times to evaluate each candidate

    def print_buffer_statistics(self):
        self._update_buffer_scores()
        for i,candidate_entry in enumerate(self.buffer):
            print_color(f"Candidate {i}. Mean score {candidate_entry['mean_score']}, eval_count {candidate_entry['eval_count']}", "blue")
        return 
    def _evaluate_candidate(self, 
                              params_to_eval_dict: Dict[str, Any], 
                              dataset: Dict[str, List[Any]], 
                              guide, 
                              evaluation_batch_size: int = 10,
                              num_threads: Optional[int] = None,
                              num_eval_times: int = 1
                              ) -> Tuple[float, int]:
        """Evaluates a given set of parameters on samples from the provided dataset."""
        if not dataset or not dataset.get('inputs') or not dataset.get('infos') or not dataset['inputs']:
            print_color("Evaluation dataset is empty or invalid. Returning score -inf, count 0.", color='yellow')
            return -np.inf, 0

        original_params = {p: copy.deepcopy(p.data) for p in self.optimizer.parameters}
        self.optimizer.update(params_to_eval_dict)      

        # Sample a subset of the dataset instead of using the entire dataset
        eval_xs, eval_infos = self._sample_minibatch(dataset, evaluation_batch_size)
        
        if not eval_xs:
            print_color("Evaluation minibatch is empty. Returning score -inf, count 0.", color='yellow')
            self.optimizer.update(original_params) 
            return -np.inf, 0

        eval_scores = evaluate(self.agent,
                               guide, 
                               eval_xs,
                               eval_infos,
                               min_score=self.min_score if hasattr(self, 'min_score') else None,
                               num_threads=num_threads or self.num_threads,
                               num_samples=num_eval_times,
                               description=f"Evaluating candidate")

        self.optimizer.update(original_params) 
        # Extract all non-None values and compute overall average
        # Handle both 1D and 2D eval_scores
        if eval_scores.ndim == 1:
            all_valid_scores = [score for score in eval_scores if score is not None]
        else:
            all_valid_scores = [score for row in eval_scores for score in row if score is not None]

        avg_score = np.mean(all_valid_scores) if all_valid_scores else 0
        
        eval_count = len(all_valid_scores) 
        
        return float(avg_score), eval_count

    def explore(self,
                guide,
                train_dataset: Dict[str, List[Any]],
                *,
                train_batch_size: int = 1,
                num_to_sample: int = 5,
                num_threads: Optional[int] = None,
                **kwargs 
                ) -> Tuple[Dict[str, Any], int]:
        """Explore the parameter space of the agent based on the current buffer, make the buffer bigger"""
        
        if not self.buffer:
            print_color("Buffer is empty, cannot explore.", 'red')
            return {}, 0
        print_color(f"Exploring with OptoPrime.", 'cyan')
        # Sample candidates from buffer using exponential weights of mean scores
        buffer_list = list(self.buffer)
        # Filter candidates that have been evaluated (eval_count > 0)
        evaluated_candidates = [c for c in buffer_list if c['eval_count'] > 0]
        
        if not evaluated_candidates:
            # Fallback to random sampling if no candidates have been evaluated
            print_color("No evaluated candidates found, falling back to random sampling.", 'yellow')
            sampled_candidates = np.random.choice(buffer_list, size=num_to_sample, replace=True)
        else:
            # Calculate mean scores for evaluated candidates
            mean_scores = np.array([c['score_sum'] / c['eval_count'] for c in evaluated_candidates])
            
            # Apply temperature scaling (lower temperature = more greedy towards higher scores)
            temperature = 0.00001  # Low temperature for more focused sampling on high-scoring candidates
            scaled_scores = mean_scores / temperature
            
            # Calculate exponential weights (using softmax to avoid overflow)
            exp_weights = np.exp(scaled_scores - np.max(scaled_scores))  # Subtract max for numerical stability
            weights = exp_weights / np.sum(exp_weights)
            
            # Sample according to exponential weights
            indices = np.random.choice(len(evaluated_candidates), size=num_to_sample, replace=True, p=weights)
            sampled_candidates = [evaluated_candidates[i] for i in indices]
            
        for candidate in sampled_candidates:
            try:
                # Load candidate parameters
                self.optimizer.update(candidate['params'])

                # Sample training batch
                train_xs, train_infos = self._sample_minibatch(train_dataset, train_batch_size)
                if not train_xs:
                    continue
                
                # Forward pass
                forward = batch_run(max_workers=num_threads, description=f"Explore: Forward pass (batch size: {len(train_xs)})")(self.forward)
                outputs = forward(self.agent, train_xs, guide, train_infos)
                
                # Process outputs
                scores, targets, feedbacks = [], [], []
                for target, score, feedback in outputs:
                    scores.append(score)
                    targets.append(target)
                    feedbacks.append(feedback)
                
                if not scores:
                    continue
                    
                # Backward pass
                target_batch = batchify(*targets)
                feedback_batch = batchify(*feedbacks).data
                
                self.optimizer.zero_feedback()
                self.optimizer.backward(target_batch, feedback_batch)
                
                # Generate new candidate with retry logic
                def optimizer_step_call():
                    update_dict = self.optimizer.step(bypassing=True, verbose=False)
                    return update_dict
                
                new_params_dict = auto_retry_with_exponential_backoff(
                    optimizer_step_call,
                    max_retries=10,
                    operation_name="Optimizer step"
                )
                
                if not isinstance(new_params_dict, dict) or not new_params_dict:
                    new_params_dict = {p: copy.deepcopy(p.data) for p in self.optimizer.parameters}

                # Ensure new_params_dict contains all parameters from optimizer
                for param in self.optimizer.parameters:
                    if param not in new_params_dict:
                        new_params_dict[param] = copy.deepcopy(param.data)
                self.total_samples += train_batch_size
                self.total_proposals += 1
                # Initial validation for the new proposal
                validation_score, validation_evals = self._evaluate_candidate(
                    new_params_dict, 
                    self.validation_dataset, 
                    guide, 
                    len(self.validation_dataset['inputs']),  # Now using subset instead of entire dataset
                    num_threads
                )
                self.total_samples += validation_evals
                # Add new candidate to buffer
                new_candidate_entry = {
                    'params': new_params_dict,
                    'score_sum': validation_score*validation_evals,
                    'eval_count': validation_evals,
                    'ucb_score': None
                }
                
                self.buffer.append(new_candidate_entry)
                       
            except Exception as e:
                print_color(f"Explore: Error processing candidate: {e}", 'red')
                continue
        return 
    
    def _update_buffer_scores(self):
        """Recalculates and updates UCB scores for all candidates in the buffer."""
        if not self.buffer:
            return
        total_evaluations_tracker = np.sum([c['eval_count'] for c in self.buffer])
        for candidate_entry in self.buffer:
            candidate_entry['ucb_score'] = self._calculate_ucb(candidate_entry, total_evaluations_tracker)
            candidate_entry['lcb_score'] = self._calculate_lcb(candidate_entry, total_evaluations_tracker)
            candidate_entry['mean_score'] = candidate_entry['score_sum'] / (candidate_entry['eval_count'] or 1E-9)

    def ucb_best_candidate(self, 
                      horizon: int, 
                      validation_dataset: Dict[str, List[Any]], 
                      guide, 
                      evaluation_batch_size: int = 20,
                      num_threads: Optional[int] = None) -> Dict[str, Any]:
        """Select the best candidate from the buffer using UCB for horizon iterations."""
        
        if not self.buffer:
            print_color("Buffer is empty, cannot select best candidate.", 'red')
            return None
        
        print_color(f"Best candidate identification: Starting {horizon} iterations", 'blue')
        
        # UCB-based best arm identification
        for iteration in range(horizon):
            print_color(f"BAI Iteration: {iteration+1}/{horizon}: ", 'cyan')
            
            # Update UCB scores
            self._update_buffer_scores()
            # self.print_buffer_statistics()
            # Select candidate with highest UCB score
            selected_candidate = self.select_candidate(self.buffer)
            try:
            # Evaluate on validation set subset
                validation_score, validation_evals = self._evaluate_candidate(
                    selected_candidate['params'], 
                    validation_dataset, 
                    guide, 
                    evaluation_batch_size,  # Now using subset instead of entire dataset
                    num_threads
                )
            except Exception as e:
                print_color(f"Best candidate identification: Error evaluating candidate: {e}", 'red')
                continue
            
            # Update candidate statistics
            if validation_score is not None and validation_score > -np.inf and validation_evals > 0:
                selected_candidate['score_sum'] += validation_score * validation_evals
                selected_candidate['eval_count'] += validation_evals
                self.total_samples += validation_evals
                print_color(f"UCB iteration {iteration+1}/{horizon}: "
                          f"Selected candidate score {validation_score:.4f} "
                          f"(evaluated on {validation_evals} samples)", 'cyan')
                
        self._update_buffer_scores()
        # Remove candidates with zero score
        self.buffer = [c for c in self.buffer if c['score_sum'] > 0]
        # Return the candidate with highest score (pure exploitation)
        best_candidate = max(self.buffer, key=lambda c: c['mean_score'])
        print_color("Original buffer after UCB best arm identification", 'blue')
        self.print_intervals(self.buffer)
        # Handle buffer overflow - keep only max_buffer_size best candidates based on lcb score
        if len(self.buffer) > self.max_buffer_size:
            # Sort by lcb score and keep only the top max_buffer_size candidates
            sorted_buffer = sorted(self.buffer, key=lambda c: c['mean_score'], reverse=True)
            self.buffer = deque(sorted_buffer[:self.max_buffer_size])
            print_color(f"Buffer size reduced from {len(sorted_buffer)} to {len(self.buffer)} based on ucb score", 'yellow')

        return best_candidate

    def train(self,
              guide,
              train_dataset: Dict[str, List[Any]],
              *,
              validation_dataset: Optional[Dict[str, List[Any]]] = None,
              test_dataset: Optional[Dict[str, List[Any]]] = None,
              train_batch_size: int = 1,
              evaluation_batch_size: int = 20,
              eval_frequency: int = 1,  
              log_frequency: Optional[int] = None,
              min_score_for_agent_update: Optional[float] = 0,
              num_to_sample: int = 5,
              num_LLM_samples: int = 2,
              num_threads: Optional[int] = None,
              num_phases: int = 5,
            #   ucb_horizon: int = 50,
              **kwargs
              ) -> Tuple[Dict[str, Any], float]:
        """Train using explore and best_candidate phases iteratively."""
        
        # Default datasets
        if validation_dataset is None:
            validation_dataset = train_dataset
        if test_dataset is None:
            test_dataset = train_dataset
        self.validate_dataset = validation_dataset
        self.validation_dataset = validation_dataset
        num_threads = num_threads or self.num_threads
        log_frequency = log_frequency or eval_frequency
        self.min_score = min_score_for_agent_update
        self.validation_dataset = validation_dataset # For use in _evaluate_candidate
        # Initialize tracking
        self.total_samples = 0
        self.total_proposals = 0
        # Two budget constraints: ucb_horizon and max_buffer_size
        self.horizon = ucb_horizon = self.max_buffer_size
        self.num_epochs = num_phases

        # Initialize buffer with initial candidate, do the initial test.
        initial_params_dict = {p: copy.deepcopy(p.data) for p in self.optimizer.parameters}
        test_score, test_evals = self._evaluate_candidate(
            initial_params_dict, test_dataset, guide, len(test_dataset['inputs']), num_threads,num_eval_times=self.num_eval_times
        )
        # Add initial validation for the initial candidate
        validation_score, validation_evals = self._evaluate_candidate(
                    initial_params_dict, 
                    validation_dataset, 
                    guide, 
                    len(validation_dataset['inputs']),  # Now using subset instead of entire dataset
                    num_threads
                )
        self.total_samples += validation_evals
        initial_candidate_entry = {
            'params': initial_params_dict,
            'score_sum': validation_score*validation_evals,
            'eval_count': validation_evals,
            'ucb_score': None,
        }
        self.buffer.append(initial_candidate_entry)

        self.logger.log('Buffer size', len(self.buffer), 0, color='yellow')
        self.logger.log('Test score', test_score, 0, color='green')
        self.logger.log('Total samples', self.total_samples, 0, color='cyan')
        self.logger.log('Total proposals', self.total_proposals, 0, color='red')        
        
        # Main training loop
        for phase in range(num_phases):
            print_color(f"\n=== Optimization Phase {phase+1}/{num_phases} ===", 'blue')
            self._current_iteration = phase
            
            # print_color(f"\n=== Phase {phase+1}/{num_phases} ===", 'blue')
            
            # Explore phase
            print_color("Starting exploration phase...", 'cyan')
            self.explore(
                guide, 
                train_dataset, 
                train_batch_size=train_batch_size,
                num_to_sample=num_to_sample,
                num_LLM_samples=num_LLM_samples,
                num_threads=num_threads
            )
            
            # Best candidate identification phase
            print_color("Starting best candidate identification phase...", 'cyan')

            # after ucb best candidate identification, return the best candidate based on lcb score
            best_candidate = self.ucb_best_candidate(
                guide=guide,
                horizon=ucb_horizon,
                validation_dataset=validation_dataset,
                evaluation_batch_size=len(validation_dataset['inputs']),
                num_threads=num_threads
            )
            best_params = best_candidate['params']
            if best_params is None:
                print_color(f"Phase {phase+1}: No best candidate found, skipping test.", 'red')
                continue
                
            # Load best candidate parameters
            self.optimizer.update(best_params)
            print_color("Buffer after discarding", 'blue')
            self.print_intervals(self.buffer)
            # total_evaluations_tracker = np.sum([c['eval_count'] for c in self.buffer])
            selected_mean_score = best_candidate['score_sum'] / (best_candidate['eval_count'] or 1E-9)
            ucb = best_candidate['ucb_score']
            lcb = best_candidate['lcb_score']
            
            # Test evaluation
            try:
                # test_score, test_evals = self._evaluate_candidate(
                #     best_params,
                #     test_dataset,
                #     guide,
                #     len(test_dataset['inputs']),  # Use subset for test evaluation too
                #     num_threads,
                #     num_eval_times=self.num_eval_times
                # )
                # At the test step, we also want to log the raw test results
                self.optimizer.update(best_params)                      
                eval_scores = evaluate(self.agent,
                                        guide, 
                                        test_dataset['inputs'],
                                        test_dataset['infos'],
                                        min_score=self.min_score,
                                        num_threads=num_threads,
                                        num_samples=self.num_eval_times,
                                        description=f"Evaluating candidate")
                 # Create table with explicit column names
                columns = [f'Eval_{i+1}' for i in range(eval_scores.shape[1])]
                table = self.logger.wandb.Table(columns=columns, data=eval_scores.tolist())
                self.logger.log(f'Raw_test_scores_at_step_{phase+1}', table, phase+1, color='green')
                # Extract all non-None values and compute overall average
                all_valid_scores = [score for row in eval_scores for score in row if score is not None]
                test_score = np.mean(all_valid_scores) if all_valid_scores else 0
                 # test_evals = len(all_valid_scores)
                # Calculate buffer statistics
                buffer_mean_scores = []
                for candidate in self.buffer:
                    if candidate['eval_count'] > 0 and candidate['score_sum'] is not None:
                        mean_score = candidate['score_sum'] / candidate['eval_count']
                        if mean_score is not None:
                            buffer_mean_scores.append(mean_score)
                
                # Log buffer statistics
                if buffer_mean_scores:
                    highest_score = max(buffer_mean_scores)
                    lowest_score = min(buffer_mean_scores)
                    buffer_mean_score = sum(buffer_mean_scores) / len(buffer_mean_scores)
                    
                    self.logger.log('Buffer highest score', highest_score, phase+1, color='magenta')
                    self.logger.log('Buffer lowest score', lowest_score, phase+1, color='magenta')
                    self.logger.log('Buffer mean score', buffer_mean_score, phase+1, color='magenta')
                    
                # Logging
                self.logger.log('Buffer size', len(self.buffer), phase+1, color='yellow')
                self.logger.log('Test score', test_score, phase+1, color='green')
                self.logger.log('Selected mean score', selected_mean_score, phase+1, color='magenta')
                self.logger.log('UCB', ucb, phase+1, color='magenta')
                self.logger.log('LCB', lcb, phase+1, color='magenta')
                self.logger.log('Total samples', self.total_samples, phase+1, color='cyan')
                self.logger.log('Total proposals', self.total_proposals, phase+1, color='red')
                
            except Exception as e:
                print_color(f"Phase {phase+1}: Test evaluation failed: {e}", 'red')
        params_values = list(best_params.values())
        self.logger.log('Final parameter 1', params_values[0], phase+1, color='magenta')
        self.logger.log('Final parameter 2', params_values[1], phase+1, color='magenta')
               
        # Final results
        print_color("ExploreAlgorithm training completed.", 'blue')
        
        return 
    
    def select_candidate(self, buffer):
        """Select the candidate with the highest UCB score."""
        return max(buffer, key=lambda c: c['ucb_score'])
    
class ExplorewithLLM(ExploreAlgorithm):
    """Explore with LLM, use the LLM to generate new candidates."""
    
    def __init__(self,
                 agent: trace.Module,
                 optimizer,
                 max_buffer_size: int = 1000,
                 ucb_exploration_factor: float = 1.0,
                 logger=None,
                 num_threads: int = None,
                 llm_model: str = "gemini/gemini-2.0-flash",
                 num_samples_in_prompt: int = 5,
                 *args,
                 **kwargs):
        super().__init__(agent, optimizer, max_buffer_size, ucb_exploration_factor, logger, num_threads, *args, **kwargs)
        
        # Initialize LLM
        self.llm_model = llm_model
        self.num_samples_in_prompt = num_samples_in_prompt
        self.llm = LLM(model=llm_model)
        
        print_color(f"Initialized ExplorewithLLM with LLM model: {llm_model}", "cyan")
    
    def _llm_generate_candidate(self, num_LLM_samples: int = 1) -> List[Dict[trace.nodes.ParameterNode, str]]:
        """
        Prompts an LLM with current buffer candidates to generate new string values for parameters.
        Retries until num_LLM_samples successful candidates are generated.
        Returns a list of dictionaries mapping ParameterNode objects to new string values.
        """
        # print_color("Attempting to generate candidate using LLM...", "blue")
        if not self.buffer:
            print_color("LLM generation: Buffer is empty, cannot provide context to LLM.", "yellow")
            return []

        # Filter buffer to only include candidates with valid UCB scores
        valid_candidates = [c for c in self.buffer if c.get('ucb_score') is not None and c.get('ucb_score') != -float('inf') and c.get('ucb_score') != float('inf')]
        
        if not valid_candidates:
            print_color("LLM generation: No candidates with valid UCB scores found.", "yellow")
            return []
        
        sorted_buffer = sorted(valid_candidates, key=lambda c: c.get('ucb_score', -float('inf')), reverse=True)
        # Include first, last, and evenly spaced middle candidates
        if len(sorted_buffer) <= self.num_samples_in_prompt:
            prompt_candidates = sorted_buffer
        elif self.num_samples_in_prompt <= 2:
            # If only 1-2 samples requested, take first and optionally last
            prompt_candidates = sorted_buffer[:self.num_samples_in_prompt]
        else:
            # Take first, last, and evenly spaced middle candidates
            prompt_candidates = [sorted_buffer[0]]  # First (highest UCB)
            if self.num_samples_in_prompt > 2:
                # Calculate indices for middle candidates
                middle_count = self.num_samples_in_prompt - 2  # Exclude first and last
                if middle_count > 0 and len(sorted_buffer) > 2:
                    # Evenly space middle candidates between index 1 and len-2
                    middle_indices = [int(1 + i * (len(sorted_buffer) - 2) / (middle_count + 1)) 
                                    for i in range(1, middle_count + 1)]
                    prompt_candidates.extend([sorted_buffer[i] for i in middle_indices])
            prompt_candidates.append(sorted_buffer[-1])  # Last (lowest UCB)
        
        serializable_candidate_summaries = []
        for cand_entry in prompt_candidates:
            summary = {
                "parameters":  {getattr(p,'py_name'): copy.deepcopy(p.data) for p in cand_entry['params']},
                "eval_count": cand_entry['eval_count'],
                "ucb_score": round(cand_entry.get('ucb_score',0), 4),
            }
            serializable_candidate_summaries.append(summary)
        
        example_param_structure_json_str = {getattr(p,'py_name'): copy.deepcopy(p.data) for p in self.agent.parameters()}

        prompt_messages = [
            {"role": "system", "content": "You are an expert in model optimization. Your task is to propose new string values for model parameters with high UCB scores. Please output ONLY a valid JSON dictionary where keys are parameter names and values are the new string values for those parameters, matching the example structure provided. Do not add any explanations or markdown formatting around the JSON."},
            {"role": "user", "content": f"Here are some current candidates from the search buffer and their statistics:\\n{serializable_candidate_summaries}\\n\\nHere is an example of the required JSON output structure (parameter names as keys, new string values as values):\\n{example_param_structure_json_str}\\n\\nPlease generate a new set of parameters in exactly the same JSON format. Make sure use double quotes for the keys and values."}
        ]
        
        response_format = {"type": "json_object"}
        
        successful_candidates = []
        max_total_retries = num_LLM_samples * 5  # Allow up to 5 retries per desired sample
        retry_count = 0
        
        while len(successful_candidates) < num_LLM_samples and retry_count < max_total_retries:
            try:
                retry_count += 1
                # print_color(f"LLM generation attempt {retry_count}/{max_total_retries}, successful candidates: {len(successful_candidates)}/{num_LLM_samples}", "blue")
                
                # Use auto_retry_with_exponential_backoff for LLM calls
                def llm_call():
                    return self.llm(prompt_messages, response_format=response_format)
                
                llm_response = auto_retry_with_exponential_backoff(
                    llm_call,
                    max_retries=5,
                    base_delay=1.0,
                    operation_name=f"LLM generation (attempt {retry_count}/{max_total_retries})"
                )
                
                if llm_response is None:
                    print_color("LLM call failed after retries, continuing to next attempt...", "yellow")
                    continue
                
                llm_response_str = llm_response.choices[0].message.content

                if not llm_response_str:
                    print_color("LLM returned an empty response, retrying...", "yellow")
                    continue
                
                cleaned_llm_response_str = llm_response_str.strip()

                try:
                    llm_params_raw = json.loads(cleaned_llm_response_str)
                except json.JSONDecodeError as e:
                    # print_color(f"JSON parsing failed: {e}, retrying...", "yellow")
                    continue

                if not isinstance(llm_params_raw, dict):
                    # print_color(f"LLM output was not a JSON dictionary: {type(llm_params_raw)}, retrying...", "yellow")
                    continue
                
                try:
                    candidate_params_dict = self.construct_update_dict(llm_params_raw)
                    successful_candidates.append(candidate_params_dict)
                    print_color(f"Successfully generated candidate {len(successful_candidates)}/{num_LLM_samples}", "green")
                except Exception as e:
                    print_color(f"Error constructing update dict: {e}, retrying...", "yellow")
                    continue
                    
            except Exception as e:
                print_color(f"LLM generation error: {e}, retrying...", "yellow")
                continue
        
        if len(successful_candidates) < num_LLM_samples:
            print_color(f"Warning: Only generated {len(successful_candidates)} candidates out of {num_LLM_samples} requested after {retry_count} attempts", "yellow")
        
        return successful_candidates
           
    
    def construct_update_dict(self, suggestion: Dict[str, Any]) -> Dict[ParameterNode, Any]:
        """Convert the suggestion in text into the right data type."""
        update_dict = {}
        for node in self.agent.parameters():
            if node.trainable and node.py_name in suggestion:
                try:
                    formatted_suggestion = suggestion[node.py_name]
                    if type(formatted_suggestion) == str and 'def' in formatted_suggestion:
                        formatted_suggestion = format_str(formatted_suggestion, mode=FileMode())
                    update_dict[node] = type(node.data)(formatted_suggestion)
                except (ValueError, KeyError) as e:
                    if getattr(self, 'ignore_extraction_error', False):
                        warnings.warn(
                            f"Cannot convert the suggestion '{suggestion[node.py_name]}' for {node.py_name} to the right data type"
                        )
                    else:
                        raise e
        return update_dict

    def explore(self,
                guide,
                train_dataset: Dict[str, List[Any]],
                *,
                train_batch_size: int = 1,
                num_to_sample: int = 5,
                num_LLM_samples: int = 2,
                num_threads: Optional[int] = None,
                **kwargs 
                ) -> Tuple[Dict[str, Any], int]:
        super().explore(
                guide,
                train_dataset,
                train_batch_size=train_batch_size,
                num_to_sample=num_to_sample,
                num_threads=num_threads,
                **kwargs 
                )
        
        # Add LLM-based exploration
        self._update_buffer_scores()

        valid_candidates = [c for c in self.buffer if c.get('ucb_score') is not None and c.get('ucb_score') != -float('inf') and c.get('ucb_score') != float('inf')]
        
        if not valid_candidates:
            print_color("LLM generation: No candidates with valid UCB scores found.", "yellow")
            return None
        print_color(f"Adding {num_LLM_samples} LLM-generated candidates...", 'magenta')
        
        # Generate LLM candidates - now returns a list directly
        new_candidates = self._llm_generate_candidate(num_LLM_samples=num_LLM_samples)
        
        # Process the generated candidates
        llm_candidates_added = 0
        for new_candidate in new_candidates:
            if new_candidate is not None and isinstance(new_candidate, dict):  # Only add if candidate is valid
                new_candidate_entry = {
                    'params': new_candidate,
                    'score_sum': 0,
                    'eval_count': 0,
                    'ucb_score': None,
                }
                self.buffer.append(new_candidate_entry)
                llm_candidates_added += 1
            else:
                print_color("LLM generated invalid candidate, skipping...", 'yellow')
        
        # Update total_proposals counter
        self.total_proposals += llm_candidates_added
        
        print_color(f"Successfully added {llm_candidates_added} LLM-generated candidates", 'green')     
        return 
        
class ExploreAlgorithm_LLMFA(ExploreAlgorithm):
    """Use LLM instead of UCB to select the best candidate."""
    def __init__(self,
                 agent: trace.Module,
                 optimizer,
                 max_buffer_size: int = 1000,
                 ucb_exploration_factor: float = 1.0,
                 logger=None,
                 num_threads: int = None,
                 llm_model: str = "gemini/gemini-2.0-flash",
                 num_samples_in_prompt: int = 5,
                 *args,
                 **kwargs):
        super().__init__(agent, optimizer, max_buffer_size, ucb_exploration_factor, logger, num_threads, *args, **kwargs)
        self.llm_model = llm_model
        self.llm = LLM(model=llm_model)
        self.selection_count = 0
    
    def select_candidate(self, buffer):
        selected_entry = self.llm_generate_candidate(buffer, verbose=False)
        self.selection_count += 1

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
        llm_response = auto_retry_with_exponential_backoff(
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
    
