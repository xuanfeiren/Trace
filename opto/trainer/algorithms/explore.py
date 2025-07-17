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

# def evaluate(agent, guide, inputs, infos, min_score=None, num_threads=None, description=None,num_samples=1):
#     """ Evaluate the agent on the inputs and return the scores

#     Args:
#         agent: The agent to evaluate
#         guide: The guide to use for evaluation
#         inputs: List of inputs to evaluate on
#         infos: List of additional information for each input
#         min_score: Minimum score to return when an exception occurs
#         num_threads: Maximum number of threads to use for parallel evaluation
#         description: Description to display in the progress bar
#     """

#     # Expand inputs and infos to have num_samples copies of each
#     expanded_inputs = []
#     expanded_infos = []
#     original_indices = []
    
#     for i, (input_item, info_item) in enumerate(zip(inputs, infos)):
#         for _ in range(num_samples):
#             expanded_inputs.append(input_item)
#             expanded_infos.append(info_item)
#             original_indices.append(i)

#     def evaluate_single(expanded_i):
#         try:
#             """create a new env for each thread"""
#             from tau_bench.envs import get_env
#             env = get_env(
#             env_name="retail",
#             user_strategy="llm",
#             user_model="gemini-2.0-flash",
#             user_provider="gemini",
#             task_split="test",
#             task_index=0  # Will be overridden during training
#         )
#             agent.set_env(env)
            
#             output = agent(expanded_inputs[expanded_i]).data
#             score = guide.metric(expanded_inputs[expanded_i], output, expanded_infos[expanded_i])
#         except:
#             score = min_score
#         return score

#     N = len(inputs)
#     expanded_N = len(expanded_inputs)
#     assert len(expanded_inputs) == len(expanded_infos), "Expanded inputs and infos must have the same length"
    
#     # Use asyncio if num_threads is not None and > 1
#     use_asyncio = num_threads is not None and num_threads > 1
#     if use_asyncio:
#         # Use provided description or generate a default one
#         eval_description = description or f"Evaluating {N} examples with {num_samples} samples each"
#         flat_scores = async_run([evaluate_single] * expanded_N, [(i,) for i in range(expanded_N)],
#                               max_workers=num_threads,
#                               description=eval_description)
#     else:
#         flat_scores = [evaluate_single(i) for i in range(expanded_N)]
    
#     # Group the flat scores back into the original structure
#     scores = [[] for _ in range(N)]
#     for expanded_i, score in enumerate(flat_scores):
#         original_i = original_indices[expanded_i]
#         scores[original_i].append(score)
    
#     return scores

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

        avg_score = np.mean(eval_scores) if  all(s is not None for s in eval_scores) else 0
        eval_count = len(eval_xs) 
        
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
            temperature = 0.1  # Low temperature for more focused sampling on high-scoring candidates
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
                
                # Add new candidate to buffer
                new_candidate_entry = {
                    'params': new_params_dict,
                    'score_sum': 0,
                    'eval_count': 0,
                    'ucb_score': None
                }
                
                self.buffer.append(new_candidate_entry)
                       
            except Exception as e:
                print_color(f"Explore: Error processing candidate: {e}", 'red')
                continue
        return 

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
                
            # Update UCB scores
            self._update_buffer_ucb_scores()
            
            # Select candidate with highest UCB score
            selected_candidate = max(self.buffer, key=lambda c: c['ucb_score'])
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
            if validation_score > -np.inf and validation_evals > 0:
                selected_candidate['score_sum'] += validation_score * validation_evals
                selected_candidate['eval_count'] += validation_evals
                self.total_samples += validation_evals
                print_color(f"UCB iteration {iteration+1}/{horizon}: "
                          f"Selected candidate score {validation_score:.4f} "
                          f"(evaluated on {validation_evals} samples)", 'cyan')

        # Return the candidate with highest mean score (pure exploitation)
        best_candidate = max(self.buffer, key=lambda c: c['score_sum'] / (c['eval_count'] or 1E-9))

        # Handle buffer overflow - keep only max_buffer_size best candidates based on mean score
        if len(self.buffer) > self.max_buffer_size:
            # Sort by mean score and keep only the top max_buffer_size candidates
            sorted_buffer = sorted(self.buffer, key=lambda c: c['score_sum'] / (c['eval_count'] or 1E-9), reverse=True)
            self.buffer = deque(sorted_buffer[:self.max_buffer_size])
            print_color(f"Buffer size reduced from {len(sorted_buffer)} to {len(self.buffer)} based on mean score", 'yellow')

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
              min_score_for_agent_update: Optional[float] = None,
              num_to_sample: int = 5,
              num_threads: Optional[int] = None,
              num_phases: int = 5,
              ucb_horizon: int = 50,
              **kwargs
              ) -> Tuple[Dict[str, Any], float]:
        """Train using explore and best_candidate phases iteratively."""
        
        # Default datasets
        if validation_dataset is None:
            validation_dataset = train_dataset
        if test_dataset is None:
            test_dataset = train_dataset
        
        num_threads = num_threads or self.num_threads
        log_frequency = log_frequency or eval_frequency
        self.min_score = min_score_for_agent_update
        
        # Initialize tracking
        self.total_samples = 0
        self.total_proposals = 0

        # Initialize buffer with initial candidate, do the initial test.
        initial_params_dict = {p: copy.deepcopy(p.data) for p in self.optimizer.parameters}
        test_score, test_evals = self._evaluate_candidate(
            initial_params_dict, test_dataset, guide, len(test_dataset['inputs']), num_threads,num_eval_times=self.num_eval_times
        )
        
        initial_candidate_entry = {
            'params': initial_params_dict,
            'score_sum': 0,
            'eval_count': 0,
            'ucb_score': None,
        }
        self.buffer.append(initial_candidate_entry)

        self.logger.log('Buffer size', len(self.buffer), 0, color='yellow')
        self.logger.log('Test score', test_score, 0, color='green')
        self.logger.log('Total samples', self.total_samples, 0, color='cyan')
        self.logger.log('Total proposals', self.total_proposals, 0, color='red')        
        
        # Main training loop
        for phase in range(num_phases):
            self._current_iteration = phase
            
            print_color(f"\n=== Phase {phase+1}/{num_phases} ===", 'blue')
            
            # Explore phase
            print_color("Starting exploration phase...", 'cyan')
            self.explore(
                guide, 
                train_dataset, 
                train_batch_size=train_batch_size,
                num_to_sample=num_to_sample,
                num_threads=num_threads
            )
            
            # Best candidate identification phase
            print_color("Starting best candidate identification phase...", 'cyan')
            best_candidate = self.ucb_best_candidate(
                horizon=ucb_horizon,
                validation_dataset=validation_dataset,
                guide=guide,
                evaluation_batch_size=evaluation_batch_size,  # Pass evaluation_batch_size
                num_threads=num_threads
            )
            best_params = best_candidate['params']
            if best_params is None:
                print_color(f"Phase {phase+1}: No best candidate found, skipping test.", 'red')
                continue
                
            # Load best candidate parameters
            self.optimizer.update(best_params)
            self.print_intervals(self.buffer)
            total_evaluations_tracker = np.sum([c['eval_count'] for c in self.buffer])
            best_mean_score = best_candidate['score_sum'] / (best_candidate['eval_count'] or 1E-9)
            ucb = self._calculate_ucb(best_candidate, total_evaluations_tracker)
            lcb = self._calculate_lcb(best_candidate, total_evaluations_tracker)
            
            # Test evaluation
            try:
                test_score, test_evals = self._evaluate_candidate(
                    best_params,
                    test_dataset,
                    guide,
                    len(test_dataset['inputs']),  # Use subset for test evaluation too
                    num_threads,
                    num_eval_times=self.num_eval_times
                )
                
                # Calculate buffer statistics
                buffer_mean_scores = []
                for candidate in self.buffer:
                    if candidate['eval_count'] > 0:
                        mean_score = candidate['score_sum'] / candidate['eval_count']
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
                self.logger.log('Best mean score', best_mean_score, phase+1, color='magenta')
                self.logger.log('UCB', ucb, phase+1, color='magenta')
                self.logger.log('LCB', lcb, phase+1, color='magenta')
                self.logger.log('Total samples', self.total_samples, phase+1, color='cyan')
                self.logger.log('Total proposals', self.total_proposals, phase+1, color='red')
                
            except Exception as e:
                print_color(f"Phase {phase+1}: Test evaluation failed: {e}", 'red')
               
        
        # Final results
        print_color("ExploreAlgorithm training completed.", 'blue')
        
        return 
    
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
    
    def _llm_generate_candidate(self) -> Optional[Dict[trace.nodes.ParameterNode, str]]:
        """
        Prompts an LLM with current buffer candidates to generate new string values for parameters.
        Returns a dictionary mapping ParameterNode objects to new string values, or None on failure.
        """
        # print_color("Attempting to generate candidate using LLM...", "blue")
        if not self.buffer:
            print_color("LLM generation: Buffer is empty, cannot provide context to LLM.", "yellow")
            return None


        # Filter buffer to only include candidates with valid UCB scores
        valid_candidates = [c for c in self.buffer if c.get('ucb_score') is not None and c.get('ucb_score') != -float('inf') and c.get('ucb_score') != float('inf')]
        
        if not valid_candidates:
            print_color("LLM generation: No candidates with valid UCB scores found.", "yellow")
            return None
        
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
        
        # print_color(f"LLM prompt (summary): {len(prompt_candidates)} candidates, structure example provided.", "magenta")
        response_format =  {"type": "json_object"}
        llm_response = self.llm(prompt_messages, response_format=response_format) 
        llm_response_str = llm_response.choices[0].message.content

        if not llm_response_str:
            print_color("LLM returned an empty response.", "red")
            return None
        
        cleaned_llm_response_str = llm_response_str.strip()

        try:
            llm_params_raw = json.loads(cleaned_llm_response_str)
            self.total_proposals += 1
        except json.JSONDecodeError as e:
            print_color(f"JSON parsing attempts failed: {e}", "red")
            print_color("Returning None.", "red")
            return None

        if not isinstance(llm_params_raw, dict):
            print_color(f"LLM output was not a JSON dictionary after parsing: {type(llm_params_raw)}", "red")
            print_color("Returning None.", "red")
            return None
        
        try:
            candidate_params_dict = self.construct_update_dict(llm_params_raw)
        except Exception as e:
            print_color(f"Error constructing update dict: {e}", "red")
            print_color("Returning None.", "red")
            return None

        return candidate_params_dict
           
    
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
                num_LLM_samples: int = 5,
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
        self._update_buffer_ucb_scores()

        valid_candidates = [c for c in self.buffer if c.get('ucb_score') is not None and c.get('ucb_score') != -float('inf') and c.get('ucb_score') != float('inf')]
        
        if not valid_candidates:
            print_color("LLM generation: No candidates with valid UCB scores found.", "yellow")
            return None
        print_color(f"Adding {num_LLM_samples} LLM-generated candidates...", 'magenta')
        
        # Generate LLM candidates in parallel
        new_candidates = async_run([self._llm_generate_candidate] * num_LLM_samples,
                                  [() for _ in range(num_LLM_samples)],
                                  max_workers=num_threads,
                                  description=f"Generating {num_LLM_samples} LLM candidates")
        
        # Process the generated candidates
        llm_candidates_added = 0
        for new_candidate in new_candidates:
            if new_candidate is not None:  # Only add if candidate is valid
                new_candidate_entry = {
                    'params': new_candidate,
                    'score_sum': 0,
                    'eval_count': 0,
                    'ucb_score': None,
                }
                self.buffer.append(new_candidate_entry)
                llm_candidates_added += 1
            else:
                print_color("LLM generated None candidate, skipping...", 'yellow')
        
        # Update total_proposals counter
        self.total_proposals += llm_candidates_added
        
        print_color(f"Successfully added {llm_candidates_added} LLM-generated candidates", 'green')     
        return 
        
