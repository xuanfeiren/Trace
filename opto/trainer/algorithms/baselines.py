#TODO: Implement MinibatchwithValidation and IslandSearch Algorithms (Done)
#TODO: log raw test scores and the final best candidate for each baseline algorithm (Done)
#TODO: debug for IslandSearchAlgorithm
import numpy as np
import copy
import time
from typing import Union
from opto import trace
from opto.trainer.algorithms.algorithm import AlgorithmBase
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

def retry_with_exponential_backoff(func, max_retries=10, base_delay=1.0, operation_name="operation"):
    """
    Retry a function with exponential backoff for rate limit and other transient errors.
    
    Args:
        func: Function to retry (should be a callable with no arguments)
        max_retries: Maximum number of retry attempts
        base_delay: Base delay for exponential backoff
        operation_name: Name of the operation for logging
    
    Returns:
        Result of the function call
        
    Raises:
        The last exception encountered if all retries fail
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
            
            # Also check specific litellm exceptions
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
                # print(f"{operation_name}: Failed after {max_retries} attempts. Error: {e}")
                raise e
            elif is_retryable:
                # Special handling for rate limit errors - use longer delays
                is_rate_limit = (
                    'rate limit' in error_str or 'ratelimiterror' in error_type or
                    'quota' in error_str or 'resource has been exhausted' in error_str or
                    'code": 429' in error_str
                )
                
                if is_rate_limit:
                    # Longer delays for rate limits: 2, 8, 18, 32, 50 seconds
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
                raise e
    
    # This should never be reached, but just in case
    raise RuntimeError(f"{operation_name}: Unexpected error - reached end of retry loop")

def standard_optimization_step(agent, x, guide, info, min_score=0):
    """ Forward and compute feedback.

        Args:
            agent: trace.Module
            x: input
            guide: (question, student_answer, info) -> score, feedback
            info: additional information for the guide
            min_score: minimum score when exception happens

        Returns:
            target: output of the agent
            score: score from the guide
            feedback: feedback from the guide
        """
    try:
        target = agent(x)
        score, feedback = guide(x, target.data, info)
    except trace.ExecutionError as e:
        target = e.exception_node
        score, feedback = min_score, target.create_feedback('full')
    return target, score, feedback


class Minibatch(AlgorithmBase):
    """ General minibatch optimization algorithm. This class defines a general training and logging routine using minimbatch sampling."""

    def __init__(self,
                 agent,
                 optimizer,
                 num_threads: int = None,   # maximum number of threads to use for parallel execution
                 logger=None,
                 *args,
                 **kwargs,
                 ):
        super().__init__(agent, num_threads=num_threads, logger=logger, *args, **kwargs)
        self.optimizer = optimizer
        self.n_iters = 0  # number of iterations
        
    def _sample_minibatch(self, dataset: Dict[str, List[Any]], batch_size: int) -> Tuple[List[Any], List[Any]]:
        """Sample a minibatch from the dataset."""
        if not dataset or not dataset.get('inputs') or not dataset.get('infos'):
            print_color("Warning: Attempted to sample from an empty or malformed dataset.", color='yellow')
            return [], []
        
        dataset_size = len(dataset['inputs'])
        if dataset_size == 0:
            print_color("Warning: Dataset is empty, cannot sample minibatch.", color='yellow')
            return [], []

        actual_batch_size = min(batch_size, dataset_size)
        indices = np.random.choice(dataset_size, actual_batch_size, replace=False)
        xs = [dataset['inputs'][i] for i in indices]
        infos = [dataset['infos'][i] for i in indices]
        return xs, infos

    def train(self,
              guide,
              train_dataset,
              *,
              ensure_improvement: bool = False,  # whether to check the improvement of the agent
              improvement_threshold: float = 0.,  # threshold for improvement
              num_epochs: int = 1,  # number of training epochs
              batch_size: int = 1,  # batch size for updating the agent
              test_dataset = None,  # dataset of (x, info) pairs to evaluate the agent
              eval_frequency: int = 1,  # frequency of evaluation
              num_eval_samples: int = 5,  # number of samples to use to evaluate each input
              log_frequency: Union[int, None] = None,  # frequency of logging
              save_frequency: Union[int, None] = None,  # frequency of saving the agent
              save_path: str = "checkpoints/agent.pkl",  # path to save the agent
              min_score: Union[int, None] = None,  # minimum score to update the agent
              verbose: Union[bool, str] = False,  # whether to print the output of the agent
              num_threads: int = None,  # maximum number of threads to use (overrides self.num_threads)
              **kwargs
              ):
        """
            Given a dataset of (x, info) pairs, the algorithm will:
            1. Forward the agent on the inputs and compute the feedback using the guide.
            2. Update the agent using the feedback.
            3. Evaluate the agent on the test dataset and log the results.
        """
        self.num_eval_times = num_eval_samples
        log_frequency = log_frequency or eval_frequency  # frequency of logging (default to eval_frequency)
        num_threads = num_threads or self.num_threads  # Use provided num_threads or fall back to self.num_threads
        test_dataset = test_dataset or train_dataset  # default to train_dataset if test_dataset is not provided
        # self.num_eval_samples = num_eval_samples  # number of samples to use to evaluate each input
        self.total_samples = 0 # log the total number of samples the algorithm has seen
        self.total_proposals = 0 # log the number of total proposals the algorithm has made

        # Evaluate the agent before learning
        if eval_frequency > 0:
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
            # self.logger.log(f'Raw_test_scores_at_step_{self.n_iters}', table,  self.n_iters, color='green')
            # Extract all non-None values and compute overall average
            all_valid_scores = [score for row in eval_scores for score in row if score is not None]
            test_score = np.mean(all_valid_scores) if all_valid_scores else 0
            self.logger.log('Test score', test_score, self.n_iters, color='green')
            self.logger.log('Total samples', self.total_samples, self.n_iters, color='cyan')
            self.logger.log('Total proposals', self.total_proposals, self.n_iters, color='red')

        # Save the agent before learning if save_frequency > 0
        if save_frequency is not None and save_frequency > 0:
            self.save_agent(save_path, self.n_iters)

        # TODO random sampling with replacement
        train_scores = []
        test_score = None

        for i in range(num_epochs):
            # Train agent
            xs, infos = self._sample_minibatch(train_dataset, batch_size)
            # Backup the current value of the parameters
            backup_dict = {p: copy.deepcopy(p.data) for p in self.agent.parameters()}

            # Forward the agent on the inputs and compute the feedback using the guide
            forward = batch_run(max_workers=num_threads, description=f"Forward pass (batch size: {len(xs)})")(self.forward)
            outputs = forward(self.agent, xs, guide, infos)

            # Update the agent
            score = self.update(outputs, verbose=verbose, num_threads=num_threads, **kwargs)
            self.total_samples += len(xs)

            # Reject the update if the score on the current batch is not improved
            if ensure_improvement:
                changes = any([backup_dict[p] != p.data for p in self.agent.parameters() ])
                if changes: # Only check improvement if there're changes in the parameters for efficiency
                    if not self.has_improvement(xs, guide, infos, score, outputs, backup_dict,
                                            threshold=improvement_threshold, num_threads=num_threads):
                        self.optimizer.update(backup_dict) # Restore the backup

            self.n_iters += 1

            # Evaluate the agent after update
            if test_dataset is not None and self.n_iters % eval_frequency == 0:
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
                self.logger.log(f'Raw_test_scores_at_step_{self.n_iters}', table, self.n_iters, color='green')
                # Extract all non-None values and compute overall average
                all_valid_scores = [score for row in eval_scores for score in row if score is not None]
                test_score = np.mean(all_valid_scores) if all_valid_scores else 0
                self.logger.log('Test score', test_score, self.n_iters, color='green')

            # Save the agent
            if save_frequency is not None and save_frequency > 0 and self.n_iters % save_frequency == 0:
                self.save_agent(save_path, self.n_iters)

            # Logging
            if score is not None:  # so that mean can be computed
                train_scores.append(score)
            if self.n_iters % log_frequency == 0:
                # print(f"Epoch: {i}. Iteration: {self.n_iters}")
                self.logger.log("Instantaneous train score", score, self.n_iters)
                self.logger.log("Average train score", np.mean(train_scores), self.n_iters)
                self.logger.log("Total samples", self.total_samples, self.n_iters)
                self.logger.log("Total proposals", self.total_proposals, self.n_iters)
                # for p in self.agent.parameters():
                #     self.logger.log(f"Parameter: {p.name}", p.data, self.n_iters, color='red')
        best_params = {p: copy.deepcopy(p.data) for p in self.optimizer.parameters}
        params_values = list(best_params.values())
        self.logger.log('Final parameter 1', params_values[0], self.n_iters, color='magenta')
        self.logger.log('Final parameter 2', params_values[1], self.n_iters, color='magenta')
        return train_scores, test_score

    def evaluate(self, agent, guide, xs, infos, min_score=None, num_samples=1, num_threads=None, description=None):
        """ Evaluate the agent on the given dataset. """
        num_threads = num_threads or self.num_threads  # Use provided num_threads or fall back to self.num_threads
        eval_scores = evaluate(agent, guide, xs, infos, min_score=min_score, num_threads=num_threads,
                               num_samples=num_samples, description=description, )
        if eval_scores.ndim == 1:
            all_valid_scores = [score for score in eval_scores if score is not None]
        else:
            all_valid_scores = [score for row in eval_scores for score in row if score is not None]

        avg_score = np.mean(all_valid_scores) if all_valid_scores else 0
        
        # eval_count = len(all_valid_scores) 
        
        return float(avg_score)
        
    def has_improvement(self, xs, guide, infos, current_score, current_outputs, backup_dict, threshold=0, num_threads=None, *args, **kwargs):
        # This function can be overridden by subclasses to implement their own improvement check.
        """ Check if the updated agent is improved compared to the current one.

            Args:
                xs: inputs
                infos: additional information for the guide
                current_score: current score of the agent
                current_outputs: outputs of the agent, guide interaction
                backup_dict: backup of the current value of the parameters
                improvement_threshold: threshold for improvement
                num_threads: maximum number of threads to use
        """
        num_threads = num_threads or self.num_threads  # Use provided num_threads or fall back to self.num_threads
        new_score = self.evaluate(self.agent, guide, xs, infos, num_threads=num_threads,
                                 description=f"Checking improvement (iteration {self.n_iters})",
                                 *args, **kwargs)  # evaluate the updated agent
        self.total_samples += len(xs) # more samples have been used to evaluate the agent
        if new_score is None or new_score <= current_score - threshold:
            print_color(f"Update rejected: Current score {current_score}, New score {new_score}", 'red')
            return False
        else:
            print_color(f"Update accepted: Current score {current_score}, New score {new_score}", 'green')
            return True


    def forward(self, agent, x, guide, info):
        """ Forward the agent on the input and compute the feedback using the guide.
            Args:
                agent: trace.Module
                x: input
                guide: (question, student_answer, info) -> score, feedback
                info: additional information for the guide
            Returns:
                outputs that will be used to update the agent
        """
        raise NotImplementedError("Subclasses must implement this method")

    def update(self, outputs, verbose=False, num_threads=None, **kwargs):
        """ Subclasses can implement this method to update the agent.
            Args:
                outputs: returned value from self.step
                verbose: whether to print the output of the agent
                num_threads: maximum number of threads to use (overrides self.num_threads)
            Returns:
                score: average score of the minibatch of inputs
        """
        num_threads = num_threads or self.num_threads  # Use provided num_threads or fall back to self.num_threads
        raise NotImplementedError("Subclasses must implement this method")



@trace.bundle()
def batchify(*items):
    """ Concatenate the items into a single string """
    output = ''
    for i, item in enumerate(items):
        output += f'ID {[i]}: {item}\n'
    return output


class MinibatchAlgorithm(Minibatch):
    """
        The computed output of each instance in the minibatch is aggregated and a batched feedback is provided to update the agent.
    """

    def forward(self, agent, x, guide, info):
        return standard_optimization_step(agent, x, guide, info)  # (score, target, feedback)

    def update(self, outputs, verbose=False, num_threads=None, **kwargs):
        """ Subclasses can implement this method to update the agent.
            Args:
                outputs: returned value from self.step
                verbose: whether to print the output of the agent
                num_threads: maximum number of threads to use (overrides self.num_threads)
            Returns:
                score: average score of the minibatch of inputs

        """
        num_threads = num_threads or self.num_threads  # Use provided num_threads or fall back to self.num_threads

        scores, targets, feedbacks = [], [], []
        # Concatenate the targets and feedbacks into a single string
        for target, score, feedback in outputs:
            scores.append(score)
            targets.append(target)
            feedbacks.append(feedback)
        target = batchify(*targets)
        feedback = batchify(*feedbacks).data  # str
        average_score = np.mean(scores) if all([s is not None for s in scores]) else None

        # Update the agent using the feedback
        self.optimizer.zero_feedback()
        self.optimizer.backward(target, feedback)
        self.optimizer_step(verbose=verbose, num_threads=num_threads, **kwargs)  # update the agent

        return average_score  # return the average score of the minibatch of inputs

    def optimizer_step(self, bypassing=False, verbose=False, num_threads=None, **kwargs):
        """ Subclasses can implement this method to update the agent. """
        # We separate this method from the update method to allow subclasses to implement their own optimization step.
        
        
        # Backup current parameters before attempting update
        current_params = {p: copy.deepcopy(p.data) for p in self.agent.parameters()}
        
        # Wrap optimizer.step with retry logic
        def optimizer_step_func():
            params = self.optimizer.step(bypassing=bypassing, verbose=verbose, **kwargs)
            self.total_proposals += 1
            return params
        
        try:
            return retry_with_exponential_backoff(
                optimizer_step_func, 
                operation_name=f"Optimizer step (iteration {getattr(self, 'n_iters', 'unknown')})"
            )
        except Exception as e:
            
            # If all retries failed, fall back to current parameters
            print(f"Optimizer step failed after all retries. Falling back to current parameters. Error: {e}")
            self.optimizer.update(current_params)
            return current_params


class BasicSearchAlgorithm(MinibatchAlgorithm):
    """ A basic search algorithm that calls the optimizer multiple times to get candidates and selects the best one based on validation set. """
    """ This is a modified version of the BasicSearchAlgorithm, every step randomly samples a minibatch from the training set. If setting num_proposals to 2, validation set has size 50, then at each step, the algorithm will evaluate 2*50=100 inputs. 20 steps will evaluate 20*100=2000 inputs."""
    def train(self,
              guide, # guide to provide feedback
              train_dataset,  # dataset of (x, info) pairs to train the agent
              *,
              validate_dataset = None, # dataset of (x, info) pairs to evaluate the agent for candidate selection
              validate_guide = None,  #  to provide scores for the validation set
              num_proposals = 4,  # number of proposals to get from the optimizer
              num_epochs = 20,  # number of training epochs
              batch_size = 2,  # batch size for updating the agent
              test_dataset = None, # dataset of (x, info) pairs to evaluate the agent
              eval_frequency = 1, # frequency of evaluation
              log_frequency = None,  # frequency of logging
              min_score = None,  # minimum score to update the agent
              verbose = False,  # whether to print the output of the agent
              num_threads = None,  # maximum number of threads to use
              **kwargs
              ):

        self.num_proposals = num_proposals
        self.validate_dataset = validate_dataset or train_dataset  # default to train_dataset
        self.validate_guide = validate_guide or guide
        self.min_score = min_score
        self.current_score = None
        
        return super().train(guide, train_dataset, num_epochs=num_epochs, batch_size=batch_size,
                      test_dataset=test_dataset, eval_frequency=eval_frequency, log_frequency=log_frequency,
                      min_score=min_score, verbose=verbose, num_threads=num_threads, **kwargs)

    # This code should be reusable for other algorithms
    def optimizer_step(self, bypassing=False, verbose=False, num_threads=None, **kwargs):
        """ Use the optimizer to propose multiple updates and select the best one based on validation score. """

        num_threads = num_threads or self.num_threads  # Use provided num_threads or fall back to self.num_threads

        def validate():
            """ Validate the agent on the validation dataset. """
            score = self.evaluate(self.agent,
                              self.validate_guide,
                              self.validate_dataset['inputs'],
                              self.validate_dataset['infos'],
                              min_score=self.min_score,
                              num_threads=num_threads,
                              description="Validating proposals")
            self.total_samples += len(self.validate_dataset['inputs']) # more samples have been used to validate
            return score

        # TODO perhaps we can ask for multiple updates in one query or use different temperatures in different queries
        # Generate different proposals
        step_kwargs = dict(bypassing=True, verbose='output' if verbose else False)  # we don't print the inner full message
        step_kwargs.update(kwargs)  # update with additional kwargs if provided
                
        # Use aysnc_run to run the optimizer_step in parallel
        # NOTE optimizer_step is coupled via async_run 
        # update_dicts = async_run([super().optimizer_step]*self.num_proposals,
        #                         kwargs_list=[step_kwargs] * self.num_proposals,
        #                         max_workers=num_threads,
        #                         description=f"Generating {self.num_proposals} proposals")  # async step
        update_dicts = []
        while len(update_dicts) < self.num_proposals:
            try:
                update_dict = super().optimizer_step(**step_kwargs)
                update_dicts.append(update_dict)
            except Exception as e:
                print(f"Error in optimizer step: {e}")
                continue
        
        # Validate the proposals
        candidates = []
        backup_dict = {p: copy.deepcopy(p.data) for p in self.agent.parameters()}  # backup the current value
        for update_dict in update_dicts:
            if len(update_dict) == 0:
                continue
            self.optimizer.update(update_dict)  # set the agent with update_dict
            score = validate()  # check the score on the validation set
            candidates.append((score, update_dict))
            self.optimizer.update(backup_dict)  # restore the backup

        # Include the current parameter as a candidate
        if self.current_score is None:
            self.current_score = validate()
        candidates.append((self.current_score, backup_dict))

        # Find the candidate with the best score
        best_score, best_update = max(candidates, key=lambda x: x[0])
        self.current_score = best_score

        if verbose:
            print_color(f"Best score: {best_score} out of scores {[c[0] for c in candidates]}", 'green')
            print(f"Selected Update:\n{best_update}")

        # Make the best update
        self.optimizer.update(best_update)

        # Logging
        self.logger.log('Validation score', best_score, self.n_iters, color='green')

class MinibatchwithValidation(MinibatchAlgorithm):
    """ This is a modified version of the MinibatchAlgorithm, every step randomly samples a minibatch from the training set, for total 20 steps using 20*2=40 inputs.
    We want to make the total samples to be about 2000. So After 20 steps we got 21 candidates. For each candidate, we do 100 evaluations (evaluate on each input in the validation set twice. 
    So we got 21*100=2100 evaluations.
    Output the candidate with the highest validation score for the final test.
    """
    def add_new_candidate(self, candidate_params_dict):
        candidate_entry = {
                    'params': candidate_params_dict,
                    'score_sum': 0,
                    'eval_count': 0,
                }
        self.buffer.append(candidate_entry)

    def buffer_validation(self):
        for i,candidate in enumerate(self.buffer):
            self.optimizer.update(candidate['params'])
            avg_score = self.evaluate(self.agent, self.validate_guide, self.validate_dataset['inputs'], self.validate_dataset['infos'],
                                min_score=self.min_score, num_threads=self.num_threads,num_samples=self.validate_times,
                                description=f"Final validation of candidate {i} in {len(self.buffer)} candidates")  
            candidate['mean_score'] = avg_score
            self.total_samples += len(self.validate_dataset['inputs'])*self.validate_times
            print_color(f"Candidate {i} in {len(self.buffer)} candidates: Mean score {avg_score}", 'green')
        return self.buffer
    
    def train(self,
              guide, # guide to provide feedback
              train_dataset,  # dataset of (x, info) pairs to train the agent
              *,
              validate_dataset = None, # dataset of (x, info) pairs to evaluate the agent for candidate selection
              validate_guide = None,  #  to provide scores for the validation set
              num_proposals = 4,  # number of proposals to get from the optimizer
              num_epochs = 20,  # number of training epochs
              batch_size = 2,  # batch size for updating the agent
              test_dataset = None, # dataset of (x, info) pairs to evaluate the agent
              eval_frequency = 1, # frequency of evaluation
              log_frequency = None,  # frequency of logging
              min_score = None,  # minimum score to update the agent
              verbose = False,  # whether to print the output of the agent
              num_threads = 20,  # maximum number of threads to use
              num_eval_samples = 5,  # number of samples to use to evaluate each input
              **kwargs
              ):
        self.buffer = deque(maxlen=50) 

        self.num_proposals = num_proposals
        self.validate_dataset = validate_dataset or train_dataset  # default to train_dataset
        self.validate_guide = validate_guide or guide
        self.min_score = min_score
        self.current_score = None
        self.validate_times = 2 # To use the sample budget
        self.num_eval_times = num_eval_samples # number of times to evaluate each candidate
        log_frequency = log_frequency or eval_frequency  # frequency of logging (default to eval_frequency)
        num_threads = num_threads or self.num_threads  # Use provided num_threads or fall back to self.num_threads
        test_dataset = test_dataset or train_dataset  # default to train_dataset if test_dataset is not provided
        # self.num_eval_samples = num_eval_samples  # number of samples to use to evaluate each input
        self.total_samples = 0 # log the total number of samples the algorithm has seen
        self.total_proposals = 0 # log the number of total proposals the algorithm has made

        # Evaluate the agent before learning
        if eval_frequency > 0:
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
            # self.logger.log(f'Raw_test_scores_at_step_{self.n_iters}', table, self.n_iters, color='green')
            # Extract all non-None values and compute overall average
            all_valid_scores = [score for row in eval_scores for score in row if score is not None]
            test_score = np.mean(all_valid_scores) if all_valid_scores else 0
            self.logger.log('Test score', test_score, self.n_iters, color='green')
            self.logger.log('Total samples', self.total_samples, self.n_iters, color='cyan')
            self.logger.log('Total proposals', self.total_proposals, self.n_iters, color='red')


        # TODO random sampling with replacement
        train_scores = []
        test_score = None
        
        candidate_params_dict = {p: copy.deepcopy(p.data) for p in self.agent.parameters()}  # backup the current value
        self.add_new_candidate(candidate_params_dict)
        for i in range(num_epochs):
            # Train agent
            xs, infos = self._sample_minibatch(train_dataset, batch_size)
            # Backup the current value of the parameters
            backup_dict = {p: copy.deepcopy(p.data) for p in self.agent.parameters()}

            # Forward the agent on the inputs and compute the feedback using the guide
            forward = batch_run(max_workers=num_threads, description=f"Forward pass (batch size: {len(xs)})")(self.forward)
            outputs = forward(self.agent, xs, guide, infos)

            # Update the agent
            score = self.update(outputs, verbose=verbose, num_threads=num_threads, **kwargs)
            self.total_samples += len(xs)
            # Add the new candidate to the buffer
            candidate_params_dict = {p: copy.deepcopy(p.data) for p in self.agent.parameters()}  # backup the current value
            self.add_new_candidate(candidate_params_dict)
            # Reject the update if the score on the current batch is not improved
            

            self.n_iters += 1

            # Evaluate the agent after update
            if test_dataset is not None and self.n_iters % eval_frequency == 0:
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
                self.logger.log(f'Raw_test_scores_at_step_{self.n_iters}', table, self.n_iters, color='green')
                # Extract all non-None values and compute overall average
                all_valid_scores = [score for row in eval_scores for score in row if score is not None]
                test_score = np.mean(all_valid_scores) if all_valid_scores else 0
                self.logger.log('Test score', test_score, self.n_iters, color='green')

            

            # Logging
            if score is not None:  # so that mean can be computed
                train_scores.append(score)
            if self.n_iters % log_frequency == 0:
                # print(f"Epoch: {i}. Iteration: {self.n_iters}")
                self.logger.log("Instantaneous train score", score, self.n_iters)
                self.logger.log("Average train score", np.mean(train_scores), self.n_iters)
                self.logger.log("Total samples", self.total_samples, self.n_iters)
                self.logger.log("Total proposals", self.total_proposals, self.n_iters)
                # for p in self.agent.parameters():
                #     self.logger.log(f"Parameter: {p.name}", p.data, self.n_iters, color='red')
        print_color(f"Candidate generation finished. Start validation.", 'yellow')
        self.buffer_validation()
        candidate_to_test = max(self.buffer, key=lambda c: c['mean_score'])
        self.optimizer.update(candidate_to_test['params'])
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
        self.logger.log(f'Raw_test_scores_at_step_{self.n_iters}', table, self.n_iters+1, color='green')
        # Extract all non-None values and compute overall average
        all_valid_scores = [score for row in eval_scores for score in row if score is not None]
        test_score = np.mean(all_valid_scores) if all_valid_scores else 0
        self.logger.log('Test score', test_score, self.n_iters+1, color='green')
        self.logger.log('Total samples', self.total_samples, self.n_iters+1, color='cyan')
        self.logger.log('Total proposals', self.total_proposals, self.n_iters+1, color='red')
        params_values = list(candidate_to_test['params'].values())
        self.logger.log('Final parameter 1', params_values[0], self.n_iters+1, color='magenta')
        self.logger.log('Final parameter 2', params_values[1], self.n_iters+1, color='magenta')


        return train_scores, test_score

class Island:
    """An Island has a buffer (deque object) of candidate entries. It also has attribute of best_candidate_dict and best_score."""
    def __init__(self, buffer, best_candidate_dict, best_score):
        self.buffer = buffer
        self.best_candidate_dict = best_candidate_dict
        self.best_score = best_score

class IslandSearchAlgorithm(MinibatchAlgorithm):
    """This Island Search Algorithm is a baseline which only uses scores.
    The evolution process is:
    1. Initialize m islands with the initial candidate
    2. At each step, for each island create a few-shot prompt from candidate-score pairs, to generate new candidates. Add the new candidate to the same island
    3. Every once in a while, discard m/2 islands with low scores. Then initialize m/2 islands from candidates with high scores from existing islands.
    """
    """About the num_samples and num_proposals budget. Plan to set num_islands to 4, num_LLM_samples to 2, then at each step we propose 8 proposals and evaluate 8*50=400 samples. 5 epochs will use 5*400=2000 samples."""
    def __init__(self,
                 agent: trace.Module,
                 optimizer,
                 logger=None,
                 num_islands: int = 4,
                 num_threads: int = 1,
                 llm_model: str = "gemini/gemini-2.0-flash",
                 num_samples_in_prompt: int = 5,
                 num_LLM_samples: int = 2,
                 *args,
                 **kwargs):
        super().__init__(agent, optimizer, logger=logger, num_threads=num_threads, *args, **kwargs)
        self.num_islands = num_islands
        self.num_samples_in_prompt = num_samples_in_prompt
        self.llm = LLM(model=llm_model)
        self.num_eval_times = 5 # number of times to evaluate each candidate
        self.num_LLM_samples = num_LLM_samples
        print_color(f"Initialized IslandSearchAlgorithm with num_islands: {num_islands}, llm_model: {llm_model}", "cyan")
        
    def _sample_minibatch(self, dataset: Dict[str, List[Any]], batch_size: int) -> Tuple[List[Any], List[Any]]:
        """Sample a minibatch from the dataset."""
        if not dataset or not dataset.get('inputs') or not dataset.get('infos'):
            print_color("Warning: Attempted to sample from an empty or malformed dataset.", color='yellow')
            return [], []
        
        dataset_size = len(dataset['inputs'])
        if dataset_size == 0:
            print_color("Warning: Dataset is empty, cannot sample minibatch.", color='yellow')
            return [], []

        actual_batch_size = min(batch_size, dataset_size)
        indices = np.random.choice(dataset_size, actual_batch_size, replace=False)
        xs = [dataset['inputs'][i] for i in indices]
        infos = [dataset['infos'][i] for i in indices]
        return xs, infos
    
    def _evaluate_candidate(self, 
                            params_to_eval_dict: Dict[str, Any], 
                            dataset: Dict[str, List[Any]], 
                            guide, 
                            num_threads: Optional[int] = None,
                            num_eval_times: int = 1,
                            evaluation_batch_size: int = None,
                            ) -> Tuple[float, int]:
        """One self-defined evaluation function, could evaluate the dataset on randomly sampled evaluation_batch_size inputs. By default, it will evaluate the entire dataset."""
        if evaluation_batch_size is None: # By default, evaluate the entire dataset.
            evaluation_batch_size = len(dataset['inputs'])

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

    def _llm_generate_candidate(self, buffer, num_LLM_samples: int = 1) -> List[Dict[trace.nodes.ParameterNode, str]]:
        """
        Prompts an LLM with current buffer candidates to generate new string values for parameters.
        Retries until num_LLM_samples successful candidates are generated.
        Returns a list of dictionaries mapping ParameterNode objects to new string values.
        """
        # print_color("Attempting to generate candidate using LLM...", "blue")
        if not buffer:
            print_color("LLM generation: Buffer is empty, cannot provide context to LLM.", "yellow")
            return []

        # Filter buffer to only include candidates with valid mean scores
        valid_candidates = [c for c in buffer if c.get('mean_score') is not None and c.get('mean_score') != -float('inf') and c.get('mean_score') != float('inf')]
        
        if not valid_candidates:
            print_color("LLM generation: No candidates with valid mean scores found.", "yellow")
            return []
        
        sorted_buffer = sorted(valid_candidates, key=lambda c: c.get('mean_score', -float('inf')), reverse=True)
        # Include first, last, and evenly spaced middle candidates
        if len(sorted_buffer) <= self.num_samples_in_prompt:
            prompt_candidates = sorted_buffer
        elif self.num_samples_in_prompt <= 2:
            # If only 1-2 samples requested, take first and optionally last
            prompt_candidates = sorted_buffer[:self.num_samples_in_prompt]
        else:
            # Take first, last, and evenly spaced middle candidates
            prompt_candidates = [sorted_buffer[0]]  # First (highest mean score)
            if self.num_samples_in_prompt > 2:
                # Calculate indices for middle candidates
                middle_count = self.num_samples_in_prompt - 2  # Exclude first and last
                if middle_count > 0 and len(sorted_buffer) > 2:
                    # Evenly space middle candidates between index 1 and len-2
                    middle_indices = [int(1 + i * (len(sorted_buffer) - 2) / (middle_count + 1)) 
                                    for i in range(1, middle_count + 1)]
                    prompt_candidates.extend([sorted_buffer[i] for i in middle_indices])
            prompt_candidates.append(sorted_buffer[-1])  # Last (lowest mean score)
        
        serializable_candidate_summaries = []
        for cand_entry in prompt_candidates:
            parameter_node_1 = list(cand_entry['params'].keys())[0]
            try:
                name_1 = parameter_node_1.py_name
            except:
                breakpoint()
                print_color(f"Parameter node error", "red")
            summary = {
                # "parameters":  {getattr(p,'py_name'): copy.deepcopy(p.data) for p in cand_entry['params']},
                "parameters":  {getattr(p,'py_name'): cand_entry['params'][p] for p in cand_entry['params'].keys()},
                "mean_score": round(cand_entry.get('mean_score',0), 4),
            }
            serializable_candidate_summaries.append(summary)
        
        example_param_structure_json_str = {getattr(p,'py_name'): copy.deepcopy(p.data) for p in self.agent.parameters()}

        prompt_messages = [
            {"role": "system", "content": "You are an expert in model optimization. Your task is to propose new string values for model parameters with high mean scores. Please output ONLY a valid JSON dictionary where keys are parameter names and values are the new string values for those parameters, matching the example structure provided. Do not add any explanations or markdown formatting around the JSON."},
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
                
                llm_response = retry_with_exponential_backoff(
                    llm_call,
                    max_retries=10,
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


    def train(self,
              guide,        
              train_dataset,
              validate_dataset: Optional[Dict[str, List[Any]]] = None,  # Validation set for evaluation, defaults to train_dataset
              test_dataset: Optional[Dict[str, List[Any]]] = None,
              train_batch_size: int = 2, 
              num_epochs: int = 5,
              discard_frequency: int = 2, # Discard m/2 islands with low scores every discard_frequency epochs
              evaluation_batch_size: int = 20, # Renamed from validation_batch_size, used for all explicit evaluations
              verbose: Union[bool, str] = False,
              num_threads: Optional[int] = None,
              **kwargs
              ) :
        """ 1. Initialize m islands with the initial candidate
            2. At each step, for each island create a few-shot prompt from candidate-score pairs, to generate new candidates. Add the new candidate to the same island
            3. Every once in a while, discard m/2 islands with low scores. Then initialize m/2 islands from candidates with high scores from existing islands."""
        self.total_samples = 0
        self.total_proposals = 0
        self.min_score = 0
        # Initialize m islands with the initial parameter. Each island is a deque() with maxlen=50.
        self.islands = [Island(deque(maxlen=50), None, -np.inf) for _ in range(self.num_islands)]
        initial_params_dict = {p: copy.deepcopy(p.data) for p in self.optimizer.parameters}
        eval_scores = evaluate(self.agent,
                                        guide, 
                                        test_dataset['inputs'],
                                        test_dataset['infos'],
                                        min_score=self.min_score,
                                        num_threads=num_threads,
                                        num_samples=self.num_eval_times,
                                        description=f"Evaluating candidate")
                 # Create table with explicit column names
        # columns = [f'Eval_{i+1}' for i in range(eval_scores.shape[1])]
        # table = self.logger.wandb.Table(columns=columns, data=eval_scores.tolist())
        # self.logger.log(f'Raw_test_scores_at_step_{phase+1}', table, phase+1, color='green')
        # Extract all non-None values and compute overall average
        all_valid_scores = [score for row in eval_scores for score in row if score is not None]
        test_score = np.mean(all_valid_scores) if all_valid_scores else 0
        self.logger.log('Test score', test_score, 0, color='green')
        self.logger.log('Total samples', self.total_samples, 0, color='cyan')
        self.logger.log('Total proposals', self.total_proposals, 0, color='red')
        
        validate_score, validate_evals = self._evaluate_candidate(
            initial_params_dict, validate_dataset, guide, 
            num_threads=num_threads, num_eval_times=1, evaluation_batch_size=len(validate_dataset['inputs'])
        )
        self.total_samples += validate_evals
        
        
        initial_candidate_entry = {
            'params': initial_params_dict,
            'score_sum': 0,
            'eval_count': 0,
            'mean_score': validate_score, # Add initial validate score to the initial candidate
        }
        for island in self.islands: # Initialize the islands with initial statistics
            island.buffer.append(initial_candidate_entry)
            island.best_candidate_dict = initial_params_dict
            island.best_score = validate_score

        # At each step, for each island create a few-shot prompt from candidate-score pairs, to generate new candidates. Add the new candidate to the same island
        for epoch in range(num_epochs):
            # For each island, generate new candidates, do validation on new candidates.
            for island in self.islands:
                new_candidates = self._llm_generate_candidate(island.buffer, num_LLM_samples=self.num_LLM_samples)
                self.total_proposals += len(new_candidates)  # Track the number of proposals generated
                for candidate in new_candidates:
                    validate_score, validate_evals = self._evaluate_candidate(
                        candidate, validate_dataset, guide, 
                        num_threads=num_threads, num_eval_times=1, evaluation_batch_size=len(validate_dataset['inputs'])
                    )
                    self.total_samples += validate_evals
                    new_candidate_entry = {
                        'params': candidate,
                        'score_sum': validate_score*validate_evals,
                        'eval_count': validate_evals,
                        'mean_score': validate_score,
                    }
                    island.buffer.append(new_candidate_entry)
                    if validate_score > island.best_score:
                        island.best_candidate_dict = new_candidate_entry['params']
                        island.best_score = new_candidate_entry['mean_score']

            # At the end of each epoch, output a candidate with the highest score to do the test.
            
            # Find the island with the highest best_score
            best_island = max(self.islands, key=lambda island: island.best_score)
            
            best_overall_score = best_island.best_score
            best_overall_candidate = best_island.best_candidate_dict
            
            # Set the agent parameters to the best candidate
            self.optimizer.update(best_overall_candidate)
            
            # Perform test evaluation
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
            self.logger.log(f'Raw_test_scores_at_step_{epoch+1}', table, epoch+1, color='green')
            # Extract all non-None values and compute overall average
            all_valid_scores = [score for row in eval_scores for score in row if score is not None]
            test_score = np.mean(all_valid_scores) if all_valid_scores else 0

            # Logging at the current epoch
            self.logger.log('Test score', test_score, epoch+1, color='green')
            self.logger.log('Total samples', self.total_samples, epoch+1, color='cyan')
            self.logger.log('Total proposals', self.total_proposals, epoch+1, color='red')
            
            if epoch % discard_frequency == 0:
                # Discard the islands with the lowest best_score
                self.islands.sort(key=lambda island: island.best_score, reverse=True)
                self.islands = self.islands[:self.num_islands//2]
                num_deleted = self.num_islands - len(self.islands)
                print_color(f"Discarding {num_deleted} islands", 'yellow')
                # Initialize the discarded islands with the candidate, by randomly sampling an island and selecting its best_candidate_dict
                for _ in range(num_deleted):
                    random_island = random.choice(self.islands)
                    new_candidate_entry = {
                        'params': random_island.best_candidate_dict,
                        'mean_score': random_island.best_score,
                    }
                    new_island = Island(deque(maxlen=50), new_candidate_entry['params'], new_candidate_entry['mean_score'])
                    new_island.buffer.append(new_candidate_entry)
                    self.islands.append(new_island)
                print_color(f"Initialized {num_deleted} islands", 'green')
        # Log the final best candidate.
        params_values = list(best_overall_candidate.values())
        self.logger.log('Final parameter 1', params_values[0], epoch+1, color='magenta')
        self.logger.log('Final parameter 2', params_values[1], epoch+1, color='magenta')
               
        # Final results
        print_color("IslandSearchAlgorithm training completed.", 'blue')
        
import pandas as pd

class DetectCorrelation(MinibatchAlgorithm):
    """
    This is not a real baseline algorithm, but a tool to detect correlation between candidates before and after OptoPrime.
    In the training process, we keep a buffer with candidates and their scores. 
    At each step we randomly sample a candidate from the buffer, do the forward process on a train mini-batch to get a new proposal. 
    Evaluate this new proposal on the entire validation set. Then log scores of the original candidate and the new proposal.
    Put the new proposal into the buffer.
    Finally save the (score_before_opto, score_after_opto) pairs into a csv file, naming it as "correlation_detection.csv".
    If the file already exists, append the new data to the end of the file.
    """
    def __init__(self, agent, optimizer,num_threads: int = None, logger=None,*args, **kwargs):
        super().__init__(agent, optimizer, num_threads=num_threads, logger=logger, *args, **kwargs)

    def _sample_minibatch(self, dataset: Dict[str, List[Any]], batch_size: int) -> Tuple[List[Any], List[Any]]:
        """Sample a minibatch from the dataset."""
        if not dataset or not dataset.get('inputs') or not dataset.get('infos'):
            print_color("Warning: Attempted to sample from an empty or malformed dataset.", color='yellow')
            return [], []
        
        dataset_size = len(dataset['inputs'])
        if dataset_size == 0:
            print_color("Warning: Dataset is empty, cannot sample minibatch.", color='yellow')
            return [], []

        actual_batch_size = min(batch_size, dataset_size)
        indices = np.random.choice(dataset_size, actual_batch_size, replace=False)
        xs = [dataset['inputs'][i] for i in indices]
        infos = [dataset['infos'][i] for i in indices]
        return xs, infos
    
    def evaluate_candidate(self) -> float:
        """Evaluate the current agent on the validation set."""
        eval_scores = evaluate(self.agent,self.guide,self.validate_dataset['inputs'],self.validate_dataset['infos'],min_score=self.min_score,num_threads=self.num_threads,num_samples=1,description=f"Evaluating candidate")
        all_valid_scores = [score for score in eval_scores if score is not None]
        return np.mean(all_valid_scores) if all_valid_scores else 0
    
    def train(self,
              guide,
              train_dataset,
              validate_dataset,
              train_batch_size: int = 2,
              num_epochs: int = 10,
              verbose: Union[bool, str] = False,
              num_threads: Optional[int] = None,
              **kwargs
              ):
        """Do the training according to the algorithm description above."""
        self.buffer = deque(maxlen=2000)
        self.min_score = 0
        self.guide = guide
        self.validate_dataset = validate_dataset
        # evaluate the initial candidate
        initial_score = self.evaluate_candidate()
        initial_candidate_entry = {
            'params': {p: copy.deepcopy(p.data) for p in self.optimizer.parameters},
            'mean_score': initial_score, # Add initial validate score to the initial candidate
        }
        self.buffer.append(initial_candidate_entry)
        score_pairs = []
        for epoch in range(num_epochs):
            # randomly sample a candidate from the buffer
            random_candidate_entry = random.choice(self.buffer)
            self.optimizer.update(random_candidate_entry['params'])
            train_xs, train_infos = self._sample_minibatch(train_dataset, train_batch_size)
            forward = batch_run(max_workers = num_threads, description = f"Forward pass (batch size: {len(train_xs)})")(self.forward)
            outputs = forward(self.agent,train_xs,guide, train_infos)
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
            
            new_params_dict = retry_with_exponential_backoff(
                optimizer_step_call,
                max_retries=10,
                operation_name="Optimizer step"
            )
            
            if not isinstance(new_params_dict, dict) or not new_params_dict:
                continue

            # Ensure new_params_dict contains all parameters from optimizer
            for param in self.optimizer.parameters:
                if param not in new_params_dict:
                    new_params_dict[param] = copy.deepcopy(param.data)
            
            self.optimizer.update(new_params_dict)
            new_score = self.evaluate_candidate()
            # Add new candidate to buffer
            new_candidate_entry = {
                'params': new_params_dict,
                'mean_score': new_score,
            }
            self.buffer.append(new_candidate_entry)
            # Log the new candidate
            # print_color(f"Score before and after OptoPrime: {random_candidate_entry['mean_score']}, {new_score}", 'green')
            score_pairs.append((random_candidate_entry['mean_score'], new_score))
            self.logger.log('Score before and after OptoPrime', (random_candidate_entry['mean_score'], new_score), epoch+1, color='green')
        # Save the score pairs to a csv file
        df = pd.DataFrame(score_pairs, columns=['score_before_opto', 'score_after_opto'])
        import os
        if os.path.exists('correlation_detection.csv'):
            # File exists, append without header
            df.to_csv('correlation_detection.csv', mode='a', header=False, index=False)
            print_color(f"Appended {len(score_pairs)} score pairs to existing correlation_detection.csv", 'green')
        else:
            # File doesn't exist, create new with header
            df.to_csv('correlation_detection.csv', index=False)
            print_color(f"Created correlation_detection.csv and saved {len(score_pairs)} score pairs", 'green')
        return 