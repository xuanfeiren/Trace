#TODO: Implement MinibatchwithValidation and IslandSearch Algorithms

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

        log_frequency = log_frequency or eval_frequency  # frequency of logging (default to eval_frequency)
        num_threads = num_threads or self.num_threads  # Use provided num_threads or fall back to self.num_threads
        test_dataset = test_dataset or train_dataset  # default to train_dataset if test_dataset is not provided
        self.num_eval_samples = num_eval_samples  # number of samples to use to evaluate each input
        self.total_samples = 0 # log the total number of samples the algorithm has seen
        self.total_proposals = 0 # log the number of total proposals the algorithm has made

        # Evaluate the agent before learning
        if eval_frequency > 0:
            test_score = self.evaluate(self.agent, guide, test_dataset['inputs'], test_dataset['infos'],
                          min_score=min_score, num_threads=num_threads,num_samples=num_eval_samples,
                          description=f"Evaluating agent (iteration {self.n_iters})")  # and log
            self.logger.log('Test score', test_score, self.n_iters, color='green')

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
                test_score = self.evaluate(self.agent, guide, test_dataset['inputs'], test_dataset['infos'],
                                min_score=min_score, num_threads=num_threads,num_samples=num_eval_samples,
                                description=f"Evaluating agent (iteration {self.n_iters})")  # and log
                self.logger.log('Test score', test_score, self.n_iters, color='green')

            # Save the agent
            if save_frequency is not None and save_frequency > 0 and self.n_iters % save_frequency == 0:
                self.save_agent(save_path, self.n_iters)

            # Logging
            if score is not None:  # so that mean can be computed
                train_scores.append(score)
            if self.n_iters % log_frequency == 0:
                print(f"Epoch: {i}. Iteration: {self.n_iters}")
                self.logger.log("Instantaneous train score", score, self.n_iters)
                self.logger.log("Average train score", np.mean(train_scores), self.n_iters)
                self.logger.log("Total samples", self.total_samples, self.n_iters)
                self.logger.log("Total proposals", self.total_proposals, self.n_iters)
                # for p in self.agent.parameters():
                #     self.logger.log(f"Parameter: {p.name}", p.data, self.n_iters, color='red')

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
        self.total_proposals += 1
        
        # Backup current parameters before attempting update
        current_params = {p: copy.deepcopy(p.data) for p in self.agent.parameters()}
        
        # Wrap optimizer.step with retry logic
        def optimizer_step_func():
            return self.optimizer.step(bypassing=bypassing, verbose=verbose, **kwargs)
        
        try:
            return retry_with_exponential_backoff(
                optimizer_step_func, 
                operation_name=f"Optimizer step (iteration {getattr(self, 'n_iters', 'unknown')})"
            )
        except Exception as e:
            self.total_proposals -= 1
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
        update_dicts = async_run([super().optimizer_step]*self.num_proposals,
                                kwargs_list=[step_kwargs] * self.num_proposals,
                                max_workers=num_threads,
                                description=f"Generating {self.num_proposals} proposals")  # async step        
        self.total_proposals += self.num_proposals
        
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

        log_frequency = log_frequency or eval_frequency  # frequency of logging (default to eval_frequency)
        num_threads = num_threads or self.num_threads  # Use provided num_threads or fall back to self.num_threads
        test_dataset = test_dataset or train_dataset  # default to train_dataset if test_dataset is not provided
        self.num_eval_samples = num_eval_samples  # number of samples to use to evaluate each input
        self.total_samples = 0 # log the total number of samples the algorithm has seen
        self.total_proposals = 0 # log the number of total proposals the algorithm has made

        # Evaluate the agent before learning
        if eval_frequency > 0:
            test_score = self.evaluate(self.agent, guide, test_dataset['inputs'], test_dataset['infos'],
                          min_score=min_score, num_threads=num_threads,num_samples=num_eval_samples,
                          description=f"Evaluating agent (iteration {self.n_iters})")  # and log
            self.logger.log('Test score', test_score, self.n_iters, color='green')


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
                test_score = self.evaluate(self.agent, guide, test_dataset['inputs'], test_dataset['infos'],
                                min_score=min_score, num_threads=num_threads,num_samples=num_eval_samples,
                                description=f"Evaluating agent (iteration {self.n_iters})")  # and log
                self.logger.log('Test score', test_score, self.n_iters, color='green')

            

            # Logging
            if score is not None:  # so that mean can be computed
                train_scores.append(score)
            if self.n_iters % log_frequency == 0:
                print(f"Epoch: {i}. Iteration: {self.n_iters}")
                self.logger.log("Instantaneous train score", score, self.n_iters)
                self.logger.log("Average train score", np.mean(train_scores), self.n_iters)
                self.logger.log("Total samples", self.total_samples, self.n_iters)
                self.logger.log("Total proposals", self.total_proposals, self.n_iters)
                # for p in self.agent.parameters():
                #     self.logger.log(f"Parameter: {p.name}", p.data, self.n_iters, color='red')
        self.buffer_validation()
        candidate_to_test = max(self.buffer, key=lambda c: c['mean_score'])
        self.optimizer.update(candidate_to_test['params'])
        self.test_score = self.evaluate(self.agent, guide, test_dataset['inputs'], test_dataset['infos'],
                                min_score=min_score, num_threads=num_threads,num_samples=num_eval_samples,
                                description=f"Final test of candidate") 
        self.logger.log('Test score', self.test_score, self.n_iters, color='green')


        return train_scores, test_score