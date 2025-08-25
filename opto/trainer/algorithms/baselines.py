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
import math
from opto.trainer.utils import retry_with_exponential_backoff, sample_minibatch

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
              num_eval_samples: int = 1,  # number of samples to use to evaluate each input
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
        self.min_score = min_score
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
            if eval_scores.ndim >1:
                columns = [f'Eval_{i+1}' for i in range(eval_scores.shape[1])]
                table = self.logger.wandb.Table(columns=columns, data=eval_scores.tolist())
            # self.logger.log(f'Raw_test_scores_at_step_{self.n_iters}', table,  self.n_iters, color='green')
            # Extract all non-None values and compute overall average
                all_valid_scores = [score for row in eval_scores for score in row if score is not None]
            else:
                all_valid_scores = [score for score in eval_scores if score is not None]
            test_score = np.mean(all_valid_scores) if all_valid_scores else 0
            self.logger.log('Test score', test_score, self.n_iters, color='green')
            self.logger.log('Total samples', self.total_samples, self.n_iters, color='cyan')
            self.logger.log('Total proposals', self.total_proposals, self.n_iters, color='red')
        if eval_frequency ==1:
            score_before_opto = test_score
        # Save the agent before learning if save_frequency > 0
        if save_frequency is not None and save_frequency > 0:
            self.agent.score = test_score
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
                if eval_scores.ndim>1:
                    columns = [f'Eval_{i+1}' for i in range(eval_scores.shape[1])]
                    table = self.logger.wandb.Table(columns=columns, data=eval_scores.tolist())
                    self.logger.log(f'Raw_test_scores_at_step_{self.n_iters}', table,  self.n_iters, color='green')
                # Extract all non-None values and compute overall average
                    all_valid_scores = [score for row in eval_scores for score in row if score is not None]
                else:
                    all_valid_scores = [score for score in eval_scores if score is not None]
                test_score = np.mean(all_valid_scores) if all_valid_scores else 0
                self.logger.log('Test score', test_score, self.n_iters, color='green')
                if eval_frequency ==1:
                    score_after_opto = test_score
                    self.logger.log('score_before_opto', score_before_opto, self.n_iters, color='green')
                    self.logger.log('score_after_opto', score_after_opto, self.n_iters, color='green')
                    score_before_opto = score_after_opto
            # Save the agent
            if save_frequency is not None and save_frequency > 0 and self.n_iters % save_frequency == 0:
                self.agent.score = test_score
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
                if len(update_dict) >0:
                    update_dicts.append(update_dict)
            except Exception as e:
                print(f"Error in optimizer step: {e}")
                continue
        
        # Validate the proposals
        candidates = []
        backup_dict = {p: copy.deepcopy(p.data) for p in self.agent.parameters()}  # backup the current value
        print_color(f"Number of new proposals: {len(update_dicts)}", 'blue')
        for update_dict in update_dicts:
            # if len(update_dict) == 0:
            #     print_color("No update dict found, Continue for the next proposal", 'red')
            #     continue
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
    ## TODO: Test the performance of UCB best arm identification. Maybe try multiple steps of validation. Compare with evenly split validation.

    def print_intervals(self, buffer):
        """Print confidence intervals for debugging in the form of open intervals (LCB, UCB)"""
        print_color("Confidence intervals for all candidates:", 'cyan')
        total_evaluations_tracker = np.sum([c['eval_count'] for c in buffer])
        for i, candidate_entry in enumerate(buffer):
            lcb = self._calculate_lcb(candidate_entry, total_evaluations_tracker)
            ucb = self._calculate_ucb(candidate_entry, total_evaluations_tracker)
            mean_score = candidate_entry['score_sum'] / (candidate_entry['eval_count'] or 1E-9)
            eval_count = candidate_entry['eval_count']
            
            # Format as open interval (LCB, UCB) with mean score and evaluation count
            interval_str = f"Action {i+1}: ({lcb:.4f}, {ucb:.4f}) [mean: {mean_score:.4f}, n: {eval_count}]"
            print_color(interval_str, 'cyan')
    def add_new_candidate(self, candidate_params_dict):
        candidate_entry = {
                    'params': candidate_params_dict,
                    'score_sum': 0,
                    'eval_count': 0,
                }
        self.buffer.append(candidate_entry)

    def evenly_split_buffer_validation(self, test_dataset=None, guide=None):
        """
        Evenly split buffer validation with 10 iterations.
        In each iteration, evaluate all candidates in the buffer on validation tasks,
        then test the best candidate and log results.
        """
        if not self.buffer:
            print_color("Buffer is empty, cannot perform validation.", 'red')
            return None
            
        print_color(f"Starting evenly split buffer validation: 10 iterations with {len(self.buffer)} candidates", 'blue')
        self.evenly_split_total_samples = 0
        # Main loop: 10 iterations
        for iteration in range(5):
            print_color(f"Evenly split iteration {iteration+1}/5", 'cyan')
            
            # Randomly sample validation tasks for this iteration
            validation_batch_size = min(20, len(self.validate_dataset['inputs']))
            
            # Evaluate ALL candidates in the buffer on the sampled validation tasks
            for candidate_idx, candidate in enumerate(self.buffer):
                self.optimizer.update(candidate['params'])
                try:
                    # Evaluate on validation set subset
                    validation_score, validation_evals = self._evaluate_candidate(
                        candidate['params'], 
                        self.validate_dataset, 
                        self.validate_guide, 
                        validation_batch_size,  # Now using subset instead of entire dataset
                        self.num_threads
                    )
                except Exception as e:
                    print_color(f"Best candidate identification: Error evaluating candidate: {e}", 'red')
                    continue
                
                # Update candidate statistics
                if validation_score is not None and validation_score > -np.inf and validation_evals > 0:
                    candidate['score_sum'] += validation_score * validation_evals
                    candidate['eval_count'] += validation_evals
                    self.evenly_split_total_samples += validation_evals
            
            # Find the best candidate after evaluating all candidates
            self._update_buffer_scores()
            current_best = max(self.buffer, key=lambda c: c.get('mean_score', -float('inf')))
            
            # Test the best candidate on the full test dataset if provided
            if test_dataset is not None and guide is not None:
                self.optimizer.update(current_best['params'])
                
                # Test on full test dataset
                test_eval_scores = evaluate(self.agent, guide, test_dataset['inputs'], test_dataset['infos'],
                                          min_score=self.min_score, num_threads=self.num_threads,
                                          num_samples=self.num_eval_times,
                                          description=f"Testing best candidate at iteration {iteration+1}")
                
                # Log test results
                all_test_scores = [score for row in test_eval_scores for score in row if score is not None]
                test_score = np.mean(all_test_scores) if all_test_scores else 0
                
                if hasattr(self, 'logger'):
                    self.n_iters += 1
                    self.print_intervals(self.buffer)
                    self.logger.log('Evenly_Split_Test_Score', test_score, self.n_iters, color='green')
                    self.logger.log('Evenly_Split_Validation_Samples', self.evenly_split_total_samples, self.n_iters, color='cyan')
                    self.logger.log('Evenly_Split_Best_Val_Score', current_best['mean_score'], self.n_iters, color='yellow')
                
                print_color(f"Iteration {iteration+1}: Best val score: {current_best['mean_score']:.4f}, Test score: {test_score:.4f}", 'green')
        
        # Return the final best candidate
        candidate_to_test = max(self.buffer, key=lambda c: c['mean_score'])
        print_color(f"Evenly split validation completed. Selected candidate with val score: {candidate_to_test['mean_score']:.4f}", 'blue')
        return candidate_to_test
    # Start to implement UCB sample budget allocation
    def _calculate_ucb(self, candidate_buffer_entry: Dict, total_tracked_evaluations: int) -> float:
        """Calculates UCB score for a candidate in the buffer."""
        if candidate_buffer_entry['eval_count'] == 0:
            return float('inf')  # Explore unvisited states first
        
        mean_score = candidate_buffer_entry['score_sum'] / candidate_buffer_entry['eval_count']
       
        if total_tracked_evaluations == 0: # Should not happen if we init with one eval
             total_tracked_evaluations = 1
        exploration_term = self.ucb_exploration_factor * \
                           math.sqrt(math.log(total_tracked_evaluations) / candidate_buffer_entry['eval_count'])
        
        return mean_score + exploration_term
    
    def _calculate_lcb(self, candidate_buffer_entry: Dict, total_tracked_evaluations: int) -> float:
        """Calculates Lower Confidence Bound for a candidate in the buffer."""
        if candidate_buffer_entry['eval_count'] == 0:
            return float('-inf')  # Unvisited states get lowest bound
        
        mean_score = candidate_buffer_entry['score_sum'] / candidate_buffer_entry['eval_count']
        
       
        if total_tracked_evaluations == 0: # Should not happen if we init with one eval
             total_tracked_evaluations = 1
        
        exploration_term = self.ucb_exploration_factor * \
                           math.sqrt(math.log(total_tracked_evaluations) / candidate_buffer_entry['eval_count'])
        
        return mean_score - exploration_term
    def _update_buffer_scores(self):
        """Recalculates and updates UCB scores for all candidates in the buffer."""
        if not self.buffer:
            return
        total_evaluations_tracker = np.sum([c['eval_count'] for c in self.buffer])
        for candidate_entry in self.buffer:
            candidate_entry['ucb_score'] = self._calculate_ucb(candidate_entry, total_evaluations_tracker)
            candidate_entry['lcb_score'] = self._calculate_lcb(candidate_entry, total_evaluations_tracker)
            candidate_entry['mean_score'] = candidate_entry['score_sum'] / (candidate_entry['eval_count'] or 1E-9)

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
    def ucb_best_candidate(self, 
                      horizon: int=80, 
                      evaluation_batch_size: int = 20,
                      num_threads: Optional[int] = None,
                      test_dataset=None,
                      guide=None) -> Dict[str, Any]:
        """Select the best candidate from the buffer using UCB for horizon iterations."""
        self.ucb_total_samples = 0
        if not self.buffer:
            print_color("Buffer is empty, cannot select best candidate.", 'red')
            return None
        print_color(f"Best candidate identification: Starting {horizon} iterations", 'blue')
        # Do a initial evaluation of the buffer. For each candidate, evaluate on 20 samples from validation set
        # Total samples in this step is 20 * len(self.buffer) = 100
        for candidate in self.buffer:
            validation_score, validation_evals = self._evaluate_candidate(
                candidate['params'], 
                self.validate_dataset, 
                self.validate_guide, 
                20,  # Now using subset instead of entire dataset
                num_threads
            )
            candidate['score_sum'] += validation_score * validation_evals
            candidate['eval_count'] += validation_evals
            self.ucb_total_samples += validation_evals
            self.total_samples += validation_evals
        self._update_buffer_scores()
        current_best = max(self.buffer, key=lambda c: c['mean_score'])
        self.optimizer.update(current_best['params'])
        
        # Test on full test dataset
        test_eval_scores = evaluate(self.agent, guide, test_dataset['inputs'], test_dataset['infos'],
                                    min_score=self.min_score, num_threads=num_threads or self.num_threads,
                                    num_samples=self.num_eval_times,
                                    description=f"UCB testing...")
        
        # Log test results
        all_test_scores = [score for row in test_eval_scores for score in row if score is not None]
        test_score = np.mean(all_test_scores) if all_test_scores else 0
        
        if hasattr(self, 'logger'):
            self.n_iters += 1
            self.print_intervals(self.buffer)
            self.logger.log('Test score', test_score, self.n_iters, color='green')
            self.logger.log('UCB Validation_Samples', self.ucb_total_samples, self.n_iters, color='cyan')
            self.logger.log('Total samples', self.total_samples, self.n_iters, color='yellow')
            self.logger.log('UCB_Best_Mean_Score', current_best['mean_score'], self.n_iters, color='yellow')
        
        print_color(f"UCB iteration {0}: Best mean score: {current_best['mean_score']:.4f}, Test score: {test_score:.4f}", 'green')
        # UCB-based best arm identification
        for iteration in range(horizon):
            print_color(f"UCB iteration {iteration+1}/{horizon}: ", 'blue')
            # Update UCB scores
            self._update_buffer_scores()
            
            # Select candidate with highest UCB score
            selected_candidate = max(self.buffer, key=lambda c: c['ucb_score'])
            try:
                # Evaluate on validation set subset
                validation_score, validation_evals = self._evaluate_candidate(
                    selected_candidate['params'], 
                    self.validate_dataset, 
                    self.validate_guide, 
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
                self.ucb_total_samples += validation_evals
                self.total_samples += validation_evals
                print_color(f"UCB iteration {iteration+1}/{horizon}: "
                          f"Selected candidate score {validation_score:.4f} "
                          f"(evaluated on {validation_evals} samples)", 'cyan')
            
            # Test and log every 5 iterations
            if (iteration + 1) % 8 == 0 and test_dataset is not None and guide is not None:
                self._update_buffer_scores()
                current_best = max(self.buffer, key=lambda c: c['mean_score'])
                self.optimizer.update(current_best['params'])
                
                # Test on full test dataset
                test_eval_scores = evaluate(self.agent, guide, test_dataset['inputs'], test_dataset['infos'],
                                          min_score=self.min_score, num_threads=num_threads or self.num_threads,
                                          num_samples=self.num_eval_times,
                                          description=f"UCB testing at iteration {iteration+1}")
                
                # Log test results
                all_test_scores = [score for row in test_eval_scores for score in row if score is not None]
                test_score = np.mean(all_test_scores) if all_test_scores else 0
                
                if hasattr(self, 'logger'):
                    self.n_iters += 1
                    self.print_intervals(self.buffer)
                    self.logger.log('Test score', test_score, self.n_iters, color='green')
                    self.logger.log('UCB Validation_Samples', self.ucb_total_samples, self.n_iters, color='cyan')
                    self.logger.log('Total samples', self.total_samples, self.n_iters, color='cyan')
                    self.logger.log('UCB_Best_Mean_Score', current_best['mean_score'], self.n_iters, color='yellow')
                
                print_color(f"UCB iteration {iteration+1}: Best mean score: {current_best['mean_score']:.4f}, Test score: {test_score:.4f}", 'green')
                
        self._update_buffer_scores()
        # Return the candidate with highest mean score (pure exploitation)
        best_candidate = max(self.buffer, key=lambda c: c['mean_score'])

        return best_candidate
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
              num_eval_samples = 1,  # number of samples to use to evaluate each input
              validation_method = "ucb",  # flag to choose validation method: "evenly_split" or "ucb"
              **kwargs
              ):
        self.buffer = deque(maxlen=50) 
        self.ucb_exploration_factor = 0.3

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

        # Initial evaluation (simplified)
        if eval_frequency > 0:
            eval_scores = evaluate(self.agent, guide, test_dataset['inputs'], test_dataset['infos'],
                                 min_score=self.min_score, num_threads=num_threads,
                                 num_samples=self.num_eval_times, description=f"Initial evaluation")
            all_valid_scores = [score for row in eval_scores for score in row if score is not None]
            test_score = np.mean(all_valid_scores) if all_valid_scores else 0
            self.logger.log('Initial_Test_Score', test_score, self.n_iters, color='blue')
            self.logger.log('Total samples', self.total_samples, self.n_iters, color='cyan')
            self.logger.log('Total proposals', self.total_proposals, self.n_iters, color='red')
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
            # Logging
            if score is not None:  # so that mean can be computed
                train_scores.append(score)
            if self.n_iters % log_frequency == 0:
                # print(f"Epoch: {i}. Iteration: {self.n_iters}")
                self.logger.log("Instantaneous train score", score, self.n_iters)
                self.logger.log("Average train score", np.mean(train_scores), self.n_iters)
                # self.logger.log("Total samples", self.total_samples, self.n_iters)
                # self.logger.log("Total proposals", self.total_proposals, self.n_iters)
                # for p in self.agent.parameters():
                #     self.logger.log(f"Parameter: {p.name}", p.data, self.n_iters, color='red')
        print_color(f"Candidate generation finished. Start validation.", 'yellow')
        # Choose validation method based on flag
        print_color("Using UCB-based validation method", 'blue')
        self.ucb_best_candidate(test_dataset=test_dataset, guide=guide)
        # set all the stats in the buffer to be initial ones
        # for candidate in self.buffer:
        #     candidate['score_sum'] = 0
        #     candidate['eval_count'] = 0
        #     candidate['mean_score'] = None
        #     candidate['ucb_score'] = None
        #     candidate['lcb_score'] = None
        # # Final evaluation of the selected candidate
        
        # print_color("Using evenly split validation method", 'blue')
        # self.evenly_split_buffer_validation(test_dataset=test_dataset, guide=guide)

        # self.optimizer.update(candidate_to_test['params'])
        
        
        # # Log final results
        # all_valid_scores = [score for row in eval_scores for score in row if score is not None]
        # test_score = np.mean(all_valid_scores) if all_valid_scores else 0
        # self.logger.log('Test score', test_score, self.n_iters+1, color='green')
        # self.logger.log('Total samples', self.total_samples, self.n_iters+1, color='cyan')
        # self.logger.log('Total proposals', self.total_proposals, self.n_iters+1, color='red')
        # params_values = list(candidate_to_test['params'].values())
        # self.logger.log('Final_Parameter_1', params_values[0], self.n_iters+1, color='magenta')
        # self.logger.log('Final_Parameter_2', params_values[1], self.n_iters+1, color='magenta')


        return 
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
            self.logger.log('Score before OptoPrime', random_candidate_entry['mean_score'], epoch+1, color='green')
            self.logger.log('Score after OptoPrime', new_score, epoch+1, color='green')
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
    
class EvaluateInitialCandidate(MinibatchAlgorithm):
    """
    This is not a real baseline algorithm, but a tool to evaluate the initial candidate.
    """
    def __init__(self, agent, optimizer,num_threads: int = None, logger=None,*args, **kwargs):
        super().__init__(agent, optimizer, num_threads=num_threads, logger=logger, *args, **kwargs)
    def train(self,
              guide,
              test_dataset,
              **kwargs
              ):
        """Evaluate the initial candidate."""
        self.min_score = 0
        self.guide = guide
        max_eval_times = 5000
        
        current_dataset = test_dataset
        num_eval_times = 0
        total_tasks = len(test_dataset['inputs'])
        solved_tasks = 0
        
        print_color(f"Starting evaluation with {total_tasks} tasks", 'blue')
        
        # One new version: attempt to evaluate the agent on the test set, after each evaluation remove all the successful tasks. Repeat this process until all the tasks are solved.

        while (len(current_dataset['inputs']) > 0) and (num_eval_times < max_eval_times):
            # Sample a minibatch from the current dataset
            
            eval_scores = evaluate(self.agent, guide, current_dataset['inputs'], current_dataset['infos'], 
                                 min_score=self.min_score, num_threads=self.num_threads, 
                                 num_samples=1, description=f"Evaluating candidate iteration {num_eval_times+1}")
            num_eval_times += 1
            
            # Extract the labels of success tasks (score == 1) 
            successful_indices = []
            for i, score in enumerate(eval_scores):
                if score == 1:  # Task was successful
                    successful_indices.append(i)
            
            # Count newly solved tasks
            newly_solved = len(successful_indices)
            solved_tasks += newly_solved
            
            # Update the current dataset - remove successful tasks
            if successful_indices:
                remaining_inputs = [current_dataset['inputs'][i] for i in range(len(current_dataset['inputs'])) 
                                  if i not in successful_indices]
                remaining_infos = [current_dataset['infos'][i] for i in range(len(current_dataset['infos'])) 
                                 if i not in successful_indices]
                current_dataset = {'inputs': remaining_inputs, 'infos': remaining_infos}
            
            # Print the pass rate. passed tasks / total tasks
            pass_rate = solved_tasks / total_tasks
            remaining_tasks = len(current_dataset['inputs'])
            
            print_color(f"Iteration {num_eval_times}: Solved {newly_solved} new tasks. "
                       f"Total solved: {solved_tasks}/{total_tasks} ({pass_rate:.2%}). "
                       f"Remaining: {remaining_tasks}", 'green')
            
            # Log progress
            if hasattr(self, 'logger'):
                self.logger.log('Pass Rate', pass_rate, num_eval_times, color='green')
                self.logger.log('Solved Tasks', solved_tasks, num_eval_times, color='blue')
                self.logger.log('Remaining Tasks', remaining_tasks, num_eval_times, color='yellow')
                self.logger.log('Evaluation Iterations', num_eval_times, num_eval_times, color='cyan')
            
            # Break if all tasks are solved
            if remaining_tasks == 0:
                print_color(f"All tasks solved after {num_eval_times} evaluations!", 'green')
                break
        
        # Final summary
        final_pass_rate = solved_tasks / total_tasks
        print_color(f"Final Results: {solved_tasks}/{total_tasks} tasks solved ({final_pass_rate:.2%}) "
                   f"in {num_eval_times} evaluations", 'blue')
        
        if hasattr(self, 'logger'):
            self.logger.log('Final Pass Rate', final_pass_rate, num_eval_times, color='magenta')
            self.logger.log('Final Solved Tasks', solved_tasks, num_eval_times, color='magenta')
            self.logger.log('Total Evaluations', num_eval_times, num_eval_times, color='magenta')
        
        return num_eval_times
    
class LearnFromSuccessAlgorithm(MinibatchAlgorithm):
    """
    This is an algorithm that learns from the success of the agent. At each epoch, we run the agent on the current train dataset, if the agent solves the task (score == 1), we add the conversation history of success task to the agent's conversations dict. Then delete those success case from the train dataset.
    """
    def __init__(self, agent, optimizer, num_threads: int = None, logger=None, *args, **kwargs):
        super().__init__(agent, optimizer, num_threads=num_threads, logger=logger, *args, **kwargs)
        self.successful_conversations = []
        
    def _update_agent_conversations(self, task_indices, successful_conversations):
        """Update agent's conversations dictionary with task-specific successful examples."""
        if not successful_conversations or not task_indices:
            return
            
        # Get current conversations dictionary
        current_conversations = self.agent.conversations.data if self.agent.conversations.data else {}
        
        # Add new successful conversations for specific task indices
        for task_idx, conversation in zip(task_indices, successful_conversations):
            current_conversations[task_idx] = conversation
            
        # Update the agent's conversations dictionary using the optimizer
        if hasattr(self.agent, 'conversations'):
            update_dict = {self.agent.conversations: current_conversations}
            self.optimizer.update(update_dict)
            
            print_color(f"Updated conversations dict with {len(successful_conversations)} task-specific successful examples", 'cyan')
            print_color(f"Total conversations in dict: {len(current_conversations)}", 'blue')
        else:
            print_color("Warning: Agent does not have conversations attribute", 'yellow')

    def train(self,
              guide,
              train_dataset,
              test_dataset,
              num_epochs: int = 50,
              eval_frequency: int = 1,
              **kwargs
              ):
        """Learn from the success of the agent."""
        self.min_score = 0
        self.guide = guide
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.num_eval_times = 5
        self.n_iters = 0
        
        # Initialize tracking variables
        self.total_samples = 0
        self.total_proposals = 0
        
        # Initial evaluation
        if eval_frequency > 0:
            eval_scores = evaluate(self.agent,
                                 guide, 
                                 test_dataset['inputs'],
                                 test_dataset['infos'],
                                 min_score=self.min_score,
                                 num_threads=self.num_threads,
                                 num_samples=self.num_eval_times,
                                 description=f"Initial evaluation")
            
            # Extract all non-None values and compute overall average
            if eval_scores.ndim > 1:
                all_valid_scores = [score for row in eval_scores for score in row if score is not None]
            else:
                all_valid_scores = [score for score in eval_scores if score is not None]
            test_score = np.mean(all_valid_scores) if all_valid_scores else 0
            self.logger.log('Test score', test_score, self.n_iters, color='green')
            self.logger.log('Total samples', self.total_samples, self.n_iters, color='cyan')

        # Create a working copy of the training dataset
        current_train_inputs = train_dataset['inputs'].copy()
        current_train_infos = train_dataset['infos'].copy()
        
        for epoch in range(num_epochs):
            # Check if we have any training data left
            if not current_train_inputs:
                print_color("No more training data available. Stopping training.", 'yellow')
                break
                
            # Run agent on ALL remaining training tasks
            xs = current_train_inputs
            infos = current_train_infos
            
            print_color(f"Epoch {epoch + 1}: Running agent on {len(xs)} remaining training tasks", 'blue')
            
            # Forward pass on all remaining training tasks
            forward = batch_run(max_workers=self.num_threads, 
                              description=f"Forward pass on all {len(xs)} remaining tasks")(self.forward)
            outputs = forward(self.agent, xs, guide, infos)
            
            # Track samples used
            self.total_samples += len(xs)
            
            # Identify successful tasks and collect their conversation histories
            successful_indices = []
            successful_task_indices = []
            epoch_successful_conversations = []
            
            for i, (target, score, feedback) in enumerate(outputs):
                if score == 1:  # Task was successful
                    # Extract conversation history from the target or feedback
                    # The feedback typically contains the full conversation
                    conversation_history = str(feedback) 
                    epoch_successful_conversations.append(conversation_history)
                    successful_indices.append(i)
                    successful_task_indices.append(xs[i])  # Store the actual task index for the conversation dict
                    
            # Add successful conversations to our collection
            self.successful_conversations.extend(epoch_successful_conversations)
            
            # Log successful tasks found in this epoch
            self.logger.log('Successful tasks this epoch', len(epoch_successful_conversations), epoch + 1, color='green')
            self.logger.log('Total successful tasks', len(self.successful_conversations), epoch + 1, color='blue')
            
            # Remove successful tasks from the training dataset
            if successful_indices:
                # Sort indices in reverse order to avoid index shifting issues
                successful_indices_sorted = sorted(set(successful_indices), reverse=True)
                for idx in successful_indices_sorted:
                    current_train_inputs.pop(idx)
                    current_train_infos.pop(idx)
                
                print_color(f"Removed {len(successful_indices)} successful tasks. "
                           f"Remaining training tasks: {len(current_train_inputs)}", 'cyan')
            
            # Update agent's conversations dictionary with successful conversations
            if epoch_successful_conversations:
                self._update_agent_conversations(successful_task_indices, epoch_successful_conversations)
                # self.total_proposals += 1  # Count parameter updates as proposals
            
            self.n_iters += 1
            
            # Evaluate the updated agent on the test dataset
            if epoch % eval_frequency == 0:
                eval_scores = evaluate(self.agent,
                                     guide, 
                                     test_dataset['inputs'],
                                     test_dataset['infos'],
                                     min_score=self.min_score,
                                     num_threads=self.num_threads,
                                     num_samples=self.num_eval_times,
                                     description=f"Evaluation after epoch {epoch + 1}")
                
                # Extract all non-None values and compute overall average
                if eval_scores.ndim > 1:
                    all_valid_scores = [score for row in eval_scores for score in row if score is not None]
                else:
                    all_valid_scores = [score for score in eval_scores if score is not None]
                test_score = np.mean(all_valid_scores) if all_valid_scores else 0
                
                # Log results
                self.logger.log('Test score', test_score, epoch + 1, color='green')
                self.logger.log('Total samples', self.total_samples, epoch + 1, color='cyan')
                self.logger.log('Remaining training tasks', len(current_train_inputs), epoch + 1, color='yellow')
                
                print_color(f"Epoch {epoch + 1}: Test score: {test_score:.4f}, "
                           f"Successful conversations: {len(self.successful_conversations)}, "
                           f"Remaining training tasks: {len(current_train_inputs)}", 'green')
        
        print_color(f"Training completed. Total successful conversations collected: {len(self.successful_conversations)}", 'blue')
        
        return self.successful_conversations
    
from opto.trainer.utils import evaluate_agent
class UCBAlgorithm(MinibatchAlgorithm):
    """
    This is a search algorithm that uses the UCB score to select the candidate. At each epoch, the algorithm selects the candidate with the highest UCB score in the buffer, to do the forward process, and generate the next candidate. We have two options of estimating the score of the new candidate:
    1. Using the control variate method to estimate the score of the new candidate. (when enable_control_variate is True)
    2. Using the raw score of the new candidate. (when enable_control_variate is False)
    """
    def __init__(self, agent, optimizer, num_threads: int = None, logger=None,ucb_exploration_factor: float = 0.1, enable_control_variate: bool = False, *args, **kwargs):
        super().__init__(agent, optimizer, num_threads=num_threads, logger=logger, *args, **kwargs)
        self.buffer = deque(maxlen=500)
        self.exploration_factor = ucb_exploration_factor
        self.enable_control_variate = enable_control_variate
        self.total_samples = 0
        self.min_score = 0
        # initialize the buffer with the initial parameter entry
        initial_update_dict = {p: copy.deepcopy(p.data) for p in self.optimizer.parameters}
        initial_candidate_entry = {
            'params': initial_update_dict,
            'score_sum': 0,
            'eval_count': 0,
        }
        self.buffer.append(initial_candidate_entry)

    def print_buffer_statistics(self):
        """print the buffer statistics"""
        print_color("Buffer statistics:", "magenta")
        for i,candidate_entry in enumerate(self.buffer):            
            # print the mean score and evaluation count, and confidence intervals.
            print_color(f"Candidate {i}. Mean score {candidate_entry['mean_score']}, eval_count {candidate_entry['eval_count']}. Confidence interval: [{candidate_entry['lcb_score']} , {candidate_entry['ucb_score']}]", "green")
            # for p in candidate_entry['params']:
            #     print_color(f"Parameter value: {candidate_entry['params'][p]}", "cyan")
        return 
    
    def update_buffer_scores(self):
        """Update the buffer statistics."""
        for candidate_entry in self.buffer:
            candidate_entry['mean_score'] = candidate_entry['score_sum'] / (candidate_entry['eval_count'] or 1E-9)
            if candidate_entry['eval_count'] == 0:
                candidate_entry['ucb_score'] = np.inf
                candidate_entry['lcb_score'] = -np.inf
            else:
                candidate_entry['ucb_score'] = candidate_entry['mean_score'] + self.exploration_factor * np.sqrt(np.log(self.total_samples) / candidate_entry['eval_count'] )
                candidate_entry['ucb_score'] = np.clip(candidate_entry['ucb_score'], 0, 1)
                candidate_entry['lcb_score'] = candidate_entry['mean_score'] - self.exploration_factor * np.sqrt(np.log(self.total_samples) / candidate_entry['eval_count'] )
                candidate_entry['lcb_score'] = np.clip(candidate_entry['lcb_score'], 0, 1)
        return 
    
    def update(self, outputs, verbose=False, num_threads=None, **kwargs):
        """I made some modifications to the original update method.
        The original update method is:
        """
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
        while True: # retry until the new parameters are generated successfully
            try:
                new_update_dict = self.optimizer.step(**step_kwargs)
                break
            except Exception as e:
                print_color(f"Error when generating new parameters: {e}", "red")

        return average_score, new_update_dict  # return the average score of the minibatch of inputs
    
    def train(self,
              guide,
              train_dataset,
              *,
              num_epochs: int = 1,  # number of training epochs
              batch_size: int = 1,  # batch size for updating the agent
              test_dataset = None,  # dataset of (x, info) pairs to evaluate the agent
              eval_frequency: int = 5,  # frequency of evaluation
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
            At each epoch, the algorithm will:
            1. Forward the agent (using the parameter with the highest UCB score) on the inputs and compute the feedback using the guide.
            2. Generate a new parameter using the optimizer.
            3. Evaluate the agent on the test dataset and log the results.
        """
        # Initial evaluation
        if eval_frequency > 0:
            eval_scores = evaluate_agent(self.agent, guide, test_dataset, num_threads=num_threads, num_eval_times=num_eval_samples)
            self.logger.log('Test score', eval_scores, 0, color='green')
            self.logger.log('Total samples', self.total_samples, 0, color='cyan')

        for i in range(num_epochs):
            self.update_buffer_scores()
            self.print_buffer_statistics()
            # select the candidate with the highest UCB score
            selected_candidate_entry = max(self.buffer, key=lambda x: x['ucb_score'])
            self.optimizer.update(selected_candidate_entry['params'])
            
            # sample a minibatch from the train dataset
            xs, infos = self._sample_minibatch(train_dataset, batch_size)
            
            forward = batch_run(max_workers=num_threads, description=f"Forward pass (batch size: {len(xs)})")(self.forward)
            outputs = forward(self.agent, xs, guide, infos)

            # Update the agent
            score, new_update_dict = self.update(outputs, verbose=verbose, num_threads=num_threads, **kwargs)
            self.total_samples += len(xs)
            # update the buffer statistics of the current candidate
            selected_candidate_entry['score_sum'] += score*len(xs)
            selected_candidate_entry['eval_count'] += len(xs)

            # evaluate the new candidate on the same minibatch
            mini_batch = {'inputs': xs, 'infos': infos}
            self.optimizer.update(new_update_dict)
            new_score = evaluate_agent(self.agent, guide, mini_batch, num_threads=num_threads, num_eval_times=1)
            self.total_samples += len(xs)
            # log raw scores
            self.logger.log('Train score', score, i+1, color='cyan')
            self.logger.log('New candidate raw score', new_score, i+1, color='green')
            
            # update the buffer statistics of the new candidate
            if self.enable_control_variate:
                new_score = new_score - score + selected_candidate_entry['mean_score']
                # clip the new score to be between 0 and 1
                # new_score = np.clip(new_score, 0, 1)
                self.logger.log('New candidate controlled score', new_score, i+1, color='yellow')

            new_candidate_entry = {
                'params': new_update_dict,
                'score_sum': new_score*len(xs),
                'eval_count': len(xs)
            }
            self.buffer.append(new_candidate_entry)
            
            if (i+1) % eval_frequency == 0:
                self.update_buffer_scores()
                self.print_buffer_statistics()
                # select the candidate with the highest UCB score
                best_candidate_entry = max(self.buffer, key=lambda x: x['mean_score'])
                self.optimizer.update(best_candidate_entry['params'])
                # evaluate the best candidate on the test dataset
                test_score = evaluate_agent(self.agent, guide, test_dataset, num_threads=num_threads, num_eval_times=num_eval_samples)
                self.logger.log('Selected candidate mean score', best_candidate_entry['mean_score'], i+1, color='blue')
                self.logger.log('Test score', test_score, i+1, color='green')
                self.logger.log('Total samples', self.total_samples, i+1, color='cyan')
                
        return 