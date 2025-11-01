import numpy as np
import copy
import sys
import os
from typing import Union, List, Tuple, Dict, Any, Optional
from opto.features.priority_search.search_template import Samples
# import pretrained regressors
from opto.optimizers.utils import print_color
from opto.trainer.utils import safe_mean

from opto.features.priority_search.priority_search import PrioritySearch, ModuleCandidate, HeapMemory
from opto.features.priority_search.priority_search_with_regressor import PrioritySearch_with_Regressor
import heapq

def calculate_distance_to_memory(memory, new_candidate):
        """For a new candidate, calculate the distance to the current memory. That's the least L2 distance to any candidate in the memory.
        
        To use this funciton in PrioritySearch, set memory to be self.memory.memory.
        """
        assert new_candidate.num_rollouts == 0, "New candidates should have no rollouts."
        # assert new candidate and all candidates in the memory have the  embedding.
        assert hasattr(new_candidate, 'embedding') and all(hasattr(candidate, 'embedding') for _, candidate in memory), "All candidates should have the embedding attribute."
        # calculate the distance to the current memory. That's the least L2 distance to any candidate in the memory.
        min_distance = float('inf')
        for _, candidate in memory:
            distance = np.linalg.norm(np.array(new_candidate.embedding) - np.array(candidate.embedding))
            if distance < min_distance:
                min_distance = distance
        return min_distance

class ExhaustedPrioritySearch_v2(PrioritySearch_with_Regressor):
    """
    Use the original Priority Search framework.
    We do not need a depth attribute for each node (no Tree structure for the memory).
    Step 0: Pull the original candidate many times to generate many children.
    Step 1:
        At each iteration,
            a. Exploration: select multiple candidates based on predicted scores (and/or bonus) to accelerate information collection process.
            b. Ensuring nodes exhausted: select multiple candidates (with the highest scores, unexplored), pull one more time to cover its children (exhausted).

        Only keep children which are not in the current eps-cover. Others into a temporary buffer.
    Step 2: 
        If all nodes in the current memory are explored (num_rollouts>=k), reset eps and include more nodes from the buffer to the memory.
    """
    def __init__(self,
                 max_depth: int = 100,
                 epsilon: float = 0.5,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.max_depth = max_depth
        self.epsilon = epsilon # epsilon-greedy exploration parameter
        self.exhausted_pull_times = 10 # number of times to pull the exhausted candidate.
        self.buffer = [] # buffer to store the candidates that are not added to the memory.

    def train(self,num_batches: int = 1, *args, **kwargs):
        assert num_batches == 1, "ExhaustedPrioritySearch_v2 only supports num_batches of 1."
        print_color(f"Training with ExhaustedPrioritySearch_v2 algorithm...", "green")
        print_color(f"For the exhausted candidate, generate {self.exhausted_pull_times} children. Initial epsilon = {self.epsilon}", "green")

        super().train(num_batches=num_batches, *args, **kwargs)

    def update(self,
               samples: Union[Samples, None] = None,
               verbose: bool = False,
               **kwargs): #-> Tuple[Dict[ParameterNode, Any], List[trace.Module], Dict[str, Any]]:
        """ Update the agent using the collected samples. Added some logging information for the exhausted search tree.
        """
        print_color(f"Updating the agent using the collected samples... Iteration:{self.n_iters} ",  "blue")
        # samples is None in the first iteration
        if samples is not None:
            # 0. Update the regressor right after collecting samples. We need to first update popped candidates with their new samples. After this update, all candidates in the memory have new predicted scores and the memory is sorted by the predicted scores. All exploration candidates get predicted scores, but they are not added to the memory yet.
            self.update_regressor_with_samples(samples)
            # 1. Propose new parameters based on running LLM optimizers on the collected samples. It doesn't matter whether we do this before or after updating the regressor.
            candidates = self.propose(samples, verbose=verbose, **kwargs)  # List of ModuleCandidates
            self.regressor.add_embeddings_to_candidates(candidates)
            # 2. Validate the proposed parameters
            validate_results = self.validate(candidates, samples, verbose=verbose, **kwargs)  # this updates the priority queue
            # 3. Update the priority queue with the validation results
            self.update_memory(validate_results, verbose=verbose, **kwargs)  # samples are provided here in case candidates do not capture full information
            self.update_memory_with_regressor(verbose=verbose, **kwargs)
            if self.n_iters % self.log_frequency == 0:
                self.logger.log('SearchTree/len_memory', len(self.memory), self.n_iters, color='blue')
                self.logger.log('SearchTree/len_buffer', len(self.buffer), self.n_iters, color='blue')
                self.logger.log('SearchTree/len_unexplored_nodes',len([candidate for _, candidate in self.memory.memory if candidate.num_rollouts < self.exhausted_pull_times * self.batch_size * self.num_batches ]), self.n_iters, color='blue')
                # highest_mean_score = 
                self.logger.log('SearchTree/highest_mean_score',max([candidate.mean_score() for _, candidate in self.memory.memory if candidate.mean_score() is not None]) , self.n_iters, color='blue')
                self.logger.log('SearchTree/highest_predicted_score',max([candidate.predicted_score for _, candidate in self.memory.memory]) , self.n_iters, color='blue')
                # log epsilon
                self.logger.log('SearchTree/epsilon', self.epsilon, self.n_iters, color='blue')
        else:  # The first iteration.
            self.default_batch_size, self.default_num_batches = self.get_sampler_batch_size()
            self.memory.push(self.max_score, ModuleCandidate(self.agent, optimizer=self.optimizer))  # Push the base agent as the first candidate (This gives the initialization of the priority queue)
            self.update_memory_with_regressor(verbose=verbose, **kwargs)
        self.print_memory_stats()
        # TODO Log information about the update
        info_log = {
            'n_iters': self.n_iters,  # number of iterations
            'short_term_memory_size': len(self.short_term_memory),  # size of the short-term memory
            'long_term_memory_size': len(self.long_term_memory),  # size of the long-term memory
            'using_short_term_memory': self.memory is self.short_term_memory,  # whether the current memory is the short-term memory
            'using_long_term_memory': self.memory is self.long_term_memory,  # whether the current memory is the long-term memory
        }
        total_samples = sum([candidate.num_rollouts for _, candidate in self.short_term_memory]) + \
                        sum([candidate.num_rollouts for _, candidate in self.long_term_memory])
        info_log.update({'total_samples': total_samples})
        # 4. Explore and exploit the priority queue
        self._best_candidate, self._best_candidate_priority, info_exploit = self.exploit(verbose=verbose, **kwargs)  # get the best candidate (ModuleCandidate) from the priority queue
        self._exploration_candidates, self._exploration_candidates_priority, info_explore = self.explore(verbose=verbose, **kwargs)  # List of ModuleCandidates
        info_log.update(info_exploit)  # add the info from the exploit step
        info_log.update(info_explore)  # add the info from the explore step
        return self._best_candidate.update_dict, [c.get_module() for c in self._exploration_candidates], info_log

    def update_regressor_with_samples(self,samples: Samples):
        """ Update the regressor with the samples. 
        This function adds new samples to the exploration candidates, then updates the regressor and all predicted scores. It doesn't add new candidates to the memory.
        """
        matched_exploration_candidates_and_samples = self.match_candidates_and_samples(self._exploration_candidates, samples.samples)
        exploration_results = {}  # dict of ModuleCandidate id: (ModuleCandidate, list of rollouts)
        for c, rollouts in matched_exploration_candidates_and_samples.items():  # rollouts is a list of BatchRollouts
            exploration_results[c] = [ r for rr in rollouts for r in rr.to_list()]
        for candidate, rollouts in exploration_results.items():
            candidate.add_rollouts(rollouts)  # add the rollouts to the candidate
        exploration_memory = [(0, candidate) for candidate in self._exploration_candidates]
        print_color(f"len(exploration_results.keys()): {len(exploration_results.keys())}", "red")
        try:
            assert len(exploration_results.keys()) == self.num_candidates or len(exploration_results.keys()) == 1 or len(self.memory) == 0, "The number of exploration results should be equal to the number of exploration candidates."
        except AssertionError as e:
            print_color(f"AssertionError: {e}", "red")
            print_color(f"len(exploration_results.keys()): {len(exploration_results.keys())}", "red")
            print_color(f"self.num_candidates: {self.num_candidates}", "red")
            raise e
        
        # print_color(f'len of long_term_memory: {len(self.long_term_memory.memory)},  len of exploration_memory: {len(exploration_memory)}',  "red")
        self.regressor.update(self.long_term_memory.memory+self.short_term_memory.memory+exploration_memory)
        self.children_regressor.update(self.long_term_memory.memory+self.short_term_memory.memory+exploration_memory)
        # update the predicted scores for all candidates with data
        predicted_scores = self.regressor.predict_scores(self.long_term_memory.memory+self.short_term_memory.memory+exploration_memory)
        self.children_regressor.predict_scores(self.long_term_memory.memory+self.short_term_memory.memory+exploration_memory)
        self.highest_predicted_score = max(predicted_scores)

        self.regressor.predict_scores([(0, self.base_agent_ModuleCandidate)])
        self.children_regressor.predict_scores([(0, self.base_agent_ModuleCandidate)])
        self.base_agent_predicted_score = self.base_agent_ModuleCandidate.predicted_score
        # heapify the memory
        self.heapify_memory(self.long_term_memory.memory)
        self.heapify_memory(self.short_term_memory.memory)
    

    def validate(self,
                 candidates: List[ModuleCandidate],
                 samples: Samples,
                 verbose: bool = False,
                 **kwargs):
        """ 
        Override the validate method. 
        In this version, if use_validation is False, we can only use training data to update arm statistics. No validation is performed.
        If use_validation is True, we use the validation set to update arm statistics. The same as the parent class.

        Updated on Oct 14, 2025: I added exploration samples to exploration candidates before this function. So only care about validation samples here. The current logic is, all exploration candidates have already been added exploration samples. Here we may have exploration candidates, new candidates with probably validation samples.
        """
        print("--- Validating candidates...") if verbose else None
        assert isinstance(samples, Samples), "samples must be an instance of Samples."
        exploration_candidates = self._exploration_candidates  # exploration candidates from the previous iteration
        assert self._exploration_candidates is not None, "exploration_candidates must be set before calling validate."

        # The current batch of samples can be used to validate the exploration candidates
        # validate_samples = copy.copy(samples)
        validate_samples = Samples([], {'inputs': [], 'infos': []})
        if self.use_validation:
        # Validate newly proposed candidates
            use_prev_batch = self.use_prev_batch  # when True, self.validate_sampler == self.train_sampler, and the current batch is used for validation
            candidate_agents = [c.get_module() for c in candidates]  # get the modules from the candidates
            validate_samples.add_samples(Samples(*self.validate_sampler.sample(candidate_agents,
                                                                                                use_prev_batch=use_prev_batch,
                                                                                                description_prefix='Validating newly proposed candidates: ')))  # list of BatchRollout objects

            if self.validate_exploration_candidates:
                if not use_prev_batch:   # validate the exploration candidates that collected the samples as well
                    # validate the agents in the validate_dataset
                    exploration_agents = [c.get_module() for c in exploration_candidates]  # get the modules from the exploration candidates
                    exploration_samples = Samples(*self.validate_sampler.sample(exploration_agents,
                                                description_prefix='Validating exploration candidates: '))  # sample the exploration agents
                    validate_samples.add_samples(exploration_samples)  # append the exploration samples to the validate_samples 
        # Here we should set self._enforce_using_data_collecting_candidates to False
        matched_candidates_and_samples = self.match_candidates_and_samples(exploration_candidates+candidates, validate_samples.samples)
        
        # # Append new candidates with out rollouts to matched_candidates_and_samples
        # matched_candidates_and_samples.update({c: [] for c in candidates })
        results = {}  # dict of ModuleCandidate id: (ModuleCandidate, list of rollouts)
        for c, rollouts in matched_candidates_and_samples.items():  # rollouts is a list of BatchRollouts
            results[c] = [ r for rr in rollouts for r in rr.to_list()]  # we only need the list of dicts

        return results
    
    def update_memory(self, validate_results, verbose: bool = False, **kwargs):# update based on eps cover
        """ At each update_memory method, first add all old candidates (num_rollouts > 0) to the memory. Them, for each new candidate, calculate the distance to the memory, if larger than self.epsilon, push it to the memory.
        """
        print("--- Updating memory with validation results...") if verbose else None
        new_candidates= [] # new candidates will be added here.
        for candidate, rollouts in validate_results.items():
            candidate.add_rollouts(rollouts)  # add the rollouts to the
            if candidate.num_rollouts > 0: # old candidate
                heapq.heappush(self.memory.memory, (0, candidate))
                # self.memory.push(self.max_depth+1, candidate) # after explored, old candidates get a very large priority, make it impossible to be popped again.
            else: # new candidate
                new_candidates.append(candidate)
        count_new_candidates = 0
        count_buffer = 0
        for new_candidate in new_candidates:
            distance = calculate_distance_to_memory(self.memory.memory, new_candidate)
            if distance > self.epsilon: # only collect new candidates those are not in the epsilon-neighborhood of the memory.
                count_new_candidates += 1
                heapq.heappush(self.memory.memory, (0, new_candidate))
            else:
                self.buffer.append(new_candidate)
                count_buffer += 1
        print_color(f"Proposed {len(new_candidates)} new candidates, {count_new_candidates} of them are added to the memory, {count_buffer} of them are added to the temporary buffer. Buffer size: {len(self.buffer)}.", "green")


    def update_memory_with_regressor(self, verbose: bool = False, **kwargs):
        """ 
        Update the priority queue with the regressor results.
        This function does not add new candidates to the memory. It only updates the predicted scores of the existing candidates. Then reorder the memory according to the predicted scores.
        """
        print("--- Updating memory with regressor results...") if verbose else None
        # Use all data to update the regressor
        # only update the memory when self.use_validation is True. Otherwise we have done this before.
        if self.use_validation or self.n_iters == 0:
            self.regressor.update(self.long_term_memory.memory+self.short_term_memory.memory)
            # self.children_regressor.update(self.long_term_memory.memory+self.short_term_memory.memory)
            # Always keep track of the predicted score of the base agent. Ideally this number should converge to the true score of the base agent, when we have more and more data.
            self.regressor.predict_scores([(0, self.base_agent_ModuleCandidate)])
            # self.children_regressor.predict_scores([(0, self.base_agent_ModuleCandidate)])
            self.base_agent_predicted_score = self.base_agent_ModuleCandidate.predicted_score
        # Predict the scores for the long-term memory and the short-term memory
        self.regressor.predict_scores(self.long_term_memory.memory)
        # self.children_regressor.predict_scores(self.long_term_memory.memory)
        self.regressor.predict_scores(self.short_term_memory.memory)
        # self.children_regressor.predict_scores(self.short_term_memory.memory)
        # update the highest predicted score
        self.highest_predicted_score = max(0, max([candidate.predicted_score for _, candidate in self.long_term_memory.memory+self.short_term_memory.memory]))
        # Reorder both long_term_memory and short_term_memory according to the predicted scores
        self.heapify_memory(self.long_term_memory.memory)
        self.heapify_memory(self.short_term_memory.memory)
        
    def print_memory_stats(self):
        # For debugging, print all candidates: number, mean_score(), num_rollouts, predicted_score. It is better to see an increasing trend in the predicted scores.
        print("--- Printing memory stats...")
        print("Long-term memory:")
        # If len(self.long_term_memory.memory)>40, only print the first 20 and the last 20 candidates
        for i, (_, candidate) in enumerate(self.long_term_memory.memory):
            if len(self.long_term_memory.memory) <= 40 or i < 20 or i >= len(self.long_term_memory.memory) - 20:
                mean_score = candidate.mean_score()
                std = candidate.standard_deviation()
                mean_score_str = f"{mean_score:.4g}" if mean_score is not None else "None"
                std_str = f"{std:.4g}" if std is not None else "None"
                print(f"Candidate {i}, Mean Score: {mean_score_str}, Std: {std_str}, Num Rollouts: {candidate.num_rollouts}, Predicted Score: {candidate.predicted_score}")
                # def get_parameter_text(candidate):
                #     """Get the parameter text for a ModuleCandidate."""
                #     if not candidate.update_dict:
                #         return "base_module_parameters"
                #     # Convert parameter nodes to readable names for deterministic embedding
                #     params_with_names = {k.py_name: v for k, v in candidate.update_dict.items()}
                #     return str(params_with_names)
                # print(f"Parameters: {get_parameter_text(candidate)}")
        # print("Short-term memory:")
        # for i, (neg_predicted_score, candidate) in enumerate(self.short_term_memory.memory):
        #     print(f"Candidate {i}, Mean Score: {candidate.mean_score()}, Num Rollouts: {candidate.num_rollouts}, Predicted Score: {-neg_predicted_score}")

    def compute_exploitation_priority(self, candidate) -> float: # choose one with data
        """ Compute the priority for the candidate based on the predicted score. """
        if not isinstance(candidate, ModuleCandidate):
            raise TypeError("candidate must be an instance of ModuleCandidate.")
        # The generalization ability of the regressor is not good enough, so we won't pick unexplored candidates to exploit.
        if candidate.mean_score() is None:
            return 0.0
        return candidate.predicted_score 
    
    def reset_memory(self, factor: float = 0.9): # shrink eps to include more cands
        """
        Reduce the value of self.epsilon by a factor. Then update the epsilon-cover memory using nodes in buffer.
        """
        assert len(self.buffer) > 0, "Buffer is empty. Cannot reset the memory with the temporary buffer."
        print_color(f"All candidates have been explored. Resetting the memory.  Epsilon changes from {self.epsilon} to {self.epsilon*factor}.", "green")

        self.epsilon *= factor
        # Use the updated regressor to predict the scores of the buffer.
        batch = [(0,candidate) for candidate in self.buffer]
        self.regressor.predict_scores(batch)
        # sort the buffer by the predicted scores
        self.buffer.sort(key=lambda x: x.predicted_score, reverse=True)
        count_added = 0
        for candidate in self.buffer:
            distance = calculate_distance_to_memory(self.memory.memory, candidate)
            if distance > self.epsilon:
                heapq.heappush(self.memory.memory, (-candidate.predicted_score, candidate))
                self.buffer.remove(candidate)
                count_added += 1
        
        
        print_color(f"Added {count_added} candidates to the memory. Buffer size: {len(self.buffer)}.", "green")
        if count_added > 0:
            self.heapify_memory(self.memory.memory)
            self.print_memory_stats()

    def _fresh_candidates(self):
        """
        List of candidates in the memory, that have not been exhaustedly pulled.
        """
        # Pulling one candidate one time means sampling it on one minibatch of data.
        return [candidate for _, candidate in self.memory.memory if candidate.num_rollouts < self.exhausted_pull_times * self.batch_size]

    def explore(self, verbose: bool = False, **kwargs): # Exploration+Exhaustion method
        """ 
        a. Select one candidate (with the highest score, unexplored), pull k (we are going to pull each arm at least this number) times to cover its children (exhausted).
        b. Select multiple candidates based on predicted scores (and/or bonus) to accelerate information collection process.
        """
        print_color(f"Using exhausted priority search v2 exploration method...", "green")

        while len(self._fresh_candidates()) == 0:
            # All candidates have been exhaustedly pulled. Reset the memory.
            self.reset_memory()
        
        top_candidates = []
        priorities = []

        # Step a: Exploration: select multiple candidates based on predicted scores (and/or bonus) to accelerate information collection process.

        num_exploration_candidates = self.num_candidates//2
        num_exhaustion_candidates = self.num_candidates - num_exploration_candidates

        while len(top_candidates) < num_exploration_candidates and len(self.memory) > 0:
            neg_priority, candidate = self.memory.pop()  # pop the top candidate from the priority queue
            priority = - neg_priority  # remember that we stored negative scores in the priority queue
            priorities.append(priority)  # store the priority of the candidate
            top_candidates.append(candidate)  # add the candidate to the top candidates


        # Step b: Exhaustion: select multiple candidates (with the highest scores, unexplored), pull one more time to cover its children (exhausted).

        unexplored_candidates = self._fresh_candidates()
        # the candidate to be exhausted is the one with the highest predicted score among the unexplored candidates.
        exhausted_candidates = unexplored_candidates[:num_exhaustion_candidates]
        
        initial_length = len(self.memory.memory) # for defensive programming
        for candidate in exhausted_candidates:
            assert (-candidate.predicted_score, candidate) in self.memory.memory, "The candidate should be in the memory."
            self.memory.memory.remove((-candidate.predicted_score, candidate))
            priorities.append(candidate.predicted_score)

        assert len(self.memory.memory) == initial_length - len(exhausted_candidates), "The memory should have one less candidate after exhausting one."

        top_candidates.extend(exhausted_candidates)
        
        self.heapify_memory(self.memory.memory)

        # assert no duplicates in top_candidates
        assert len(top_candidates) == len(set(top_candidates)) == len(priorities), "There are duplicates in the top candidates, or the number of top candidates is not equal to the number of priorities."

        

       
        mean_scores = [c.mean_score() for c in top_candidates]
        mean_scores = [s for s in mean_scores if s is not None]  # filter out None scores
        info_dict = {
            'num_exploration_candidates': len(top_candidates),
            'exploration_candidates_mean_priority': safe_mean(priorities),  # list of priorities of the exploration candidates
            'exploration_candidates_mean_score': safe_mean(mean_scores),  # list of mean scores of the exploration candidates
            'exploration_candidates_average_num_rollouts': safe_mean([c.num_rollouts for c in top_candidates]),
        }

        return top_candidates, priorities, info_dict

class ExhaustedPrioritySearch(PrioritySearch_with_Regressor):
    """
    A search algorithm that uses a priority queue to explore the parameter space and propose new candidates.
    """

    def __init__(self,
                 max_depth: int = 100,
                 epsilon: float = 0.5,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.max_depth = max_depth
        self.epsilon = epsilon # epsilon-greedy exploration parameter
        self.buffer = [] # buffer to store the candidates that are not added to the memory.

    def train(self,num_candidates: int = 1,batch_size: int = 2,num_batches: int = 10, *args, **kwargs):
        assert num_candidates == 1, "ExhaustedPrioritySearch only supports one candidate at a time."
        print_color(f"ExhaustedPrioritySearch: num_candidates = {num_candidates}, batch_size = {batch_size}, num_batches = {num_batches}", "green")
        print_color(f"For each candidate, generate {num_batches} children. epsilon = {self.epsilon}", "green")

        super().train(num_candidates=num_candidates, batch_size=batch_size, num_batches=num_batches, *args, **kwargs)
    
    def update(self,
               samples: Union[Samples, None] = None,
               verbose: bool = False,
               **kwargs): #-> Tuple[Dict[ParameterNode, Any], List[trace.Module], Dict[str, Any]]:
        """ Update the agent using the collected samples.
        """

        # samples is None in the first iteration
        if samples is not None:
            # 1. Propose new parameters based on running LLM optimizers on the collected samples
            candidates = self.propose(samples, verbose=verbose, **kwargs)  # List of ModuleCandidates
            # add embedding to the candidates asynchronously
            self.regressor.add_embeddings_to_candidates(candidates)
            # 2. Validate the proposed parameters
            validate_results = self.validate(candidates, samples, verbose=verbose, **kwargs)  # this updates the priority queue
            # 3. Update the priority queue with the validation results
            self.update_memory(validate_results, verbose=verbose, **kwargs)  # samples are provided here in case candidates do not capture full information
            # Log some statistics about the search tree.
            if self.n_iters % self.log_frequency == 0:
                self.logger.log('SearchTree/depth', self.depth+1, self.n_iters, color='blue')
                self.logger.log('SearchTree/num_candidates', len(self.memory), self.n_iters, color='blue')
                # highest_mean_score = 
                self.logger.log('SearchTree/highest_mean_score',max([candidate.mean_score() for _, candidate in self.memory.memory if candidate.mean_score() is not None]) , self.n_iters, color='blue')
                # log epsilon
                self.logger.log('SearchTree/epsilon', self.epsilon, self.n_iters, color='blue')
        else:  # The first iteration.
            max_mem_size = self.memory.size if self.memory.size is not None else float('inf')
            while len(self.memory) < min(max_mem_size, self.num_candidates):
                original_candidate = ModuleCandidate(self.agent, optimizer=self.optimizer)
                self.regressor.add_embeddings_to_candidates([original_candidate])
                original_candidate.depth = 1
                heapq.heappush(self.memory.memory, (original_candidate.depth, original_candidate))
                # self.memory.push(original_candidate.depth, original_candidate)  # Push the base agent as the first candidate (This gives the initialization of the priority queue)
        self.regressor.update(self.memory.memory)
        predicted_scores = self.regressor.predict_scores(self.memory.memory)

        if self.n_iters % self.log_frequency == 0:
            # log the highest predicted score
            self.logger.log('SearchTree/highest_predicted_score',max(predicted_scores), self.n_iters, color='blue')

        self.print_memory_stats()

        

        # Log information about the update
        info_log = {
            'n_iters': self.n_iters,  # number of iterations
            'short_term_memory_size': len(self.short_term_memory),  # size of the short-term memory
            'long_term_memory_size': len(self.long_term_memory),  # size of the long-term memory
            'using_short_term_memory': self.memory is self.short_term_memory,  # whether the current memory is the short-term memory
            'using_long_term_memory': self.memory is self.long_term_memory,  # whether the current memory is the long-term memory
        }
        # Due to some api errors, failed sampling process is not counted in the total samples. If we want to count it, we can calculate total_samples from the parameters of the trainer. In this way, different runs could have same number of total samples, not influenced by the randomness.
        total_samples = sum([candidate.num_rollouts for _, candidate in self.short_term_memory]) + \
                        sum([candidate.num_rollouts for _, candidate in self.long_term_memory])
        info_log.update({'total_samples': total_samples})

        # 4. Explore and exploit the priority queue
        self._best_candidate, self._best_candidate_priority, info_exploit = self.exploit(verbose=verbose, **kwargs)  # get the best candidate (ModuleCandidate) from the priority queue
        self._exploration_candidates, self._exploration_candidates_priority, info_explore = self.explore(verbose=verbose, **kwargs)  # List of ModuleCandidates
        

        info_log.update(info_exploit)  # add the info from the exploit step
        info_log.update(info_explore)  # add the info from the explore step
        return self._best_candidate.update_dict, [c.get_module() for c in self._exploration_candidates], info_log

    
    # def heapify_memory(self,memory):
    #     """ Heapify the memory, based on the predicted scores. Input could be something like self.long_term_memory.memory."""
    #     long_term_candidates_with_scores = [(-candidate.predicted_score, candidate) for _, candidate in memory]
    #     memory[:] = long_term_candidates_with_scores  # Slice assignment modifies original
    #     heapq.heapify(memory)

    def validate(self,
                 candidates: List[ModuleCandidate],
                 samples: Samples,
                 verbose: bool = False,
                 **kwargs):
        # TODO: delete the validation
        """ Validate the proposed candidate parameters
        Args:
            candidates (list of ModuleCandidate): A list of ModuleCandidate objects representing the proposed parameters.
            samples (list of dict, optional): A list of samples collected in the current iteration. Defaults to None.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            **kwargs: Additional keyword arguments that may be used by the implementation.
        Returns:
            results (dict): A dictionary where the keys are ids of ModuleCandidate objects and the values are ModuleCandidate and lists of rollouts (list of dicts) containing the module, x, info, target, score, feedback.
        """
        print("--- Validating candidates...") if verbose else None
        assert isinstance(samples, Samples), "samples must be an instance of Samples."
        exploration_candidates = self._exploration_candidates  # exploration candidates from the previous iteration
        assert self._exploration_candidates is not None, "exploration_candidates must be set before calling validate."

        # The current batch of samples can be used to validate the exploration candidates
        validate_samples = copy.copy(samples)

        matched_candidates_and_samples = self.match_candidates_and_samples(exploration_candidates+candidates, validate_samples.samples)
        results = {}  # dict of ModuleCandidate id: (ModuleCandidate, list of rollouts)
        for c, rollouts in matched_candidates_and_samples.items():  # rollouts is a list of BatchRollouts
            results[c] = [ r for rr in rollouts for r in rr.to_list()]  # we only need the list of dicts

        return results

    def update_memory(self, validate_results, verbose: bool = False, **kwargs):
        """ At each update_memory method, first add all old candidates (num_rollouts > 0) to the memory. Them, for each new candidate, calculate the distance to the memory, if larger than self.epsilon, push it to the memory.
        """
        print("--- Updating memory with validation results...") if verbose else None
        new_candidates= [] # new candidates will be added here.
        for candidate, rollouts in validate_results.items():
            candidate.add_rollouts(rollouts)  # add the rollouts to the
            if candidate.num_rollouts > 0: # old candidate
                heapq.heappush(self.memory.memory, (self.max_depth+1, candidate))
                # self.memory.push(self.max_depth+1, candidate) # after explored, old candidates get a very large priority, make it impossible to be popped again.
            else: # new candidate
                new_candidates.append(candidate)
        count_new_candidates = 0
        count_buffer = 0
        for new_candidate in new_candidates:
            distance = calculate_distance_to_memory(self.memory.memory, new_candidate)
            new_candidate.depth = self.depth + 1 # self.depth is the depth of the last popped candidate.
            if distance > self.epsilon: # only collect new candidates those are not in the epsilon-neighborhood of the memory.
                count_new_candidates += 1
                heapq.heappush(self.memory.memory, (self.depth+1, new_candidate))
                # self.memory.push(self.depth+1, new_candidate)
            else:
                self.buffer.append(new_candidate)
                count_buffer += 1
        print_color(f"Proposed {len(new_candidates)} new candidates, {count_new_candidates} of them are added to the memory, {count_buffer} of them are added to the temporary buffer. Buffer size: {len(self.buffer)}.", "green")
            
    def print_memory_stats(self):
        # For debugging, print all candidates: number, mean_score(), num_rollouts, predicted_score. It is better to see an increasing trend in the predicted scores.
        print("--- Printing memory stats...")
        try:
            assert all([(priority == candidate.depth or priority == self.max_depth+1) for priority,candidate in self.memory.memory]), "Priority should be the depth or self.max_depth+1."
        except AssertionError as e:
            print_color(f"Error: {e}", "red")
            print_color(f"Memory: {self.memory.memory}", "red")
            raise e
        
        print("Long-term memory:")
        for i, (priority, candidate) in enumerate(self.long_term_memory.memory):
            mean_score = candidate.mean_score()
            mean_score_str = f"{mean_score:.4g}" if mean_score is not None else "None"
            print(f" Priority: {priority}, Depth: {candidate.depth}, Candidate {i}, Mean Score: {mean_score_str}, Num Rollouts: {candidate.num_rollouts}, Predicted Score: {candidate.predicted_score}")

    def compute_exploitation_priority(self, candidate) -> float:
        """ Compute the priority for the candidate based on the predicted score. """
        if not isinstance(candidate, ModuleCandidate):
            raise TypeError("candidate must be an instance of ModuleCandidate.")
        # The generalization ability of the regressor is not good enough, so we won't pick unexplored candidates to exploit.
        if candidate.mean_score() is None:
            return 0.0
        return candidate.predicted_score  

    def exploit(self, verbose: bool = False, **kwargs) -> Tuple[ModuleCandidate, Dict[str, Any]]:
        """ Use positive priority.
        """
        print("--- Exploiting the best candidate...") if verbose else None
        if not self.memory:
            raise ValueError("The priority queue is empty. Cannot exploit.")
        priority, best_candidate = self.memory.best(self.compute_exploitation_priority)  # (priority, candidate)
        return best_candidate, priority, {
            'best_candidate_priority': priority,  # remember that we stored negative scores in the priority queue
            'best_candidate_mean_score': best_candidate.mean_score(),  # mean score of the candidate's rollouts
            'best_candidate_num_rollouts': best_candidate.num_rollouts,  # number of rollouts of the candidate
        }

    # def reset_memory(self):
    #     """ Reset the priority queue. Initialize each candidate with the priority to be the depth.
    #     """
    #     # Initialize all priorities again.
    #     self.memory.memory = [(candidate.depth, candidate) for _, candidate in self.memory.memory]
    #     heapq.heapify(self.memory.memory)

    def reset_memory(self, factor: float = 0.9):
        """
        Reduce the value of self.epsilon by a factor. Then update the epsilon-cover memory using nodes in buffer.
        """
        assert len(self.buffer) > 0, "Buffer is empty. Cannot reset the memory with the temporary buffer."
        self.epsilon *= factor
        print_color(f"All candidates have been explored. Resetting the memory. New epsilon: {self.epsilon}.", "green")
        batch = [(0,candidate) for candidate in self.buffer]
        self.regressor.predict_scores(batch)
        # sort the buffer by the predicted scores
        self.buffer.sort(key=lambda x: x.predicted_score, reverse=True)
        count_added = 0
        for candidate in self.buffer:
            distance = calculate_distance_to_memory(self.memory.memory, candidate)
            if distance > self.epsilon:
                heapq.heappush(self.memory.memory, (candidate.depth, candidate))
                self.buffer.remove(candidate)
                count_added += 1
        
        
        print_color(f"Added {count_added} candidates to the memory. Buffer size: {len(self.buffer)}.", "green")
        if count_added > 0:
            self.print_memory_stats()
        

            
    def explore(self, verbose: bool = False, **kwargs):
       
        # pop top self.num_candidates candidates from the priority queue
        top_candidates = [] 
        priorities = [] 
        
        while min([priority for priority, _ in self.memory.memory]) == self.max_depth+1:            
            # check if all candidates have been explored
            self.reset_memory()
        priority, candidate = self.memory.pop()  # pop the top candidate from the priority queue
        priorities.append(priority)  # store the priority of the candidate
        top_candidates.append(candidate)  # add the candidate to the top candidates
        # only one candidate is popped from the memory, so we can get the depth from the candidate. This is used for adding depth attribute to new candidates.
        self.depth = candidate.depth
        
        # NOTE some top_candidates can be duplicates
        mean_scores = [c.mean_score() for c in top_candidates]
        mean_scores = [s for s in mean_scores if s is not None]  # filter out None scores
        info_dict = {
            'num_exploration_candidates': len(top_candidates),
            'exploration_candidates_mean_priority': safe_mean(priorities),  # list of priorities of the exploration candidates
            'exploration_candidates_mean_score': safe_mean(mean_scores),  # list of mean scores of the exploration candidates
            'exploration_candidates_average_num_rollouts': safe_mean([c.num_rollouts for c in top_candidates]),
        }

        return top_candidates, priorities, info_dict

class ExhaustedPrioritySearch_highscore(ExhaustedPrioritySearch):
    """Choose the candidate with the highest predicted score to explore, rather than the one with the least depth."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.epsilon = 0.1

    def update(self,
               samples: Union[Samples, None] = None,
               verbose: bool = False,
               **kwargs): #-> Tuple[Dict[ParameterNode, Any], List[trace.Module], Dict[str, Any]]:
        """ Update the agent using the collected samples.
        """

        # samples is None in the first iteration
        if samples is not None:
            # 1. Propose new parameters based on running LLM optimizers on the collected samples
            candidates = self.propose(samples, verbose=verbose, **kwargs)  # List of ModuleCandidates
            # add embedding to the candidates asynchronously
            self.regressor.add_embeddings_to_candidates(candidates)
            self.regressor.predict_scores([(0,candidate) for candidate in candidates])
            # 2. Validate the proposed parameters
            validate_results = self.validate(candidates, samples, verbose=verbose, **kwargs)  # this updates the priority queue
            # 3. Update the priority queue with the validation results
            self.update_memory(validate_results, verbose=verbose, **kwargs)  # samples are provided here in case candidates do not capture full information
            # Log some statistics about the search tree.
            if self.n_iters % self.log_frequency == 0:
                self.logger.log('SearchTree/current_depth', self.depth+1, self.n_iters, color='blue')
                self.logger.log('SearchTree/max_depth', max([candidate.depth for _, candidate in self.memory.memory]), self.n_iters, color='blue')
                self.logger.log('SearchTree/num_candidates', len(self.memory), self.n_iters, color='blue')
                # highest_mean_score = 
                self.logger.log('SearchTree/highest_mean_score',max([candidate.mean_score() for _, candidate in self.memory.memory if candidate.mean_score() is not None]) , self.n_iters, color='blue')
                # log epsilon
                self.logger.log('SearchTree/epsilon', self.epsilon, self.n_iters, color='blue')
        else:  # The first iteration.
            max_mem_size = self.memory.size if self.memory.size is not None else float('inf')
            while len(self.memory) < min(max_mem_size, self.num_candidates):
                original_candidate = ModuleCandidate(self.agent, optimizer=self.optimizer)
                self.regressor.add_embeddings_to_candidates([original_candidate])
                self.regressor.predict_scores([(0, original_candidate)])
                original_candidate.depth = 1
                heapq.heappush(self.memory.memory, (-original_candidate.predicted_score, original_candidate))
                # self.memory.push(original_candidate.depth, original_candidate)  # Push the base agent as the first candidate (This gives the initialization of the priority queue)
        self.regressor.update(self.memory.memory)
        predicted_scores = self.regressor.predict_scores(self.memory.memory)
        self.heapify_memory(self.memory.memory)


        if self.n_iters % self.log_frequency == 0:
            # log the highest predicted score
            self.logger.log('SearchTree/highest_predicted_score',max(predicted_scores), self.n_iters, color='blue')

        self.print_memory_stats()

        

        # Log information about the update
        info_log = {
            'n_iters': self.n_iters,  # number of iterations
            'short_term_memory_size': len(self.short_term_memory),  # size of the short-term memory
            'long_term_memory_size': len(self.long_term_memory),  # size of the long-term memory
            'using_short_term_memory': self.memory is self.short_term_memory,  # whether the current memory is the short-term memory
            'using_long_term_memory': self.memory is self.long_term_memory,  # whether the current memory is the long-term memory
        }
        # Due to some api errors, failed sampling process is not counted in the total samples. If we want to count it, we can calculate total_samples from the parameters of the trainer. In this way, different runs could have same number of total samples, not influenced by the randomness.
        total_samples = sum([candidate.num_rollouts for _, candidate in self.short_term_memory]) + \
                        sum([candidate.num_rollouts for _, candidate in self.long_term_memory])
        info_log.update({'total_samples': total_samples})

        # 4. Explore and exploit the priority queue
        self._best_candidate, self._best_candidate_priority, info_exploit = self.exploit(verbose=verbose, **kwargs)  # get the best candidate (ModuleCandidate) from the priority queue
        self._exploration_candidates, self._exploration_candidates_priority, info_explore = self.explore(verbose=verbose, **kwargs)  # List of ModuleCandidates
        

        info_log.update(info_exploit)  # add the info from the exploit step
        info_log.update(info_explore)  # add the info from the explore step
        return self._best_candidate.update_dict, [c.get_module() for c in self._exploration_candidates], info_log

    

    def update_memory(self, validate_results, verbose: bool = False, **kwargs):
        """ At each update_memory method, first add all old candidates (num_rollouts > 0) to the memory. Them, for each new candidate, calculate the distance to the memory, if larger than self.epsilon, push it to the memory.
        """
        print("--- Updating memory with validation results...") if verbose else None
        new_candidates= [] # new candidates will be added here.
        for candidate, rollouts in validate_results.items():
            candidate.add_rollouts(rollouts)  # add the rollouts to the
            if candidate.num_rollouts > 0: # old candidate
                heapq.heappush(self.memory.memory, (-candidate.predicted_score, candidate))
                # self.memory.push(self.max_depth+1, candidate) # after explored, old candidates get a very large priority, make it impossible to be popped again.
            else: # new candidate
                new_candidates.append(candidate)
        count_new_candidates = 0
        count_buffer = 0
        for new_candidate in new_candidates:
            distance = calculate_distance_to_memory(self.memory.memory, new_candidate)
            new_candidate.depth = self.depth + 1 # self.depth is the depth of the last popped candidate.
            if distance > self.epsilon: # only collect new candidates those are not in the epsilon-neighborhood of the memory.
                count_new_candidates += 1
                heapq.heappush(self.memory.memory, (-new_candidate.predicted_score, new_candidate))
                # self.memory.push(self.depth+1, new_candidate)
            else:
                self.buffer.append(new_candidate)
                count_buffer += 1
        print_color(f"Proposed {len(new_candidates)} new candidates, {count_new_candidates} of them are added to the memory, {count_buffer} of them are added to the temporary buffer. Buffer size: {len(self.buffer)}.", "green")
            
   

    

   

    # def reset_memory(self):
    #     """ Reset the priority queue. Initialize each candidate with the priority to be the depth.
    #     """
    #     # Initialize all priorities again.
    #     self.memory.memory = [(candidate.depth, candidate) for _, candidate in self.memory.memory]
    #     heapq.heapify(self.memory.memory)

    # def reset_memory(self, factor: float = 0.9):
    #     # TODO: make sure all predicted scores are updated before resetting the memory.
    #     # TODO: update this function
    #     """
    #     Reduce the value of self.epsilon by a factor. Then update the epsilon-cover memory using nodes in buffer.
    #     """
    #     assert len(self.buffer) > 0, "Buffer is empty. Cannot reset the memory with the temporary buffer."
    #     self.epsilon *= factor
    #     print_color(f"All candidates have been explored. Resetting the memory. New epsilon: {self.epsilon}.", "green")
    #     batch = [(0,candidate) for candidate in self.buffer]
    #     self.regressor.predict_scores(batch)
    #     # sort the buffer by the predicted scores
    #     self.buffer.sort(key=lambda x: x.predicted_score, reverse=True)
    #     count_added = 0
    #     for candidate in self.buffer:
    #         distance = calculate_distance_to_memory(self.memory.memory, candidate)
    #         if distance > self.epsilon:
    #             heapq.heappush(self.memory.memory, (candidate.depth, candidate))
    #             self.buffer.remove(candidate)
    #             count_added += 1
        
        
    #     print_color(f"Added {count_added} candidates to the memory. Buffer size: {len(self.buffer)}.", "green")
    #     if count_added > 0:
    #         self.print_memory_stats()
        
    def explore(self, verbose: bool = False, **kwargs):
       
        # pop top self.num_candidates candidates from the priority queue
        top_candidates = [] 
        priorities = [] 
        
        # while min([priority for _, candidate in self.memory.memory]) == self.max_depth+1:            
        #     # check if all candidates have been explored
        #     self.reset_memory()
        neg_priority, candidate = self.memory.pop()  # pop the top candidate from the priority queue
        priority = - neg_priority  # remember that we stored negative scores in the priority queue
        priorities.append(priority)  # store the priority of the candidate
        top_candidates.append(candidate)  # add the candidate to the top candidates
        # only one candidate is popped from the memory, so we can get the depth from the candidate. This is used for adding depth attribute to new candidates.
        self.depth = candidate.depth
        
        # NOTE some top_candidates can be duplicates
        mean_scores = [c.mean_score() for c in top_candidates]
        mean_scores = [s for s in mean_scores if s is not None]  # filter out None scores
        info_dict = {
            'num_exploration_candidates': len(top_candidates),
            'exploration_candidates_mean_priority': safe_mean(priorities),  # list of priorities of the exploration candidates
            'exploration_candidates_mean_score': safe_mean(mean_scores),  # list of mean scores of the exploration candidates
            'exploration_candidates_average_num_rollouts': safe_mean([c.num_rollouts for c in top_candidates]),
        }

        return top_candidates, priorities, info_dict

    def print_memory_stats(self):
        # For debugging, print all candidates: number, mean_score(), num_rollouts, predicted_score. It is better to see an increasing trend in the predicted scores.
        print("--- Printing memory stats...")
        
        print("Long-term memory:")
        # sort the memory by the depth then print stats
        # do not change the order of the memory.
        
        # Create a temporary memory sorted by depth for printing
        temp_memory = sorted(self.long_term_memory.memory, key=lambda x: x[1].depth)
        
        for i, (priority, candidate) in enumerate(temp_memory):
            mean_score = candidate.mean_score()
            mean_score_str = f"{mean_score:.4g}" if mean_score is not None else "None"
            print(f" Depth: {candidate.depth}, Candidate {i}, Mean Score: {mean_score_str}, Num Rollouts: {candidate.num_rollouts}, Predicted Score: {candidate.predicted_score}")
            
    

class PS_Regressor_EpsilonCover(PrioritySearch_with_Regressor):
    """ 
    A subclass of PrioritySearch_with_Regressor, which keeps an epsilon-cover of memory. Reject new candidates that are in the epsilon-cover of the memory.
    """
    def __init__(self,
                 epsilon: float = 0.1,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
    
    def update_memory(self, validate_results, verbose: bool = False, **kwargs):
        """ 
        Reject new candidates that are in the epsilon-cover of the memory.
        """
        print("--- Updating memory with validation results...") if verbose else None
        new_candidates= [] # new candidates will be added here.
        for candidate, rollouts in validate_results.items():
            if not self.use_validation:
                assert len(rollouts) == 0, "No validation, there should be no rollouts here."
            candidate.add_rollouts(rollouts)  # add the rollouts to the
            if candidate.num_rollouts > 0: # old candidate
                self.memory.push(self.max_score, candidate)
            else: # new candidate
                new_candidates.append(candidate)
        count_new_candidates = 0
        self.regressor.add_embeddings_to_candidates(new_candidates)
        for new_candidate in new_candidates:
            distance = calculate_distance_to_memory(self.memory.memory, new_candidate)
            if distance > self.epsilon: # only collect new candidates those are not in the epsilon-neighborhood of the memory.
                count_new_candidates += 1
                self.memory.push(self.max_score, new_candidate)
        print_color(f"Proposed {len(new_candidates)} new candidates, {count_new_candidates} of them are added to the memory.", "green")

   

   

        

    