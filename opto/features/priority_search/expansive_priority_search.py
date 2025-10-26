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

class ExpansivePrioritySearch(PrioritySearch_with_Regressor):
    """
    A search algorithm that uses a priority queue to explore the parameter space and propose new candidates.
    """

    def __init__(self,
                 max_depth: int = 100,
                 epsilon: float = 0.3,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.max_depth = max_depth
        self.epsilon = epsilon # epsilon-greedy exploration parameter

    def train(self,num_candidates: int = 1,batch_size: int = 2,num_batches: int = 10, *args, **kwargs):
        assert num_candidates == 1, "ExpansivePrioritySearch only supports one candidate at a time."
        print_color(f"ExpansivePrioritySearch: num_candidates = {num_candidates}, batch_size = {batch_size}, num_batches = {num_batches}", "green")
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
        else:  # The first iteration.
            max_mem_size = self.memory.size if self.memory.size is not None else float('inf')
            while len(self.memory) < min(max_mem_size, self.num_candidates):
                original_candidate = ModuleCandidate(self.agent, optimizer=self.optimizer)
                self.regressor.add_embeddings_to_candidates([original_candidate])
                original_candidate.depth = 1
                heapq.heappush(self.memory.memory, (original_candidate.depth, original_candidate))
                # self.memory.push(original_candidate.depth, original_candidate)  # Push the base agent as the first candidate (This gives the initialization of the priority queue)
        self.regressor.update(self.memory.memory)
        self.regressor.predict_scores(self.memory.memory)

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
        for new_candidate in new_candidates:
            distance = calculate_distance_to_memory(self.memory.memory, new_candidate)
            if distance > self.epsilon: # only collect new candidates those are not in the epsilon-neighborhood of the memory.
                count_new_candidates += 1
                new_candidate.depth = self.depth + 1 # self.depth is the depth of the last popped candidate.
                heapq.heappush(self.memory.memory, (self.depth+1, new_candidate))
                # self.memory.push(self.depth+1, new_candidate)
        print_color(f"Proposed {len(new_candidates)} new candidates, {count_new_candidates} of them are added to the memory.", "green")
            
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
        # if candidate.mean_score() is None:
        #     return 0.0
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
            
    def explore(self, verbose: bool = False, **kwargs):
       
        # pop top self.num_candidates candidates from the priority queue
        top_candidates = [] 
        priorities = [] 
        
        while len(top_candidates) < self.num_candidates and len(self.memory) > 0:
            priority, candidate = self.memory.pop()  # pop the top candidate from the priority queue
            if priority == self.max_depth+1: 
                # In this case, all candidates in the search tree have been explored. But we may not reach num_steps. To handle this, we reset the priority queue. Initialize each candidate with the priority to be the depth.
                print_color(f"All candidates have been explored. Resetting the priority queue.", "magenta")
                # push back the candidate we just popped.
                heapq.heappush(self.memory.memory, (candidate.depth, candidate))
                # Initialize all priorities again.
                self.memory.memory = [(candidate.depth, candidate) for _, candidate in self.memory.memory]
                heapq.heapify(self.memory.memory)
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

   

        

    