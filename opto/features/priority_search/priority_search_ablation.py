import numpy as np
import copy
import heapq
import time
from typing import Union, List, Tuple, Dict, Any, Optional, Callable
from opto import trace
from opto.trace.nodes import ParameterNode
from opto.optimizers.optimizer import Optimizer
from opto.trainer.utils import async_run, safe_mean
from opto.trainer.algorithms.basic_algorithms import batchify
from opto.features.priority_search.search_template_ablation import SearchTemplate, Samples, BatchRollout, save_train_config
from opto.features.priority_search.utils import set_module_parameters, remap_update_dict, create_module_from_update_dict, is_module_copy, deepcopy_module
from opto.features.priority_search.regressor import EnsembleLogisticRegressor
from opto.optimizers.utils import print_color


    


class ModuleCandidate:
    """ A container used by PrioritySearch to store a candidate module as (its base module and update dictionary) and its statistics. """

    def __init__(self,
                 base_module: trace.Module,
                 depth: int=None,
                 update_dict: Optional[Dict[ParameterNode, Any]] = None,
                 optimizer: Optimizer = None,
                 ):
        """ A candidate module with its base module and update dictionary.
        Args:
            base_module (trace.Module): The base module to use as a template for the candidate.
            update_dict (dict): A dictionary of ParameterNode: value pairs to update the base module; the key can be a deep copy of the base module's parameters.
            stats (dict): A dictionary of statistics about the candidate.
        """
        assert isinstance(base_module, trace.Module), "base_module must be a trace.Module."
        if update_dict is None:
            # if no update_dict is provided, use the base_module's parameters as the update_dict
            update_dict = {p: p.data for p in base_module.parameters()}
        else:
            assert isinstance(optimizer, Optimizer), "optimizer must be an instance of Optimizer when update_dict is provided."
        assert update_dict is not None, "update_dict must be provided."
        self.base_module = base_module
        self.update_dict = update_dict
        self.optimizer = optimizer  # the optimizer used to generate the update_dict; can be None, which indicates the base_module is used.
        self.update_dict = remap_update_dict(self.base_module, self.update_dict)
        self.rollouts = []  # list of dicts containing the rollout information (not BatchRollout, but a list of dicts)
        self.created_time = time.time()
        self.children = []  # list of ModuleCandidate objects that are children of this candidate
        assert depth is not None, "depth must be provided."
        self.depth = depth
        self.is_new = True # If ever been added to the memory, set to False.
        self.trajecories = [] # list of conversations of tasks.

    def get_module(self):
        """ Apply the update_dict to the base_module and return the updated module.
        A new module is always created so the base_module is not modified.
        The new module has a new attribute _module_candidate which is this candidate."""
        module = create_module_from_update_dict(self.base_module, self.update_dict) if self.update_dict else deepcopy_module(self.base_module)  #
        setattr(module, '__TRACE_RESERVED_module_candidate_id', id(self))
        return module  # return the updated module

    def apply_update(self, base_module=None):
        """ Apply update to the base_module in place. """
        set_module_parameters(base_module or self.base_module, self.update_dict)


    def __eq__(self, other):
        """ Check if two candidates are equal based on their base_module and update_dict. """
        assert isinstance(other, ModuleCandidate), "other must be an instance of ModuleCandidate."
        return (self.update_dict == other.update_dict) and is_module_copy(self.base_module, other.base_module) and (id(self.optimizer) == id(other.optimizer))

    def __lt__(self, other):
        """ Compare two candidates based on their update_dict. """
        assert isinstance(other, ModuleCandidate), "other must be an instance of ModuleCandidate."
        return self.created_time > other.created_time
        # self < other if, self is created later than other
        # Since we will use minheap, and this would give priority to later created candidates in the heap memory.

    def __hash__(self):
        """ Hash the candidate based on its update_dict. """
        return hash((frozenset(self.update_dict.items()), id(self.optimizer), id(self.base_module)))

    def add_rollouts(self, rollouts: List[Dict[str, Any]]):
        """ Add rollouts to the candidate. """
        assert isinstance(rollouts, list), "rollouts must be a list of dicts."
        assert all(isinstance(r, dict) for r in rollouts), "All rollouts must be dicts."
        # Each rollout is a dict with keys: 'module', 'x', 'info', 'target', 'score', 'feedback'
        assert all('module' in r and 'x' in r and 'info' in r and 'target' in r and 'score' in r and 'feedback' in r for r in rollouts), \
            "Each rollout must contain 'module', 'x', 'info', 'target', 'score', and 'feedback' keys."

        self.rollouts.extend(rollouts)

    def mean_score(self):
        """ Compute the score of the candidate based on the rollouts. """
        if not self.rollouts:
            return None
        return safe_mean([r['score'] for r in self.rollouts])

    def standard_deviation(self):
        """ Compute the standard deviation of the scores of the candidate. """
        if not self.rollouts:
            return None
        return np.std([r['score'] for r in self.rollouts if r['score'] is not None])

    # Add children statistics
    def children_rollouts(self):
        """ Compute the rollouts of the children of the candidate. """
        if not self.children:
            return []
        return [rollout for child in self.children for rollout in child.rollouts]
    
    def children_num_rollouts(self):
        """ Compute the number of rollouts of the children of the candidate. """
        if not self.children:
            return 0
        return sum([len(child.rollouts) for child in self.children])
    
    def children_mean_score(self):
        """ Compute the mean score of the children of the candidate. """
        if not self.children:
            return None
        return safe_mean([r['score'] for r in self.children_rollouts()])

    def compute_score_confidence(self, min_score, max_score, scaling_constant=1.0, total_trials=1):
        """Compute the UCB, mean, LCB score for the candidate. After queried, the number of confidence queries is incremented.

        UCB = mean_score + scaling_constant * sqrt(ln(total_trials) / n_scores) * (max_score - min_score)
        UCB = clip(UCB, min_score, max_score)

        LCB = mean_score - scaling_constant * sqrt(ln(total_trials) / n_scores) * (max_score - min_score)
        LCB = clip(LCB, min_score, max_score)

        Args:
            min_score (float): The minimum score for clipping.
            max_score (float): The maximum score for clipping.
            scaling_constant (float): The scaling constant for the exploration term.
            total_trials (int): The total number of trials conducted. Must be at least 1.
        Returns:
            lcb_score (float): The lower confidence bound score.
            mean_score (float): The mean score.
            ucb_score (float): The upper confidence bound score.
        """
        # Get scores from rollouts
        scores = [r['score'] for r in self.rollouts]
        # Filter out None scores
        scores = [s for s in scores if s is not None]

        if not scores:
            return min_score, None, max_score

        # Calculate mean score for this candidate
        mean_score = np.mean(scores)
        n_scores = len(scores)
        assert n_scores == self.num_rollouts, "Number of scores should match number of rollouts."

        # Calculate how many times the confidence interval has been used to form a union bound
        assert total_trials >= 1, "total_trials must be at least 1."
        total_trials = total_trials + 1 # this is an upper bound, since log(1) = 0

        # Compute the exploration term based on Hoeffding's inequality
        exploration_term = scaling_constant * np.sqrt(np.log(total_trials) / n_scores) * (max_score - min_score)

        # Calculate UCB score
        ucb_score = mean_score + exploration_term
        ucb_score = np.clip(ucb_score, min_score, max_score)

        # Calculate LCB score
        lcb_score = mean_score - exploration_term
        lcb_score = np.clip(lcb_score, min_score, max_score)

        return lcb_score, mean_score, ucb_score


    @property
    def num_rollouts(self):
        """ Return the number of rollouts collected for this candidate. """
        return len(self.rollouts)


class HeapMemory:
    # This is a basic implementation of a heap memory that uses a priority queue to store candidates.
    # Later on this will be replaced by a memory DB.

    # NOTE that the heap memory is a max-heap, so we store negative scores to use the default min-heap behavior of heapq.
    def __init__(self, size=None, processing_fun: Callable = None):
        """ Initialize an empty heap memory. """
        self.memory = []
        self._size = size  # Optional size limit for the heap memory
        self.processing_fun = processing_fun

    def push(self, score, data):
        """ Push an item to the heap memory. """
        data = self.processing_fun(data) if self.processing_fun is not None else data
        # Set the is_new flag to False when adding to the memory.
        data.is_new = False
        heapq.heappush(self.memory, (-score, data))
        if len(self.memory) > self.size:
            # NOTE a heuristic for now
            self.memory = self.memory[:self.size]  # Keep only the top `size` items

    def pop(self):
        """ Pop the top item from the heap memory. """
        if not self.memory:
            raise IndexError("pop from an empty heap memory")
        return heapq.heappop(self.memory)

    def append(self, memory):
        """ Append another heap memory to this heap memory. """
        assert isinstance(memory, HeapMemory), "memory must be an instance of HeapMemory."
        for item in memory:
            self.push(-item[0], item[1])  # item is (-score, data)
        if len(self.memory) > self.size:
            self.memory = self.memory[:self.size]  # Keep only the top `size` items

    def reset(self):
        """ Reset the heap memory to be empty. """
        self.memory = []

    @property
    def size(self):
        """ Return the size limit of the heap memory. """
        return self._size if self._size is not None else float('inf')

    def __len__(self):
        """ Return the number of items in the heap memory. """
        return len(self.memory)

    def __bool__(self):
        """ Return True if the heap memory is not empty, False otherwise. """
        return len(self.memory) > 0

    def __iter__(self):
        """ Iterate over the items in the heap memory. """
        return iter(self.memory)

    def best(self, criterion=None):
        """ Return the best item in the heap memory without removing it.

        If criterion is None, return the item with the highest priority (lowest negative score).
        If criterion is a callable function, return the item that maximizes the criterion.
        """
        if not self.memory:
            raise IndexError("best from an empty heap memory")
        if criterion is None:
            return self.memory[0]  # return the item with the highest priority (lowest negative score)
        else:
            assert callable(criterion), "criterion must be a callable function."
            def _criterion(x):
                neg_score, candidate = x
                p = criterion(candidate)
                return p if p is not None else 0
            return max(self.memory, key=lambda x: _criterion(x))

# TODO check saving and loading
class PrioritySearch(SearchTemplate):
    """ A search algorithm that uses a priority queue to explore the parameter space and propose new candidates.

        It provides a scalable template for implementing search algorithms based on asynchronous generation, validation, and testing.
        In each iteration,
            1. It proposes a best agent and a set of `num_candidates` exploration agents that have the highest scores in the priority queue.
            2. The best agent is tested for performance if test_frequency is met.
            3. `num_batches` minibatches of `batch_size` samples are drawn from the training dataset, and the exploration agents are run on the samples. This creates a set of agent rollouts, where each rollout contains the agent module, input, info, target, score, and feedback. For each agent, rollouts of each minibatch are grouped together as a connected subgraph (represented as the BatchRollout object). In total, this step creates `num_candidates * num_batches` subgraphs.
            4. Optimizer is run on each subgraph to propose new parameters for the agents. `num_proposals` proposals are generated for each subgraph. This results in `num_subgraphs * num_proposals` total proposals.
            5. The proposed parameters are validated by running the agents on the validation dataset, which can be the current batch or a separate validation dataset when provided. When validate_exploration_candidates is set to True, the exploration candidates are also validated.
            6. The validation results are used to update the priority queue, which stores the candidates and their scores. The candidates are stored as ModuleCandidate objects, which contain the base module, update dictionary, and rollouts (i.e. raw statistics of the candidate).

        This algorithm template can be subclassed to implement specific search algorithms by overriding the `exploit`, `explore`, and `compute_exploration_priority` methods.
        The `exploit` method is used to select the best candidate from the priority queue, the `explore` method is used to generate new candidates from the priority queue, and
        the `compute_exploration_priority` method is used to compute the score for ranking in the priority queue.

        By default, `compute_exploration_priority` computes the mean score of the rollouts. `exploit` simply returns the candidate with highest priority from the priority queue, and `explore` generates the top `num_candidates` candidates from the priority queue.


        `compute_exploration_priority`, `compute_exploitation_priority` can be overridden to implement different strategies for computing the priority and selecting the best candidate.
    """

    @save_train_config
    def train(self,
              guide, # guide to provide feedback
              train_dataset,  # dataset of (x, info) pairs to train the agent
              *,
              # validation
              validate_dataset = None, # same format as train_dataset; if None, use the current batch.
              validate_guide = None,  #  to provide scores for the validation set
              # training loop
              batch_size = 1,  # batch size for updating the agent
              num_batches = 1,  # number of batches to use from the dataset in each iteration
              score_range = None,  # range of (min_score, max_score) to clip the scores; if None, no clipping is applied
              num_epochs = 1,  # number of training epochs (int or None)
              num_steps = None,  # number of training steps (int or None)
              num_threads = None,  # maximum number of threads to use
              verbose = False,  # whether to print the output of the agent
              # evaluation
              test_dataset = None, # dataset of (x, info) pairs to evaluate the agent
              test_frequency: Union[int, None] = 1, # frequency of evaluation (set it to be negative to skip the first evaluation)
              num_test_samples: int = 1,  # number of times to evaluate each input; when greater than 1, the scores are averaged.
              # logging
              log_frequency = None,  # frequency of logging
              save_frequency: Union[int, None] = None,  # frequency of saving the agent
              save_path: str = "checkpoints/agent.pkl",  # path to save the agent
              # Priority Search specific parameters
              num_candidates: int = 10,  # number of candidates to propose for exploration
              num_proposals: int = 1,  # number of proposals to generate per optimizer
              validate_exploration_candidates: bool = True,  # whether to validate the proposed parameters for exploration
              use_best_candidate_to_explore: bool = True,  # whether to use the best candidate as part of the exploration candidates
              long_term_memory_size: Optional[int] = None,  # size of the long-term heap memory to store the candidates; if None, no limit is set
              short_term_memory_size: Optional[int] = None,  # size of the short-term memory to store the most recent candidates; if None, no limit is set
              memory_update_frequency: Optional[int | None] = 0,  # number of iterations to keep the candidates in the short-term memory before merging them into the long-term memory. 0 means only long-term memory is used. None means only short-term memory is used.
              score_function: str = 'mean',  # function to compute the score for the candidates; 'mean' or 'ucb'
              ucb_exploration_constant: float = 1.0,  # exploration constant for UCB score function
              decouple_optimizers: bool = True,  # whether to decouple the optimizers for each candidate; if True, each candidate will have its own optimizer instance; if False, all candidates share the same optimizer instance.
              # Additional keyword arguments
              **kwargs
              ):
        """ Train the agent using the Priority Search algorithm.

        Args:
            guide (callable): A function that provides feedback for the agent.
            train_dataset (list): A list of (x, info) pairs to train the agent.
            validate_dataset (list, optional): A list of (x, info) pairs to validate the proposed candidates. If None, the current batch is used. Defaults to None.
            validate_guide (callable, optional): A function that provides feedback for the validation set. If None, the training guide is used. Defaults to None.
            batch_size (int, optional): The batch size for updating the agent. Defaults to 1.
            num_batches (int, optional): The number of batches to use from the dataset in each iteration. Defaults to 1.
            score_range (tuple, optional): A tuple of (min_score, max_score) to clip the scores. If None, it's set to (0, 1).
            num_epochs (int, optional): The number of training epochs. Defaults to 1.
            num_threads (int, optional): The maximum number of threads to use. If None, it uses the number of CPU cores. Defaults to None.
            verbose (bool, optional): Whether to print the output of the agent. Defaults to False.
            test_dataset (list, optional): A list of (x, info) pairs to evaluate the agent. If None, no evaluation is performed. Defaults to None.
            test_frequency (int or None, optional): The frequency of evaluation. If None, no evaluation is performed. If negative, skips the first evaluation. Defaults to 1.
            num_test_samples (int, optional): The number of times to evaluate each input; when greater than 1, the scores are averaged. Defaults to 1.
            log_frequency (int or None, optional): The frequency of logging. If None, no logging is performed. Defaults to None.
            save_frequency (int or None, optional): The frequency of saving the agent. If None, no saving is performed. Defaults to None.
            save_path (str, optional): The path to save the agent. Defaults to "checkpoints/agent.pkl".
            num_candidates (int, optional): The number of candidates to propose for exploration. Defaults to 10.
            num_proposals (int, optional): The number of proposals to generate per optimizer. Defaults to 1.
            validate_exploration_candidates (bool, optional): Whether to validate the proposed parameters for exploration. Defaults to True.
            use_best_candidate_to_explore (bool, optional): Whether to use the best candidate as part of the exploration candidates. Defaults to True.
            long_term_memory_size (int, optional): The size of the heap memory to store the candidates. If None, no limit is set. Defaults to None. long-term memory stores only feedback and score.
            short_term_memory_size (int, optional): The size of the short-term memory to store the most recent candidates. If None, no limit is set. Defaults to None. short-term memory stores full rollout information.
            memory_update_frequency (int, optional): The number of iterations to keep the candidates in the short-term memory before merging them into the long-term memory. Defaults to 0, which means only long-term memory is used. None means only short-term memory is used.
            score_function (str, optional): The function to compute the score for the candidates; 'mean' or 'ucb'. Defaults to 'mean'.
            ucb_exploration_constant (float, optional): The exploration constant for UCB score function. Defaults to 1.0.
            **kwargs: Additional keyword arguments that may be used by the implementation.
        """

        # Initialize search parameters and memory
        self._initialize_search_parameters(
            num_candidates=num_candidates,
            num_proposals=num_proposals,
            validate_exploration_candidates=validate_exploration_candidates,
            use_best_candidate_to_explore=use_best_candidate_to_explore,
            score_function=score_function,
            score_range=score_range,
            ucb_exploration_constant=ucb_exploration_constant,
            long_term_memory_size=long_term_memory_size,
            short_term_memory_size=short_term_memory_size,
            memory_update_frequency=memory_update_frequency,
            decouple_optimizers=decouple_optimizers,
        )

        self._enforce_using_data_collecting_candidates = False
        self.num_epochs = num_epochs
        
        # enforce only data collecting candidates are used in in calling match_candidates_and_samples
        # this attribute is purposefully designed to be only modified by subclasses, not through input arguments.
        self.default_batch_size, self.default_num_batches = None, None
        self.train_dataset = train_dataset
        super().train(guide=guide,
                      train_dataset=train_dataset,
                      validate_dataset=validate_dataset,
                      validate_guide=validate_guide,
                      batch_size=batch_size,
                      num_batches=num_batches,
                      score_range=score_range,
                      num_epochs=num_epochs,
                      num_steps=num_steps,
                      num_threads=num_threads,
                      verbose=verbose,
                      test_dataset=test_dataset,
                      test_frequency=test_frequency,
                      num_test_samples=num_test_samples,
                      log_frequency=log_frequency,
                      save_frequency=save_frequency,
                      save_path=save_path,
                      **kwargs)

    def _initialize_search_parameters(self, *,
                                    num_candidates,
                                    num_proposals,
                                    validate_exploration_candidates,
                                    use_best_candidate_to_explore,
                                    score_function,
                                    score_range,
                                    ucb_exploration_constant,
                                    long_term_memory_size,
                                    short_term_memory_size,
                                    memory_update_frequency,
                                    decouple_optimizers):
        """Initialize search parameters and memory structures.

        Args:
            num_candidates (int): Number of candidates to propose for exploration
            num_proposals (int): Number of proposals to generate per optimizer
            validate_exploration_candidates (bool): Whether to validate the proposed parameters
            use_best_candidate_to_explore (bool): Whether to use the best candidate as part of exploration
            score_function (str): Function to compute the score for candidates ('mean' or 'ucb')
            score_range (tuple): Range of scores for UCB computation
            ucb_exploration_constant (float): Exploration constant for UCB score function
            long_term_memory_size (int): Size of the long-term heap memory
            short_term_memory_size (int): Size of the short-term memory
            memory_update_frequency (int): The candidates are merged into long-term memory after this many iterations.
            decouple_optimizers (bool): Whether to decouple optimizers for each candidate
        """
        # Validate and adjust num_candidates based on number of optimizers
        if num_candidates < len(self._optimizers):
            print(f"Warning: num_candidates {num_candidates} is less than the number of optimizers {len(self._optimizers)}. Setting num_candidates to {len(self._optimizers)}.")
            num_candidates = len(self._optimizers)

        # Set core parameters
        self.num_candidates = num_candidates
        self.num_proposals = num_proposals
        self.validate_exploration_candidates = validate_exploration_candidates
        self.use_best_candidate_to_explore = use_best_candidate_to_explore
        self.score_function = score_function
        self.decouple_optimizers = decouple_optimizers

        self.regressor = EnsembleLogisticRegressor(
            embedding_model="gemini/text-embedding-004", num_threads=20, learning_rate=0.001, regularization_strength=1, max_iterations=20000, tolerance=5e-3, gradient_tolerance=5e-3, linear_dim=None, rich_text=True, num_regressors=5, verbose=True
        )

        # Validate and set score range for UCB
        if score_range is None:
            score_range = (0, 1)
        if score_function == 'ucb':
            assert score_range[1] - score_range[0] < float('inf'), \
                "For UCB score function, score_range must be finite. Use 'mean' score function if you want to use unbounded scores."

        self.ucb_exploration_constant = ucb_exploration_constant

        # Initialize candidate tracking variables
        self._exploration_candidates = None
        self._exploration_candidates_priority = None
        self._best_candidate = None
        self._best_candidate_priority = None

        # Initialize memory structures
        if memory_update_frequency is None:
            print("PrioritySearch initialized with only short-term memory.")
            assert short_term_memory_size is None or short_term_memory_size > 0, \
                "short_term_memory_size must be None or greater than 0 when memory_update_frequency is None."
        elif memory_update_frequency == 0:
            print("PrioritySearch initialized with only long-term memory.")
            assert long_term_memory_size is None or long_term_memory_size > 0, \
                "long_term_memory_size must be None or greater than 0 when memory_update_frequency is 0."
        else:
            print(f"PrioritySearch initialized with both short-term and long-term memory. Candidates will be merged into long-term memory every {memory_update_frequency} iterations.")

        self.long_term_memory = HeapMemory(size=long_term_memory_size, processing_fun=self.compress_candidate_memory)
        self.short_term_memory = HeapMemory(size=short_term_memory_size)
        self.memory_update_frequency = memory_update_frequency
    
    def filter_candidates(self, candidates: List[ModuleCandidate]) -> List[ModuleCandidate]:
        """ Filter candidates by their embeddings.
        This function can be overridden by subclasses to filter candidates by other criteria.
        Args:
            candidates (List[ModuleCandidate]): A list of candidates to filter.
        Returns:
            List[ModuleCandidate]: A list of filtered candidates.
        """
        return candidates

    def update(self,
               samples: Union[Samples, None] = None,
               verbose: bool = False,
               **kwargs): #-> Tuple[Dict[ParameterNode, Any], List[trace.Module], Dict[str, Any]]:
        """ Update the agent using the collected samples.
        
        self._best_candidates is a dictionary of the best candidates for each priority criterion.
        """

        # samples is None in the first iteration
        if samples is not None:
            # 1. Propose new parameters based on running LLM optimizers on the collected samples
            # added exploration rollouts to the exploration candidates.
            self.add_exploration_rollouts_to_candidates(self._exploration_candidates, samples)
            candidates = self.propose(samples, verbose=verbose, **kwargs)  # List of ModuleCandidates
            # add embeddings to the candidates asynchronously
            self.regressor.add_embeddings_to_candidates(candidates)
            candidates = self.filter_candidates(candidates)
            # 2. Validate the proposed parameters
            validate_results = self.validate(candidates, samples, verbose=verbose, **kwargs)  # this updates the priority queue
            # 3. Update the priority queue with the validation results
            self.update_memory(validate_results, verbose=verbose, **kwargs)  # samples are provided here in case candidates do not capture full information
        else:  # The first iteration.
            self.default_batch_size, self.default_num_batches = self.get_sampler_batch_size()
            self.memory.push(self.max_score, ModuleCandidate(self.agent,depth=1, optimizer=self.optimizer))  # Push the base agent as the first candidate (This gives the initialization of the priority queue)
        
        self.regressor.update(self.memory.memory)
        self.regressor.predict_scores(self.memory.memory)

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
        self._best_candidates, self._best_candidates_priorities, info_exploit = self.exploit(verbose=verbose, **kwargs)  # get the best candidate (ModuleCandidate) from the priority queue
        self._exploration_candidates, self._exploration_candidates_priority, info_explore = self.explore(verbose=verbose, **kwargs)  # List of ModuleCandidates

        info_log.update(info_exploit)  # add the info from the exploit step
        info_log.update(info_explore)  # add the info from the explore step
        return self._best_candidates, [c.get_module() for c in self._exploration_candidates], info_log

    @property
    def memory(self):
        """ Return the current memory (long-term or short-term) based on the memory update frequency. """
        if self.memory_update_frequency is None:
            return self.short_term_memory
        # memory_update_frequency is finite
        if self.memory_update_frequency == 0 or self.short_term_memory.size == 0:
            return self.long_term_memory
        # short_term_memory is non-zero and memory_update_frequency is positive
        if self.n_iters % self.memory_update_frequency == 0:
            # merge the the short-term memory into the long-term memory
            if len(self.short_term_memory) > 0:
                print('Merging short-term memory into long-term memory of PrioritySearch.')
                self.long_term_memory.append(self.short_term_memory)
                self.short_term_memory.reset()
            return self.long_term_memory
        else:
            return self.short_term_memory

    ## Illustration of `propose``
    # Suppose we have 2 exploration candidates.
    # exploration_candidates = [candidate(param1, optimizer_1), candidate(param2, optimizer_2)]
    # and two batches are collected by sampler.
    #
    # In samples returned by sampler, we have data
    #   module(param1_copy1), batch_1
    #   module(param1_copy2), batch_2
    #   module(param2_copy1), batch_1
    #   module(param2_copy2), batch_2
    #
    # We first match the samples with the exploration candidates as
    #   candidate_batchrollouts_list =
    #       [ (candidate(param1, optimizer_1), batch_1), (candidate(param1, optimizer_1), batch_2),
    #         (candidate(param2, optimizer_2), batch_1), (candidate(param2, optimizer_2), batch_2) ]
    #
    # In backward, we create deepcopies of the optimizers for each batch, and run backward asynchronously.
    #    optimizer_1_copy_1(param1) <- feedback from batch_1
    #    optimizer_1_copy_2(param1) <- feedback from batch_2
    #    optimizer_2_copy_1(param2) <- feedback from batch_1
    #    optimizer_2_copy_2(param2) <- feedback from batch_2
    #
    # In step, we further create deepcopies of the optimizers for each proposal, and run step asynchronously.
    # for n_proposals = 2, we have
    #    optimizer_1_copy_1_copy_1(param1) -> proposal_1
    #    optimizer_1_copy_1_copy_2(param1) -> proposal_2
    #    ...
    #    optimizer_2_copy_2_copy_1(param2) -> proposal_7
    #    optimizer_2_copy_2_copy_2(param2) -> proposal_8
    # which form the new candidate list returned by `propose`.
    #
    def add_exploration_rollouts_to_candidates(self, exploration_candidates: List[ModuleCandidate], samples: Samples):
        """ Add the exploration rollouts to the exploration candidates.
        """
        matched_exploration_candidates_and_samples = self.match_candidates_and_samples(exploration_candidates, samples.samples)
        exploration_results = {}  # dict of ModuleCandidate id: (ModuleCandidate, list of rollouts)
        for c, rollouts in matched_exploration_candidates_and_samples.items():  # rollouts is a list of BatchRollouts
            exploration_results[c] = [ r for rr in rollouts for r in rr.to_list()]
        for candidate, rollouts in exploration_results.items():
            candidate.add_rollouts(rollouts) 
    def propose(self,
                samples : Samples,
                verbose : bool = False,
                **kwargs):
        """ Analyzing samples and propose new parameters using self.optimizer. An independent optimizer is used for the minibatch generated by one agent and generates n_proposals proposals.

        Args:
            samples (Samples): Samples collected by the exploration candidates. If None, the agent's parameters are returned without updating.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            **kwargs: Additional keyword arguments that may be used by the implementation.

        Returns:
            candidates (list of ModuleCandidate): A list of proposed candidates for the next iteration.
        """
        print("--- Proposing new parameters...") if verbose else None
        assert isinstance(samples, Samples), "samples must be an instance of Samples."
        samples = samples.samples  # list of BatchRollout objects
        n_proposals = self.num_proposals  # number of proposals to generate per optimizer

        # Associate each BatchRollout with self._exploration_candidates
        matched_candidates_and_samples = self.match_candidates_and_samples(self._exploration_candidates, samples)
        # NOTE len(matched_candidates_and_samples) <= len(self._exploration_candidates) since some exploration candidates might be duplicated.
        candidate_batchrollouts_list = [ (k,b) for k, v in matched_candidates_and_samples.items() for b in v]
        n_batches = len(candidate_batchrollouts_list)  # number of batch rollouts in the samples

        def copy_optimizer(optimizer):
            return copy.deepcopy(optimizer) if self.decouple_optimizers else optimizer

        # need to copy optimizer for the n_batches
        def _backward(n):
            candidate, rollouts = candidate_batchrollouts_list[n]
            optimizer = candidate.optimizer or self.optimizer
            # Create a copy of the optimizer to avoid modifying the original one and to allow parallel execution
            optimizer = copy_optimizer(optimizer)
            optimizer.parameters = rollouts.module.parameters()  # set the optimizer's parameters to the proposal's parameters
            targets = [r.target for r in rollouts]
            feedbacks = [r.feedback for r in rollouts]
            # batchify the targets and feedbacks
            target = batchify(*targets)
            feedback = batchify(*feedbacks).data  # str
            # standard optimizer step
            optimizer.zero_feedback()  # reset the optimizer's feedback
            optimizer.backward(target, feedback)  # compute the gradients based on the targets and feedbacks
            return optimizer

        args_list = [(n,) for n in range(n_batches)]
        optimizers = async_run([_backward]*n_batches,  # run the optimizer step for each agent in parallel
                                 args_list=args_list,
                                 max_workers=self.num_threads,  # use the number of threads specified in the class
                                 description='Backward')
        assert len(optimizers) == n_batches, "Number of optimizers must match number of batch rollouts."
        # need to copy optimizer for the n_proposals
        # NOTE when optimizer is deepcopied, its parameters are not copied.
        optimizers = [copy_optimizer(o) for o in optimizers ] * n_proposals  # repeat args_list n_proposals times
        assert len(optimizers) == n_batches * n_proposals, "Number of optimizers must match number of batch rollouts times number of proposals."

        # For each optimizer, containing the backward feedback, we call it n_proposals times to get the proposed parameters.
        def _step(n):
            optimizer = optimizers[n]
            update_dict = optimizer.step(verbose=verbose, num_threads=self.num_threads, bypassing=True, **kwargs)
            if not update_dict:  # if the optimizer did not propose any updates
                return None # return None to indicate no updates were proposed
            # update_dict may only contain some of the parameters of the agent, we need to make sure it contains all the parameters
            # since the current agent might have different parameters than the one used by the optimizer
            for param in optimizer.parameters: # for all parameters
                if param not in update_dict: # update_dict misses some parameters
                    update_dict[param] = param.data # add the parameter to the update_dict
            # the update_dict is linked to the copied parameters of the agent, we set it back to the agent's parameters
            update_dict = remap_update_dict(self.agent, update_dict)  # remap the update dict to the agent's parameters
            return update_dict  # return the proposed parameters

        args_list = [(n,) for n in range(n_batches*n_proposals)]
        update_dicts = async_run([_step]*n_batches*n_proposals,  # run the optimizer step for each agent in parallel
                                  args_list=args_list,
                                  max_workers=self.num_threads,  # use the number of threads specified in the class
                                  description=f"Calling optimizers: Generating {n_proposals} proposals for each of {n_batches} batches",)

        # Clear the optimizers to avoid memory leaks
        for optimizer in optimizers:
            optimizer.zero_feedback()  # reset the optimizer's feedback
            optimizer.parameters = []  # clear the optimizer's parameters to avoid memory leaks

        # update_dicts is a list of dicts of length n_batches * n_proposals
        # Create ModuleCandidate objects for each proposed update_dict that is non-trivial
        # Track which parent candidate each new candidate comes from
        # Cookbook to get parent-children relationships
        candidates = []
        for i, (update_dict, optimizer) in enumerate(zip(update_dicts, optimizers)):
            if update_dict is not None:
                
                
                # Find the parent candidate that generated this new candidate
                # The optimizers list is structured as: [opt_0, opt_1, ..., opt_{n_batches-1}] repeated n_proposals times
                # So update_dicts[i] corresponds to candidate_batchrollouts_list[i % n_batches]
                parent_batch_idx = i % n_batches
                parent_candidate, _ = candidate_batchrollouts_list[parent_batch_idx]

                new_candidate = ModuleCandidate(self.agent,depth=parent_candidate.depth+1, update_dict=update_dict, optimizer=optimizer)
                # parent_candidate.children.append(new_candidate)
                candidates.append(new_candidate)
        
        return candidates

    def validate(self,
                 candidates: List[ModuleCandidate],
                 samples: Samples,
                 verbose: bool = False,
                 **kwargs):
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
        # Do a hack here: we added the samples before this step.
        validate_samples = Samples([], {'inputs': [], 'infos': []})

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


        matched_candidates_and_samples = self.match_candidates_and_samples(exploration_candidates + candidates, validate_samples.samples)
        results = {}  # dict of ModuleCandidate id: (ModuleCandidate, list of rollouts)
        for c, rollouts in matched_candidates_and_samples.items():  # rollouts is a list of BatchRollouts
            results[c] = [ r for rr in rollouts for r in rr.to_list()]  # we only need the list of dicts

        return results

    def match_candidates_and_samples(
            self,
            candidates: List[ModuleCandidate],
            samples: List[BatchRollout]):
        """
        Match the given candidates with the provided samples.

        Args:
            candidates (list of ModuleCandidate): A list of ModuleCandidate objects representing the proposed parameters.
            samples (list of BatchRollout): A Samples object containing a list of BatchRollout objects, where each BatchRollout contains rollouts collected by an agent on different inputs.
        Returns:
            results (dict): A dictionary where the keys are ModuleCandidate objects and the values are lists of BatchRollouts collected by the corresponding ModuleCandidate.

        """
        # In general, there may be multiple BatchRollouts collected by the same ModuleCandidate.
        # We group the rollouts by the agent (ModuleCandidate) and return a dictionary
        # where the keys are the ModuleCandidate objects and the values are Samples

        # Group the samples by the ModuleCandidate id
        _results = { c: [] for c in candidates}  # dict of ModuleCandidate: list of BatchRollouts
        ids = {id(c): c for c in candidates}  # dict of ModuleCandidate id: ModuleCandidate

        for rollouts in samples:
            assert isinstance(rollouts, BatchRollout), "Each element in samples must be a BatchRollout object."
            # rollouts is a BatchRollout object
            module = rollouts.module  # trace.Module
            key = getattr(module, '__TRACE_RESERVED_module_candidate_id')  # use the candidate as the key
            if key not in ids:
                raise ValueError(f"ModuleCandidate with id {key} not found in results. Samples are not collected by known candidates.")
            # Append the rollouts to the list of rollouts for the key
            _results[ids[key]].append(rollouts)

        if self._enforce_using_data_collecting_candidates:
            # assert all candidates have at least one rollout
            for c in candidates:
                assert len(_results[c]) > 0, f"ModuleCandidate with id {id(c)} has no rollouts. Samples are not collected by known candidates."

        assert len(samples) == sum(len(rollouts) for rollouts in _results.values()), "All samples must be associated with exploration candidates."
        return _results

    def update_memory(self, validate_results, verbose: bool = False, **kwargs):
        """ Update the priority queue with the validation results.
        Args:
            validate_results (dict): A dictionary where the keys are ModuleCandidate objects and the values are lists of rollouts (list of dicts) containing the module, x, info, target, score, feedback.
            **kwargs: Additional keyword arguments that may be used by the implementation.
        """
        print("--- Updating memory with validation results...") if verbose else None
        for candidate, rollouts in validate_results.items():
            candidate.add_rollouts(rollouts)  # add the rollouts to the candidate
            priority = self.compute_exploration_priority(candidate)  # compute the priority for the candidate
            self.memory.push(priority, candidate)
    def explore(self, verbose: bool = False, **kwargs):
        """ Explore the parameter space and propose new candidates.
        Args:
            **kwargs: Additional keyword arguments that may be used by the implementation.
        Returns:
            list: A list of proposed candidates.
            dict: A dictionary containing logging information about the exploration.
        """
        print(f"--- Generating {min(len(self.memory), self.num_candidates)} exploration candidates...") if verbose else None
        # in ablation study, we do not use the best candidate to explore.
        assert not self.use_best_candidate_to_explore, "In ablation study, we do not use the best candidate to explore."
        # pop top self.num_candidates candidates from the priority queue
        # self._best_candidate is the exploited candidate from the previous iteration
        top_candidates = [self._best_candidate] if self.use_best_candidate_to_explore else []
        priorities = [self._best_candidate_priority] if self.use_best_candidate_to_explore else []  # to store the priorities of the candidates for logging
        while len(top_candidates) < self.num_candidates and len(self.memory) > 0:
            neg_priority, candidate = self.memory.pop()  # pop the top candidate from te priority queue
            priority = - neg_priority  # remember that we stored negative scores in the priority queue
            if self.use_best_candidate_to_explore:
                if candidate is self._best_candidate:  # skip if it is already in the top candidates
                    continue
            priorities.append(priority)  # store the priority of the candidate
            top_candidates.append(candidate)  # add the candidate to the top candidates
        # NOTE some top_candidates can be duplicates
        mean_scores = [c.mean_score() for c in top_candidates]
        mean_scores = [s for s in mean_scores if s is not None]  # filter out None scores
        depths = [c.depth for c in top_candidates]
        assert all(depth is not None for depth in depths), "All exploration candidates must have a depth."
        assert all(depth >= 1 for depth in depths), "All exploration candidates must have a depth of at least 1."
        info_dict = {
            'num_exploration_candidates': len(top_candidates),
            'exploration_candidates_mean_priority': safe_mean(priorities),  # list of priorities of the exploration candidates
            'exploration_candidates_mean_score': safe_mean(mean_scores),  # list of mean scores of the exploration candidates
            'exploration_candidates_mean_depth': safe_mean(depths),  # list of depths of the exploration candidates
            'exploration_candidates_average_num_rollouts': safe_mean([c.num_rollouts for c in top_candidates]),
        }
        if len(top_candidates) < self.num_candidates:
            new_num_batches = int(self.default_num_batches * self.num_candidates/len(top_candidates))
            print(f'Setting sampler num_batches from {self.default_num_batches} to {new_num_batches} to accommodate {self.num_candidates} exploration candidates request using {len(top_candidates)} candidates.')
            self.set_sampler_batch_size(self.default_batch_size, new_num_batches)
        else:
            self.set_sampler_batch_size(self.default_batch_size, self.default_num_batches)
        return top_candidates, priorities, info_dict

    def _get_best_candidate_by_priority(self, priority_compute_fn, priority_name: str) -> Tuple[float, ModuleCandidate]:
        """Helper function to get the best candidate using a specific priority computation function.
        
        Args:
            priority_compute_fn: The function to compute priority for a candidate.
            priority_name (str): Name of the priority metric for assertion messages.
            
        Returns:
            Tuple[float, ModuleCandidate]: The priority value and the best candidate.
        """
        neg_priority, best_candidate = self.memory.best(priority_compute_fn)
        priority = -neg_priority
        assert best_candidate.depth is not None, f"Best candidate for {priority_name} must have a depth."
        return priority, best_candidate

    def exploit(self, verbose: bool = False, **kwargs) -> Tuple[ModuleCandidate, Dict[str, Any]]:
        """ Exploit the best candidate from the priority queue. This method should not change the priority queue.
        Args:
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            **kwargs: Additional keyword arguments that may be used by the implementation.
        Returns:
            ModuleCandidate: The best candidate from the priority queue.
        --------------------------------------------------------------------------
        For ablation study, we want to exploit the best candidates according to different criteria. UCB, LCB and mean prediction are calculated by predicted scores from different regressors. The highest predicted score is the ucb, the lowest predicted score is the lcb, and the mean of the predicted scores is the mean prediction.
        
        - Mean score: mean_score()
        - UCB score: predicted_score
        - LCB score: lcb
        - Mean prediction: mean_prediction
        """
        print("--- Exploiting the best candidate...") if verbose else None
        if not self.memory:
            raise ValueError("The priority queue is empty. Cannot exploit.")
        best_candidates = {}
        priorities = {}
        
        # empirical mean score
        priorities['empirical_mean'], best_candidates['empirical_mean'] = self._get_best_candidate_by_priority(
            self.compute_exploitation_priority_empirical_mean, 'empirical_mean')
        
        # ucb
        priorities['ucb'], best_candidates['ucb'] = self._get_best_candidate_by_priority(
            self.compute_exploitation_priority_ucb, 'ucb')
        
        # lcb
        priorities['lcb'], best_candidates['lcb'] = self._get_best_candidate_by_priority(
            self.compute_exploitation_priority_lcb, 'lcb')
        
        # mean prediction
        priorities['mean_prediction'], best_candidates['mean_prediction'] = self._get_best_candidate_by_priority(
            self.compute_exploitation_priority_mean_prediction, 'mean_prediction')

        # Log information for all 4 versions (16 items total)
        info_dict = {}
        
        # Empirical mean version
        info_dict['best_candidate_priority_empirical_mean'] = priorities['empirical_mean']
        info_dict['best_candidate_depth_empirical_mean'] = best_candidates['empirical_mean'].depth
        info_dict['best_candidate_mean_score_empirical_mean'] = best_candidates['empirical_mean'].mean_score()
        info_dict['best_candidate_num_rollouts_empirical_mean'] = best_candidates['empirical_mean'].num_rollouts
        
        # UCB version
        info_dict['best_candidate_priority_ucb'] = priorities['ucb']
        info_dict['best_candidate_depth_ucb'] = best_candidates['ucb'].depth
        info_dict['best_candidate_mean_score_ucb'] = best_candidates['ucb'].mean_score()
        info_dict['best_candidate_num_rollouts_ucb'] = best_candidates['ucb'].num_rollouts
        
        # LCB version
        info_dict['best_candidate_priority_lcb'] = priorities['lcb']
        info_dict['best_candidate_depth_lcb'] = best_candidates['lcb'].depth
        info_dict['best_candidate_mean_score_lcb'] = best_candidates['lcb'].mean_score()
        info_dict['best_candidate_num_rollouts_lcb'] = best_candidates['lcb'].num_rollouts
        
        # Mean prediction version
        info_dict['best_candidate_priority_mean_prediction'] = priorities['mean_prediction']
        info_dict['best_candidate_depth_mean_prediction'] = best_candidates['mean_prediction'].depth
        info_dict['best_candidate_mean_score_mean_prediction'] = best_candidates['mean_prediction'].mean_score()
        info_dict['best_candidate_num_rollouts_mean_prediction'] = best_candidates['mean_prediction'].num_rollouts
        
        return best_candidates, priorities, info_dict

    def compute_exploitation_priority_empirical_mean(self, candidate) -> float:
        if not isinstance(candidate, ModuleCandidate):
            raise TypeError("candidate must be an instance of ModuleCandidate.")
        # By default, we compute the mean score of the rollouts
        return candidate.mean_score()
    
    def compute_exploitation_priority_ucb(self, candidate) -> float:
        if not isinstance(candidate, ModuleCandidate):
            raise TypeError("candidate must be an instance of ModuleCandidate.")
        return candidate.ucb

    def compute_exploitation_priority_lcb(self, candidate) -> float:
        if not isinstance(candidate, ModuleCandidate):
            raise TypeError("candidate must be an instance of ModuleCandidate.")
        return candidate.lcb
    
    def compute_exploitation_priority_mean_prediction(self, candidate) -> float:
        if not isinstance(candidate, ModuleCandidate):
            raise TypeError("candidate must be an instance of ModuleCandidate.")
        return candidate.mean_prediction

    def set_sampler_batch_size(self, batch_size: int, num_batches: int, set_train_sampler: bool = True, set_validate_sampler: bool = True):

        subbatch_size, batch_size = batch_size, batch_size*num_batches

        if self.train_sampler is not None and set_train_sampler:
            self.train_sampler.loader.batch_size = batch_size  # update the batch size in the DataLoader
            self.train_sampler.subbatch_size = subbatch_size  # update the sub-batch size in the Sampler

        if self.validate_sampler is not None and set_validate_sampler:
            self.validate_sampler.loader.batch_size = batch_size  # update the batch size in the DataLoader
            self.validate_sampler.subbatch_size = subbatch_size  # no sub-batch size for validation

    def get_sampler_batch_size(self):
        if self.train_sampler is not None:
            batchsize, subbatch_size = self.train_sampler.loader.batch_size, self.train_sampler.subbatch_size
            batchsize, num_batches = subbatch_size, batchsize // subbatch_size
            return batchsize, num_batches
        else:
            raise ValueError("train_sampler is not set.")

    # NOTE This function can be overridden by subclasses to compute a different score
    def compute_exploration_priority(self, candidate) -> float:
        """ Compute the score for the candidate based on the rollouts during the validation phase.
        It can be overridden by subclasses to implement a different scoring strategy.

        Args:
            candidate (ModuleCandidate): The candidate for which to compute the score.
        Returns:
            float: The computed score for the candidate. Higher scores indicate higher priority.
        """
        if not isinstance(candidate, ModuleCandidate):
            raise TypeError("candidate must be an instance of ModuleCandidate.")
        # By default, we compute the mean score of the rollouts

        if self.score_function == 'mean':
            # Compute the mean score of the candidate's rollouts
            return candidate.mean_score()
        elif self.score_function == 'time':
            return -candidate.created_time  # latest candidates have higher priority
        elif self.score_function == 'ucb':
            # Compute the Upper Confidence Bound (UCB) score
            lcb_score, mean_score, ucb_score = candidate.compute_score_confidence(
                min_score=self.min_score,
                max_score=self.max_score,
                scaling_constant=self.ucb_exploration_constant,
                total_trials=self.n_iters + 1  # total number of trials conducted so far
            )
            return ucb_score  # return the UCB score
        else:
            raise ValueError(f"Unknown score function: {self.score_function}")

    # NOTE This function can be overridden by subclasses to compute a different score
    def compress_candidate_memory(self, candidate: ModuleCandidate) -> ModuleCandidate:
        """ Compress the memory of the candidate to save space. This is used to preprocess candidates before adding them to long-term memory.
            By default, we save only the feedback and score of each rollout for long-term memory. """
        def _process_rollout(rollout):
            # rollout is a dict containing module, x, info, target, score, feedback
            for k in rollout:
                if k not in ['score']:
                    rollout[k] = None
        candidate = copy.copy(candidate)  # make a copy of the candidate to avoid modifying the original one
        candidate.rollouts = copy.deepcopy(candidate.rollouts)  # deep copy the rollouts to avoid modifying the original one
        for rollout in candidate.rollouts:
            _process_rollout(rollout)
        return candidate

class PrioritySearchUCBExploration(PrioritySearch):
    """ A search algorithm that uses UCB exploration to explore the parameter space and propose new candidates.
    """
    def explore(self, verbose: bool = False, **kwargs):
        """
        Pop top self.num_candidates candidates based on UCB score. 

        After popping the candidates, heapify the memory. heapq.heapify(self.memory.memory)
        """
        print_color(f"Using UCB exploration to explore the parameter space...", "green")
        print(f"--- Generating {min(len(self.memory), self.num_candidates)} exploration candidates...") if verbose else None
        # in ablation study, we do not use the best candidate to explore.
        assert not self.use_best_candidate_to_explore, "In ablation study, we do not use the best candidate to explore."
        # pop top self.num_candidates candidates from the priority queue
        # self._best_candidate is the exploited candidate from the previous iteration
        
        # sort the memory by UCB score. candidate.ucb is the UCB score of the candidate.
        self.memory.memory.sort(key=lambda x: x[1].ucb, reverse=True)
        top_candidates = [candidate for _, candidate in self.memory.memory[:self.num_candidates]]
        priorities = [candidate.ucb for candidate in top_candidates]
        self.memory.memory = self.memory.memory[self.num_candidates:]
        heapq.heapify(self.memory.memory)

        # NOTE some top_candidates can be duplicates
        mean_scores = [c.mean_score() for c in top_candidates]
        mean_scores = [s for s in mean_scores if s is not None]  # filter out None scores
        depths = [c.depth for c in top_candidates]
        assert all(depth is not None for depth in depths), "All exploration candidates must have a depth."
        assert all(depth >= 1 for depth in depths), "All exploration candidates must have a depth of at least 1."
        info_dict = {
            'num_exploration_candidates': len(top_candidates),
            'exploration_candidates_mean_priority': safe_mean(priorities),  # list of priorities of the exploration candidates
            'exploration_candidates_mean_score': safe_mean(mean_scores),  # list of mean scores of the exploration candidates
            'exploration_candidates_mean_depth': safe_mean(depths),  # list of depths of the exploration candidates
            'exploration_candidates_average_num_rollouts': safe_mean([c.num_rollouts for c in top_candidates]),
        }
        if len(top_candidates) < self.num_candidates:
            new_num_batches = int(self.default_num_batches * self.num_candidates/len(top_candidates))
            print(f'Setting sampler num_batches from {self.default_num_batches} to {new_num_batches} to accommodate {self.num_candidates} exploration candidates request using {len(top_candidates)} candidates.')
            self.set_sampler_batch_size(self.default_batch_size, new_num_batches)
        else:
            self.set_sampler_batch_size(self.default_batch_size, self.default_num_batches)
        return top_candidates, priorities, info_dict

    def _get_best_candidate_by_priority(self, priority_compute_fn, priority_name: str) -> Tuple[float, ModuleCandidate]:
        """Helper function to get the best candidate using a specific priority computation function.
        
        Args:
            priority_compute_fn: The function to compute priority for a candidate.
            priority_name (str): Name of the priority metric for assertion messages.
            
        Returns:
            Tuple[float, ModuleCandidate]: The priority value and the best candidate.
        """
        neg_priority, best_candidate = self.memory.best(priority_compute_fn)
        priority = best_candidate.ucb
        assert best_candidate.depth is not None, f"Best candidate for {priority_name} must have a depth."
        return priority, best_candidate

def calculate_distance_to_memory(memory, new_candidate):
        """For a new candidate, calculate the distance to the current memory. That's the least L2 distance to any candidate in the memory.
        
        To use this funciton in PrioritySearch, set memory to be self.memory.memory.
        """
        # assert new_candidate.num_rollouts == 0, "New candidates should have no rollouts."
        # assert new candidate and all candidates in the memory have the  embedding.
        assert hasattr(new_candidate, 'embedding') and all(hasattr(candidate, 'embedding') for _, candidate in memory), "All candidates should have the embedding attribute."
        # calculate the distance to the current memory. That's the least L2 distance to any candidate in the memory.
        min_distance = float('inf')
        for _, candidate in memory:
            distance = np.linalg.norm(np.array(new_candidate.embedding) - np.array(candidate.embedding))
            if distance < min_distance:
                min_distance = distance
        return min_distance

from opto.features.priority_search.summarizer import Summarizer
class EpsilonNetPS(PrioritySearch):
    """
    A subclass of PrioritySearch, which keeps an epsilon-net as the memory. Reject new candidates that are in the epsilon-net of the memory.
    """
    def __init__(self,
                 epsilon: float = 0.01,
                 use_summarizer: bool = False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.use_summarizer = use_summarizer
        self.summarizer = Summarizer(model_name="gemini/gemini-2.0-flash")
        self.context = "Concrete recommendations for generating better agent parameters based on successful patterns observed in the trajectories: "
        
            

    def filter_candidates(self, new_candidates: List[ModuleCandidate]) -> List[ModuleCandidate]:
        """ Filter candidates by their embeddings.
        """
        exploration_memory = [(0, candidate) for candidate in self._exploration_candidates]
        current_memory = self.memory.memory + exploration_memory

        # filter new candidates based on the distance to the current memory.
        num_new_candidates = len(new_candidates)

        added_candidates = []
        success_distances = []
        
        while len(new_candidates) > 0:
            # calculate the distance to the memory for each new candidate
            distances = [calculate_distance_to_memory(current_memory, new_candidate) for new_candidate in new_candidates]
            
            # filter candidates: keep only those with distance > epsilon
            filtered_candidates = []
            filtered_distances = []
            for i, (candidate, distance) in enumerate(zip(new_candidates, distances)):
                if distance > self.epsilon:
                    filtered_candidates.append(candidate)
                    filtered_distances.append(distance)
            
            # if no candidates remain, exit the loop
            if len(filtered_candidates) == 0:
                break
            
            # add the candidate with the largest distance to the memory
            max_distance_idx = np.argmax(filtered_distances)
            new_node = filtered_candidates[max_distance_idx]
            current_memory.append((0, new_node))
            added_candidates.append(new_node)
            success_distances.append(float(filtered_distances[max_distance_idx]))
            
            # remove the added candidate from new_candidates list
            new_candidates = [c for c in filtered_candidates if c is not new_node]

        print_color(f"Proposed {num_new_candidates} new candidates, {len(added_candidates)} of them are added to the memory.", "green")
        # print the distances between the added candidates and the memory before adding them.
        print_color(f"Distances between the added candidates and the memory before adding them: {success_distances}", "green")
        return added_candidates
    
    def compress_candidate_memory(self, candidate: ModuleCandidate) -> ModuleCandidate:
        """ Keep target of each rollout for long-term memory. """
        def _process_rollout(rollout):
            # rollout is a dict containing module, x, info, target, score, feedback
            for k in rollout:
                if k not in ['score', 'target']:
                    rollout[k] = None
        candidate = copy.copy(candidate)  # make a copy of the candidate to avoid modifying the original one
        candidate.rollouts = copy.deepcopy(candidate.rollouts)  # deep copy the rollouts to avoid modifying the original one
        for rollout in candidate.rollouts:
            _process_rollout(rollout)
        return candidate
    
    

    def propose(self,
                samples : Samples,
                verbose : bool = False,
                **kwargs):
        """ 
        Override the propose method to include a summary into the context of the optimizer.
        """
        
        # Use the summarizer to summarize the memory and the exploration candidates.
        if self.use_summarizer:
            # Summarize the memory and the exploration candidates.
            exploration_memory = [(0, candidate) for candidate in self._exploration_candidates]
            print_color(f"Summarizing the history...", "green")
            try: 
                summary = self.summarizer.summarize(self.memory.memory+exploration_memory)
                print_color(f"Summary: {summary}", "green")
                self.context = f"Concrete recommendations for generating better agent parameters based on successful patterns observed in the trajectories: {summary}"
            except RuntimeError as e:
                print_color(f"Error: {e}", "red")
                print_color(f"Using fallback context: {self.context}", "red")
            # Set the context for the optimizer.
            for candidate in self._exploration_candidates:
                candidate.optimizer.set_context(self.context)
        return super().propose(samples, verbose, **kwargs)

class ParetobasedPS(PrioritySearch):
    """
    A subclass of PrioritySearch, which uses Pareto-based exploration to explore the parameter space and propose new candidates.
    """
    def compress_candidate_memory(self, candidate: ModuleCandidate) -> ModuleCandidate:
        """ Compress the memory of the candidate to save space. This is used to preprocess candidates before adding them to long-term memory.
            By default, we save only the feedback and score of each rollout for long-term memory. """
        def _process_rollout(rollout):
            # rollout is a dict containing module, x, info, target, score, feedback
            for k in rollout:
                if k not in ['x', 'score']:
                    rollout[k] = None
        candidate = copy.copy(candidate)  # make a copy of the candidate to avoid modifying the original one
        candidate.rollouts = copy.deepcopy(candidate.rollouts)  # deep copy the rollouts to avoid modifying the original one
        for rollout in candidate.rollouts:
            _process_rollout(rollout)
        return candidate

    def compute_score_for_task_x(self, candidate, x):
        """ 
        Compute the empirical mean score for the candidate for the task x.
        """
        rollouts_x = [ rollout for rollout in candidate.rollouts if rollout['x'] == x]
        return safe_mean([rollout['score'] for rollout in rollouts_x],missing_value=0)

    def get_best_candidates_for_x(self,x):
        """
        Get the candidates with the highest score for the task x.
        """
        best_candidates = []
        highest_score = max(self.compute_score_for_task_x(candidate, x) for _,candidate in self.memory.memory)
        for _,candidate in self.memory.memory:
            if self.compute_score_for_task_x(candidate, x) == highest_score:
                best_candidates.append(candidate)
        return best_candidates

       
    def explore(self, verbose: bool = False, **kwargs):
        """
        1. Keep candidates achieving the highest score for each task
        2. Remove strictly dominated candidates.

        The strict domination definition: for two candidates a,b in best_candidates_for_tasks, we say candidate a strictly dominates candidate b, if all tasks on which b is among the best, a also achieves the highest score.
        """
        print_color(f"Using Pareto-based exploration to explore the parameter space...", "green")
        # get all xs in the train dataset
        xs = [x for x in self.train_dataset['inputs']]
        # best candidates for each task
        best_candidates_for_tasks = {x: self.get_best_candidates_for_x(x) for x in xs}

        # For debugging, print the best candidates for each task.
        # for x in xs:
        #     print_color(f"Best candidates for task {x}: ", "green")
        #     for candidate in best_candidates_for_tasks[x]:
        #         print_color(f"Candidate: {id(candidate)}", "green")
        
        # collect all unique candidates from best_candidates_for_tasks
        all_candidates = list(set(candidate for candidates in best_candidates_for_tasks.values() for candidate in candidates))
        
        # remove strictly dominated candidates
        non_dominated_candidates = []
        for candidate_b in all_candidates:
            is_dominated = False
            # find all tasks where candidate_b is among the best
            tasks_b = [task_x for task_x in xs if candidate_b in best_candidates_for_tasks[task_x]]
            
            # check if any other candidate strictly dominates candidate_b
            for candidate_a in all_candidates:
                if candidate_a is candidate_b:
                    continue
                
                # check if candidate_a is among the best for all tasks where candidate_b is among the best
                if all(candidate_a in best_candidates_for_tasks[task_x] for task_x in tasks_b):
                    # candidate_a strictly dominates candidate_b
                    is_dominated = True
                    break
            
            if not is_dominated:
                non_dominated_candidates.append(candidate_b)
        
        # update best_candidates_for_tasks to only contain non-dominated candidates
        for x in xs:
            best_candidates_for_tasks[x] = [c for c in best_candidates_for_tasks[x] if c in non_dominated_candidates]

        # Print the best candidates after removing the dominated candidates
        # for x in xs:
        #     print_color(f"After removing dominated candidates, best candidates for task {x}: ", "green")
        #     for candidate in best_candidates_for_tasks[x]:
        #         print_color(f"Candidate: {id(candidate)}", "green")

        # Get all candidates in best_candidates_for_tasks as the exploration candidates. Remove the duplicates.
        top_candidates = list(set(candidate for candidates in best_candidates_for_tasks.values() for candidate in candidates))

        print_color(f"Number of pareto candidates: {len(top_candidates)}, we will take {min(len(top_candidates), self.num_candidates)} for exploration.", "green")

        # Only take <=num_candidates top candidates. Sort top_candidates by the mean score. If mean score for some candidate is None, use 0 to sort.
        top_candidates.sort(key=lambda x: x.mean_score() if x.mean_score() is not None else 0, reverse=True)
        top_candidates = top_candidates[:self.num_candidates]

        # Remove the top candidates from the memory. Do the logging stuff like PS.
        priorities = []
        items_to_remove = []
        for neg_priority, candidate in self.memory.memory:
            if candidate in top_candidates:
                priorities.append(-neg_priority)
                items_to_remove.append((neg_priority, candidate))
        assert len(items_to_remove) == len(top_candidates), "The number of removed candidates should be equal to the number of top_candidates."
        # It may be safer to remove items after traversing the memory.
        for item in items_to_remove:
            self.memory.memory.remove(item)
        heapq.heapify(self.memory.memory)

        
        mean_scores = [c.mean_score() for c in top_candidates]
        mean_scores = [s for s in mean_scores if s is not None]  # filter out None scores
        depths = [c.depth for c in top_candidates]
        assert all(depth is not None for depth in depths), "All exploration candidates must have a depth."
        assert all(depth >= 1 for depth in depths), "All exploration candidates must have a depth of at least 1."
        info_dict = {
            'num_exploration_candidates': len(top_candidates),
            'exploration_candidates_mean_priority': safe_mean(priorities),  # list of priorities of the exploration candidates
            'exploration_candidates_mean_score': safe_mean(mean_scores),  # list of mean scores of the exploration candidates
            'exploration_candidates_mean_depth': safe_mean(depths),  # list of depths of the exploration candidates
            'exploration_candidates_average_num_rollouts': safe_mean([c.num_rollouts for c in top_candidates]),
        }
        if len(top_candidates) < self.num_candidates:
            new_num_batches = int(self.default_num_batches * self.num_candidates/len(top_candidates))
            print(f'Setting sampler num_batches from {self.default_num_batches} to {new_num_batches} to accommodate {self.num_candidates} exploration candidates request using {len(top_candidates)} candidates.')
            self.set_sampler_batch_size(self.default_batch_size, new_num_batches)
        else:
            self.set_sampler_batch_size(self.default_batch_size, self.default_num_batches)
        return top_candidates, priorities, info_dict
       


        
   