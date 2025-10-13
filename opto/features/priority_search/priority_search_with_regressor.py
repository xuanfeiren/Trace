import numpy as np
import copy
import sys
import os
from typing import Union, List, Tuple, Dict, Any, Optional
from opto.features.priority_search.search_template import Samples, SearchTemplate, BatchRollout
from opto.features.priority_search.regressor import LogisticRegressor, LinearRegressor, LinearUCBRegressor, LLMRegressor
# import pretrained regressors
from my_processing_agents.pretained_regressor import PretrainedLinearRegressor, PretrainedLogisticRegressor
from opto.optimizers.utils import print_color

# Add the project root to Python path to enable imports from my_processing_agents
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '../../../..')  # Go up 4 levels to reach tau-bench root
project_root = os.path.abspath(project_root)  # Resolve the absolute path
sys.path.insert(0, project_root)

from opto.features.priority_search.priority_search import PrioritySearch, ModuleCandidate, HeapMemory
import heapq

class PrioritySearch_with_Regressor(PrioritySearch):
    """
    A subclass of PrioritySearch that uses a regressor to predict the scores of the candidates.
    """

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
              memory_update_frequency: Optional[int] = 0,  # number of iterations to keep the candidates in the short-term memory before merging them into the long-term memory. 0 means only long-term memory is used.
              score_function: str = 'mean',  # function to compute the score for the candidates; 'mean' or 'ucb'
              ucb_exploration_constant: float = 1.0,  # exploration constant for UCB score function
              # Regressor specific parameters
              decouple_optimizers: bool = True,  # whether to decouple the optimizers for each candidate; if True, each candidate will have its own optimizer instance; if False, all candidates share the same optimizer instance.
              regressor_type: str = 'logistic',  # type of the regressor; 'logistic' or 'linear' or 'linear_ucb' or 'llm'
              regressor_model_name: str = "gemini/gemini-2.0-flash",  # model name for the regressor
              regressor_embedding_model: str = "gemini/text-embedding-004",  # embedding model for the regressor
              regressor_learning_rate: float = 0.2,  # learning rate for the regressor
              regressor_regularization_strength: float = 1,  # L2 regularization strength for the regressor
              regressor_max_iterations: int = 20000,  # maximum iterations for regressor training
              regressor_tolerance: float = 1e-4,  # convergence tolerance for the regressor
              regressor_alpha: float = 1.0,  # UCB exploration parameter for the regressor
              regressor_transformation_exploration_factor: float = 0.0,  # transformation exploration factor for linear regressors (0: no transformation, 1: maximum exploration)
              regressor_projection_dim: int = None,  # projection dimension for the regressor
              regressor_rich_text: bool = True,  # whether to use rich text with problem definition for embeddings
              use_validation = False, # whether to validate new proposals with the validation set
              # Additional keyword arguments
              **kwargs
              ):
        """ Train the agent using the Priority Search algorithm with regressor.

        This extends the parent PrioritySearch by adding a regressor that predicts
        candidate scores based on the long-term memory.

        Args:
            All parameters from the parent PrioritySearch.train() method, plus:
            regressor_type (str, optional): Type of the regressor; 'logistic' or 'linear' or 'linear_ucb' or 'llm'. Defaults to 'logistic'.
            regressor_model_name (str, optional): Model name for the regressor. Defaults to "gemini/gemini-2.0-flash".
            regressor_embedding_model (str, optional): Embedding model for the regressor. Defaults to "gemini/text-embedding-004".
            regressor_learning_rate (float, optional): Learning rate for the regressor. Defaults to 0.2.
            regressor_regularization_strength (float, optional): L2 regularization strength for the regressor. Defaults to 1e-4.
            regressor_max_iterations (int, optional): Maximum iterations for regressor training. Defaults to 20000.
            regressor_tolerance (float, optional): Convergence tolerance for the regressor. Defaults to 5e-3.
            regressor_alpha (float, optional): UCB exploration parameter for the regressor. Defaults to 1.0.
            regressor_transformation_exploration_factor (float, optional): Transformation exploration factor for linear regressors. 0 means no transformation ([0,1] -> [0,1]), 1 means maximum exploration ([0,1] -> [-1,0]). Defaults to 0.0.
            regressor_projection_dim (int, optional): Projection dimension for the regressor. Defaults to None.
            regressor_rich_text (bool, optional): Whether to use rich text with problem definition for embeddings. Defaults to True.
            use_validation (bool, optional): Whether to validate new proposals with the validation set. Defaults to False.
        """

        # Initialize the search parameters and memory
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
            decouple_optimizers=decouple_optimizers
        )
        self._enforce_using_data_collecting_candidates = False
        self.use_validation = use_validation
        self.regressor_type = regressor_type
        self.highest_predicted_score = 0
        self.base_agent_ModuleCandidate = None
        self.base_agent_predicted_score = None
        
        # Initialize the regressor with the long-term memory and custom parameters - this is the only difference from parent class
        if regressor_type == 'logistic':
            self.regressor = LogisticRegressor(
            embedding_model=regressor_embedding_model,
            num_threads=num_threads,
            learning_rate=regressor_learning_rate,
            regularization_strength=regressor_regularization_strength,
            max_iterations=regressor_max_iterations,
            tolerance=regressor_tolerance,
            linear_dim=regressor_projection_dim,
            rich_text=regressor_rich_text
        )
        elif regressor_type == 'linear':
            self.regressor = LinearRegressor(
                embedding_model=regressor_embedding_model,
                num_threads=num_threads,
                regularization_strength=regressor_regularization_strength,
                transformation_exploration_factor=regressor_transformation_exploration_factor,
                linear_dim=regressor_projection_dim,
                rich_text=regressor_rich_text
            )
        elif regressor_type == 'pretrained_linear':
            # Set default paths to the regressor model files if not provided
            regressor_weights_path = os.path.join(project_root, 'regressor_models', 'linear_reg_dim768_reg0.0001_Oct3_weights.npy')
            regressor_bias_path = os.path.join(project_root, 'regressor_models', 'linear_reg_dim768_reg0.0001_Oct3_bias.npy')
            
            # Debug: print paths to verify they're correct
            print(f"Project root: {project_root}")
            print(f"Weights path: {regressor_weights_path}")
            print(f"Bias path: {regressor_bias_path}")
            print(f"Weights file exists: {os.path.exists(regressor_weights_path)}")
            print(f"Bias file exists: {os.path.exists(regressor_bias_path)}")
            
            self.regressor = PretrainedLinearRegressor(
                weights_path=regressor_weights_path,
                bias_path=regressor_bias_path,
                embedding_model=regressor_embedding_model,
                num_threads=num_threads,
                rich_text=regressor_rich_text
            )
        elif regressor_type == 'pretrained_logistic':
            regressor_weights_path = os.path.join(project_root, 'regressor_models', 'logistic_reg_Oct4_weights.npy')
            regressor_bias_path = os.path.join(project_root, 'regressor_models', 'logistic_reg_Oct4_bias.npy')
            
            self.regressor = PretrainedLogisticRegressor(
                weights_path=regressor_weights_path,
                bias_path=regressor_bias_path,
                embedding_model=regressor_embedding_model,
                num_threads=num_threads,
                rich_text=regressor_rich_text
            )
        elif regressor_type == 'linear_ucb':
            self.regressor = LinearUCBRegressor(
                embedding_model=regressor_embedding_model,
                num_threads=num_threads,
                regularization_strength=regressor_regularization_strength,
                alpha=regressor_alpha,
                transformation_exploration_factor=regressor_transformation_exploration_factor,
                linear_dim=regressor_projection_dim,
                rich_text=regressor_rich_text
            )
        elif regressor_type == 'llm':
            self.regressor = LLMRegressor(
                model_name=regressor_model_name,
                num_threads=num_threads,
            )
        else:
            raise ValueError(f"Invalid regressor type: {regressor_type}")
        SearchTemplate.train(self, guide=guide,
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
            # 2. Validate the proposed parameters
            validate_results = self.validate(candidates, samples, verbose=verbose, **kwargs)  # this updates the priority queue
            # 3. Update the priority queue with the validation results
            self.update_memory(validate_results, verbose=verbose, **kwargs)  # samples are provided here in case candidates do not capture full information
        else:  # The first iteration.
            self.base_agent_ModuleCandidate = ModuleCandidate(self.agent, optimizer=self.optimizer)
            max_mem_size = self.memory.size if self.memory.size is not None else float('inf')
            while len(self.memory) < min(max_mem_size, self.num_candidates):
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

    def validate(self,
                 candidates: List[ModuleCandidate],
                 samples: Samples,
                 verbose: bool = False,
                 **kwargs):
        """ 
        Override the validate method. 
        In this version, if use_validation is False, we can only use training data to update arm statistics. No validation is performed.
        If use_validation is True, we use the validation set to update arm statistics. The same as the parent class.
        """
        print("--- Validating candidates...") if verbose else None
        assert isinstance(samples, Samples), "samples must be an instance of Samples."
        exploration_candidates = self._exploration_candidates  # exploration candidates from the previous iteration
        assert self._exploration_candidates is not None, "exploration_candidates must be set before calling validate."

        # The current batch of samples can be used to validate the exploration candidates
        validate_samples = copy.copy(samples)
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

    def update_memory(self, validate_results, verbose: bool = False, **kwargs):
        """ Override the update_memory method. In this subclass, we update the priority of all candidates together. Cannot use the parent class's update_memory method, because now some candidates may not have predicted scores.
        """
        print("--- Updating memory with validation results...") if verbose else None
        for candidate, rollouts in validate_results.items():
            candidate.add_rollouts(rollouts)  # add the rollouts to the
            placeholder_priority = self.max_score
            self.memory.push(placeholder_priority, candidate)

    def update_memory_with_regressor(self, verbose: bool = False, **kwargs):
        """ Update the priority queue with the regressor results.
        This function does not add new candidates to the memory. It only updates the predicted scores of the existing candidates. Then reorder the memory according to the predicted scores.
        """
        print("--- Updating memory with regressor results...") if verbose else None
        # Use all data to update the regressor
        self.regressor.update(self.long_term_memory.memory+self.short_term_memory.memory)
        # Always keep track of the predicted score of the base agent. Ideally this number should converge to the true score of the base agent, when we have more and more data.
        self.regressor.predict_scores([(0, self.base_agent_ModuleCandidate)])
        self.base_agent_predicted_score = self.base_agent_ModuleCandidate.predicted_score
        # Predict the scores for the long-term memory and the short-term memory
        self.regressor.predict_scores(self.long_term_memory.memory)
        self.regressor.predict_scores(self.short_term_memory.memory)
        # update the highest predicted score
        self.highest_predicted_score = max(0, max([candidate.predicted_score for _, candidate in self.long_term_memory.memory+self.short_term_memory.memory]))
        # Reorder both long_term_memory and short_term_memory according to the predicted scores
        # Extract candidates from long_term_memory tuples and reorder by predicted scores
        long_term_candidates_with_scores = [(-candidate.predicted_score, candidate) for _, candidate in self.long_term_memory.memory]
        self.long_term_memory.memory = long_term_candidates_with_scores  # Update the internal list of HeapMemory
        heapq.heapify(self.long_term_memory.memory)  # Heapify based on -score (first element of tuple)
        
        # Extract candidates from short_term_memory tuples and reorder by predicted scores
        short_term_candidates_with_scores = [(-candidate.predicted_score, candidate) for _, candidate in self.short_term_memory.memory]
        self.short_term_memory.memory = short_term_candidates_with_scores  # Update the internal list of HeapMemory
        heapq.heapify(self.short_term_memory.memory)  # Heapify based on -score (first element of tuple)

    def print_memory_stats(self):
        # For debugging, print all candidates: number, mean_score(), num_rollouts, predicted_score. It is better to see an increasing trend in the predicted scores.
        print("--- Printing memory stats...")
        print("Long-term memory:")
        # If len(self.long_term_memory.memory)>40, only print the first 20 and the last 20 candidates
        for i, (neg_predicted_score, candidate) in enumerate(self.long_term_memory.memory):
            if len(self.long_term_memory.memory) <= 40 or i < 20 or i >= len(self.long_term_memory.memory) - 20:
                mean_score = candidate.mean_score()
                mean_score_str = f"{mean_score:.4g}" if mean_score is not None else "None"
                print(f"Candidate {i}, Mean Score: {mean_score_str}, Num Rollouts: {candidate.num_rollouts}, Predicted Score: {-neg_predicted_score}")
        # print("Short-term memory:")
        # for i, (neg_predicted_score, candidate) in enumerate(self.short_term_memory.memory):
        #     print(f"Candidate {i}, Mean Score: {candidate.mean_score()}, Num Rollouts: {candidate.num_rollouts}, Predicted Score: {-neg_predicted_score}")

    # TODO refactor below to reuse scoring
    def compute_exploitation_priority(self, candidate) -> float:
        """ Compute the priority for the candidate based on the predicted score. """
        if not isinstance(candidate, ModuleCandidate):
            raise TypeError("candidate must be an instance of ModuleCandidate.")
        if self.regressor_type == 'linear_ucb':
            # For Linear UCB, we use the predicted score (ucb score) as exploration priority, use the (pessimistic) mean prediction as exploitation priority.
            return candidate.mean_prediction
        else:
            return candidate.predicted_score

from opto.features.priority_search.generator import LLMCandidateGenerator

class PrioritySearch_with_Regressor_and_Generator(PrioritySearch_with_Regressor):
    """
    A subclass of PrioritySearch_with_Regressor that uses a generator to propose new candidates.
    """

    def __init__(self, 
                 generator_model_name: str = 'gemini/gemini-2.0-flash',
                 generator_temperature: float = 0.0,
                 generator_verbose: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        # Initialize the generator
        self.generator = LLMCandidateGenerator(
            model_name=generator_model_name,
            temperature=generator_temperature,
            verbose=generator_verbose,
            max_candidates_in_prompt=20,
            num_threads=self.num_threads
        )
    
    def train(self,
              *args,
              generator_frequency: int = 5,  # frequency of generating new candidates
              generator_attempts: int = 50,  # number of attempts to generate new candidates
              generator_patience: int = 3,  # number of attempts to generate new candidates
              num_generator_candidates: int = 5,  # number of candidates to generate
              score_improvement_threshold: float = 1e-3,  # minimum improvement in predicted score to reset patience
              **kwargs
              ):
        self.generator_frequency = generator_frequency
        self.num_generator_candidates = num_generator_candidates
        self.generator_attempts = generator_attempts
        self.generator_patience = generator_patience
        self.score_improvement_threshold = score_improvement_threshold
        super().train(*args,**kwargs)

    def propose(self,
                samples : Samples,
                verbose : bool = False,
                **kwargs):
        """self.num_generator_candidates is the maximum number of candidates to add to the memory. """
        candidates = super().propose(samples, verbose=verbose, **kwargs)
        # After generating OptoPrime candidates, start to propose candidates using the generator.
        # Always use the exploration candidates as memory to write the prompt for the generator. Ask the generator to propose candidates with higher predicted scores. At each step, generate self.num_generator_candidates candidates, predict scores using regressor, then put them 
        if self.n_iters % self.generator_frequency == 0:
            candidates_from_generator = []
            highest_predicted_score = self._best_candidate_priority
            print_color(f"Highest predicted score at the beginning: {highest_predicted_score}", "green")
            patience = 0
            for attempt in range(self.generator_attempts):
                print_color(f"[{attempt+1}/{self.generator_attempts}] Generating candidates...", "green")
                new_candidates = self.generator.generate_candidates(
                    base_module=self.agent,
                    base_score=self.base_agent_predicted_score,
                    optimizer=self.optimizer,
                    memory=self._exploration_candidates+candidates_from_generator,
                    # use the default number of candidates. 
                    # num_candidates=self.num_generator_candidates
                )
                # Skip this attempt if no candidates were generated
                if not new_candidates:
                    print_color(f"No candidates generated in attempt {attempt+1}, skipping...", "yellow")
                    patience += 1
                    continue
                    
                memory_to_predict = [(0, candidate) for candidate in new_candidates]
                # Predict scores for the new candidates
                self.regressor.predict_scores(memory_to_predict)
                predicted_scores = [candidate.predicted_score for candidate in new_candidates]
                highest_predicted_score_attempt = max(predicted_scores)
                if highest_predicted_score_attempt > highest_predicted_score + self.score_improvement_threshold:
                    patience = 0
                    print_color(f"New highest predicted score: {highest_predicted_score_attempt}", "green")
                    # Add the promising candidates from this attempt. Only add those with higher predicted scores than the current best.
                    good_candidates_attempt = [candidate for candidate in new_candidates if candidate.predicted_score > highest_predicted_score]
                    candidates_from_generator.extend(good_candidates_attempt)
                    highest_predicted_score = highest_predicted_score_attempt
                else:
                    patience += 1
                if patience > self.generator_patience:
                    break
            print_color(f"Generator attempted {attempt+1} times, highest predicted score: {highest_predicted_score}", "green")
            print_color(f"The generator generated {len(candidates_from_generator)} candidates which have predicted scores higher than the current best.", "green")
            # Filter candidates that are better than current best
            # Put parts of new candidates from the generator that are better than the current best into the candidates list. Only add the top self.num_generator_candidates candidates.
            sorted_candidates_from_generator = sorted(candidates_from_generator, key=lambda x: x.predicted_score, reverse=True)
            candidates_from_generator = sorted_candidates_from_generator[:self.num_generator_candidates]

            # Log generator results
            
            self.logger.log('Generator/num_candidates', len(candidates_from_generator), self.n_iters, color='blue')
            if len(candidates_from_generator) > 0:
                print_color(f"Added {len(candidates_from_generator)} new candidates from the generator.", "green")
                self.logger.log('Generator/Base_agent_priority', self.base_agent_predicted_score, self.n_iters, color='blue')
                self.logger.log('Generator/Best_priority_before', self._best_candidate_priority, self.n_iters, color='blue')
                self.logger.log('Generator/Highest_predicted_score_after',highest_predicted_score,self.n_iters, color='blue')
                mean_predicted_score = np.mean([candidate.predicted_score for candidate in candidates_from_generator])
                self.logger.log('Generator/mean_predicted_score', mean_predicted_score, self.n_iters, color='blue')
                # Combine original candidates with good generated candidates. 
                candidates.extend(candidates_from_generator)
            else:
                print_color("No new candidates were generated that exceed the current best score.", "yellow")
            # breakpoint()
        
        return candidates

class PrioritySearch_RG_RejectionSampling(PrioritySearch_with_Regressor_and_Generator):
    """
    A very simple idea. In the propose function, we use
     1. OptoPrime as usual to propose candidates. Make the num_proposals large.
     2. Use the generator to propose a large number of candidates.
     3. Reject the candidates that are not better than the current best.
    """
    def propose_attempt2(self,
                samples : Samples,
                verbose : bool = False,
                **kwargs):
        """Propose candidates with OptoPrime and generator with rejection sampling. """
        # Keep track of the current best predicted score
        current_best_score = self._best_candidate_priority
        # generate candidates with OptoPrime
        candidates_optoprime = PrioritySearch.propose(self, samples, verbose=verbose, **kwargs)
        self.regressor.predict_scores([(0, candidate) for candidate in candidates_optoprime])
        # generate candidates with generator, with the base agent and exploration candidates
        candidates_generator = self.generator.generate_candidates(
                    base_module=self.agent,
                    base_score=self.base_agent_predicted_score,
                    optimizer=self.optimizer,
                    memory=self._exploration_candidates,
                    num_candidates=self.num_generator_candidates
                )
        self.regressor.predict_scores([(0, candidate) for candidate in candidates_generator])
        new_raw_candidates = candidates_optoprime + candidates_generator
        assert len(new_raw_candidates) > 0, "No candidates were generated."
        # Assert that all candidates have predicted scores
        assert all(candidate.predicted_score is not None for candidate in new_raw_candidates), "predicted score must be set for all candidates."
        # rejection sampling
        new_candidates = [candidate for candidate in new_raw_candidates if candidate.predicted_score > current_best_score]
        # Log results
        self.logger.log("Propose/Best score before", current_best_score, self.n_iters, color='blue')
        self.logger.log("Propose/Num of new candidates", len(new_candidates), self.n_iters, color='blue')
        self.logger.log("Propose/Base agent predicted score", self.base_agent_predicted_score, self.n_iters, color='blue')
        if len(candidates_optoprime) > 0:
            highest_predicted_score_optoprime = max([candidate.predicted_score for candidate in candidates_optoprime])
            self.logger.log("Propose/Highest predicted score from OptoPrime", highest_predicted_score_optoprime, self.n_iters, color='blue')
        if len(candidates_generator) > 0:
            highest_predicted_score_generator = max([candidate.predicted_score for candidate in candidates_generator])
            self.logger.log("Propose/Highest predicted score from Generator", highest_predicted_score_generator, self.n_iters, color='blue')
        if len(new_candidates) > 0:
            self.logger.log("Propose/Avg predicted score of new candidates", np.mean([candidate.predicted_score for candidate in new_candidates]), self.n_iters, color='blue')
        return new_candidates

    # propose_attempt3: regressor with rejection sampling
    def propose_attempt3(self,
                samples : Samples,
                verbose : bool = False,
                **kwargs):
        """Propose candidates with OptoPrime and generator with rejection sampling. """
        # generate candidates with OptoPrime
        candidates_optoprime = PrioritySearch.propose(self, samples, verbose=verbose, **kwargs)
        predicted_scores = self.regressor.predict_scores([(0, candidate) for candidate in candidates_optoprime])
        # log statistics of the predicted scores
        self.logger.log("Propose/Best score before", self._best_candidate_priority, self.n_iters, color='blue')
        self.logger.log("Propose/Highest predicted score from OptoPrime", max(predicted_scores), self.n_iters, color='blue')
        self.logger.log("Propose/Lowest predicted score from OptoPrime", min(predicted_scores), self.n_iters, color='blue')
        self.logger.log("Propose/Base agent predicted score", self.base_agent_predicted_score, self.n_iters, color='blue')
        self.logger.log("Propose/Avg predicted score from OptoPrime", np.mean(predicted_scores), self.n_iters, color='blue')
        self.logger.log("Propose/Num of candidates from OptoPrime", len(candidates_optoprime), self.n_iters, color='blue')
        # in this process we generate num_proposals*num_candidates candidates. We do rejection sampling for the best num_candidates candidates.
        
        # sort the candidates by predicted scores
        candidates_optoprime.sort(key=lambda x: x.predicted_score, reverse=True)
        # do rejection sampling for the best num_candidates candidates
        candidates_optoprime = candidates_optoprime[:self.num_candidates]
        # log mean predicted scores after rejection sampling
        self.logger.log("Propose/Avg predicted score after rejection sampling", np.mean([candidate.predicted_score for candidate in candidates_optoprime]), self.n_iters, color='blue')
        self.logger.log("Propose/Num of candidates after rejection sampling", len(candidates_optoprime), self.n_iters, color='blue')
        return candidates_optoprime
    # propose_attempt4: attempt3 + generator
    def propose(self,
                samples : Samples,
                verbose : bool = False,
                **kwargs):
        """Propose candidates with OptoPrime and generator with rejection sampling. """
        # generate candidates with OptoPrime
        candidates_optoprime = PrioritySearch.propose(self, samples, verbose=verbose, **kwargs)
        predicted_scores = self.regressor.predict_scores([(0, candidate) for candidate in candidates_optoprime])
        # log statistics of the predicted scores
        self.logger.log("Propose/Best score before", self._best_candidate_priority, self.n_iters, color='blue')
        self.logger.log("Propose/Highest predicted score from OptoPrime", max(predicted_scores), self.n_iters, color='blue')
        self.logger.log("Propose/Lowest predicted score from OptoPrime", min(predicted_scores), self.n_iters, color='blue')
        self.logger.log("Propose/Base agent predicted score", self.base_agent_predicted_score, self.n_iters, color='blue')
        self.logger.log("Propose/Avg predicted score from OptoPrime", np.mean(predicted_scores), self.n_iters, color='blue')
        self.logger.log("Propose/Num of candidates from OptoPrime", len(candidates_optoprime), self.n_iters, color='blue')
        # in this process we generate num_proposals*num_candidates candidates. We do rejection sampling for the best num_candidates candidates.
        
        # sort the candidates by predicted scores
        candidates_optoprime.sort(key=lambda x: x.predicted_score, reverse=True)
        # do rejection sampling for the best num_candidates candidates
        candidates_optoprime = candidates_optoprime[:self.num_candidates]

        # generate candidates with generator
        candidates_generator = self.generator.generate_candidates(
            base_module=self.agent,
            base_score=self.base_agent_predicted_score,
            optimizer=self.optimizer,
            memory=candidates_optoprime,
            num_candidates=self.num_generator_candidates
        )
        self.regressor.predict_scores([(0, candidate) for candidate in candidates_generator])
        predicted_scores_generator = [candidate.predicted_score for candidate in candidates_generator]
        # log statistics of the predicted scores
        self.logger.log("Propose/Avg predicted score from Generator", np.mean(predicted_scores_generator), self.n_iters, color='blue')
        self.logger.log("Propose/Num of candidates from Generator", len(candidates_generator), self.n_iters, color='blue')

        # log mean predicted scores after rejection sampling
        self.logger.log("Propose/Avg predicted score after rejection sampling", np.mean([candidate.predicted_score for candidate in candidates_optoprime+candidates_generator]), self.n_iters, color='blue')
        self.logger.log("Propose/Num of candidates after rejection sampling", len(candidates_optoprime+candidates_generator), self.n_iters, color='blue')
        return candidates_optoprime+candidates_generator
    
    



           