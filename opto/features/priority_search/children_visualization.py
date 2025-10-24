import numpy as np
import copy
import sys
import os
from typing import Union, List, Tuple, Dict, Any, Optional
from opto.features.priority_search.search_template import Samples, SearchTemplate, BatchRollout
from opto.features.priority_search.regressor import LogisticRegressor, LinearRegressor, LinearUCBRegressor, LLMRegressor
# import pretrained regressors
from opto.optimizers.utils import print_color
from opto.trainer.utils import safe_mean
import matplotlib.pyplot as plt
try:
    import litellm
except ImportError:
    litellm = None


from opto.features.priority_search.priority_search import PrioritySearch, ModuleCandidate, HeapMemory
from opto.features.priority_search.priority_search_with_regressor import PrioritySearch_with_Regressor
import heapq
from opto.trainer.loader import DataLoader
from opto.features.priority_search.sampler import Sampler, BatchRollout


class ChildrenVisualization(PrioritySearch_with_Regressor):
    """Visualize the children of the initial candidates.
    
    Make a very simple version of the search algorithm. Always forward the initial candidate 
    on different mini-batches multiple times. For one forward pass, we can generate multiple 
    proposals. Finally store the embeddings of the initial candidates and the children, in a 
    way that candidates from the same mini-batch are put together, by setting the first column 
    of the numpy array to be the index of the mini-batch.
    
    This class overrides the explore method to always use the base agent candidate, ensuring
    that all generated candidates are children of the same parent. The visualization shows
    the t-SNE embedding of all candidates with the base agent highlighted as the parent.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize base_agent_ModuleCandidate early for use in explore method
        self.base_agent_ModuleCandidate = ModuleCandidate(self.agent, optimizer=self.optimizer)
        
    def train(self, *args, **kwargs):
        """ Train the agent. """
        print_color(f"Visualizing the children of the base agent...", "green")
        # Don't create a new base_agent_ModuleCandidate - use the one from parent class
        # self.base_agent_ModuleCandidate will be set by the parent class in super().train()
        super().train(*args, **kwargs)
        self.regressor.predict_scores(self.memory.memory)
        # save the embeddings of all candidates in the memory to a numpy array
        self.embeddings_array = np.array([c.embedding for _,c in self.memory.memory])
        # save the embedding array
        np.save('embeddings_array.npy', self.embeddings_array)
        print_color(f"Saved embeddings array to embeddings_array.npy", "green")
        # print the number of embeddings    
        print_color(f"Number of embeddings: {len(self.embeddings_array)}", "green")


        # Check if we have enough samples for t-SNE visualization
        n_samples = len(self.embeddings_array)
        if n_samples < 2:
            print_color(f"Warning: Only {n_samples} sample(s) available. Skipping t-SNE visualization (minimum 2 required).", "yellow")
            return
        
        # Check for identical embeddings that can cause t-SNE issues
        unique_embeddings = np.unique(self.embeddings_array, axis=0)
        n_unique = len(unique_embeddings)

        print_color(f"Number of unique embeddings: {n_unique}", "green")
        if n_unique < 2:
            print_color(f"Warning: Only {n_unique} unique embedding(s) found. Using simple scatter plot instead of t-SNE.", "yellow")
            # Create a simple 2D projection for visualization
            if self.embeddings_array.shape[1] >= 2:
                X_tsne = self.embeddings_array[:, :2]  # Use first 2 dimensions
            else:
                # If embeddings are 1D, create a 2D plot with y=0
                X_tsne = np.column_stack([self.embeddings_array[:, 0], np.zeros(n_samples)])
            
            # Create simple visualization for identical embeddings
            self._create_simple_visualization(X_tsne, "Raw Embedding")
            return
        else:
            # Run multiple visualization methods for robustness analysis
            self._run_multiple_visualizations(n_samples)
            return  # Exit early since we're handling all visualization in the helper method
    
    def _run_multiple_visualizations(self, n_samples):
        """Run multiple visualization methods for robustness analysis."""
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        
        # Try to import UMAP (optional dependency)
        try:
            import umap  # type: ignore  # Optional dependency
            umap_available = True
        except ImportError:
            umap_available = False
            print_color("UMAP not available. Install with: pip install umap-learn", "yellow")
        
        # Adjust perplexity based on number of samples
        perplexity = min(100.0, max(1.0, n_samples - 1))
        
        # Test different perplexity values if we have enough samples
        perplexity_values = [perplexity]
        if n_samples > 10:
            perplexity_values = [max(1.0, n_samples//4), perplexity, min(50.0, n_samples-1)]
        
        print_color(f"Running stability analysis with {len(perplexity_values)} perplexity values: {perplexity_values}", "blue")
        
        # Create a comprehensive comparison plot
        n_methods = 2 + (1 if umap_available else 0)  # t-SNE multiple runs + PCA + UMAP (if available)
        n_tsne_runs = 5  # Number of t-SNE runs with different seeds
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        plot_idx = 0
        
        # 1. Multiple t-SNE runs with same perplexity, different seeds
        print_color(f"Running {n_tsne_runs} t-SNE runs with different random seeds...", "blue")
        tsne_results = []
        
        for seed in [42, 123, 456, 789, 999]:
            try:
                tsne = TSNE(n_components=2, init='pca', random_state=seed, perplexity=perplexity)
                X_tsne = tsne.fit_transform(self.embeddings_array)
                tsne_results.append(X_tsne)
                
                if plot_idx < 6:
                    self._create_single_plot(axes[plot_idx], X_tsne, f't-SNE (seed={seed}, perp={perplexity:.1f})')
                    plot_idx += 1
                    
            except Exception as e:
                print_color(f"t-SNE with seed {seed} failed: {e}", "yellow")
        
        # 2. t-SNE with different perplexity values (if we have multiple)
        if len(perplexity_values) > 1:
            print_color(f"Testing different perplexity values: {perplexity_values}", "blue")
            for perp in perplexity_values[1:]:  # Skip the first one (already done above)
                if plot_idx >= 6:
                    break
                try:
                    tsne = TSNE(n_components=2, init='pca', random_state=42, perplexity=perp)
                    X_tsne = tsne.fit_transform(self.embeddings_array)
                    self._create_single_plot(axes[plot_idx], X_tsne, f't-SNE (perp={perp:.1f})')
                    plot_idx += 1
                except Exception as e:
                    print_color(f"t-SNE with perplexity {perp} failed: {e}", "yellow")
        
        # 3. PCA as baseline
        if plot_idx < 6:
            print_color("Running PCA as baseline...", "blue")
            try:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(self.embeddings_array)
                explained_var = pca.explained_variance_ratio_
                self._create_single_plot(axes[plot_idx], X_pca, 
                                       f'PCA (var: {explained_var[0]:.2f}, {explained_var[1]:.2f})')
                plot_idx += 1
            except Exception as e:
                print_color(f"PCA failed: {e}", "yellow")
        
        # 4. UMAP if available
        if umap_available and plot_idx < 6:
            print_color("Running UMAP...", "blue")
            try:
                # Adjust n_neighbors for small datasets
                n_neighbors = min(15, max(2, n_samples - 1))
                reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors)
                X_umap = reducer.fit_transform(self.embeddings_array)
                self._create_single_plot(axes[plot_idx], X_umap, f'UMAP (neighbors={n_neighbors})')
                plot_idx += 1
            except Exception as e:
                print_color(f"UMAP failed: {e}", "yellow")
        
        # Hide unused subplots
        for i in range(plot_idx, 6):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Embedding Visualization Stability Analysis\n({n_samples} candidates, {len(tsne_results)} successful t-SNE runs)', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('embeddings_stability_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print_color(f"Saved stability analysis with {plot_idx} visualizations to 'embeddings_stability_analysis.png'", "green")
        
        # Analyze t-SNE stability if we have multiple runs
        if len(tsne_results) >= 2:
            self._analyze_tsne_stability(tsne_results)
    
    def _create_single_plot(self, ax, X_embedded, title):
        """Create a single visualization plot."""
        # Find the index of the base agent in the memory to highlight it
        base_agent_index = None
        for i, (_, candidate) in enumerate(self.memory.memory):
            if candidate == self.base_agent_ModuleCandidate:
                base_agent_index = i
                break
        
        # Plot all embeddings as children (blue dots)
        ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c='lightblue', alpha=0.6, s=50, label='Children candidates')
        
        # Highlight the base agent (parent) if found in memory
        if base_agent_index is not None:
            ax.scatter(X_embedded[base_agent_index, 0], X_embedded[base_agent_index, 1], 
                      c='red', s=200, marker='*', edgecolors='black', linewidth=2, 
                      label='Base agent (parent)', zorder=5)
        
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _analyze_tsne_stability(self, tsne_results):
        """Analyze the stability of t-SNE results across different runs."""
        print_color("Analyzing t-SNE stability across runs...", "blue")
        
        # Calculate pairwise distances between runs using Procrustes analysis
        from scipy.spatial.distance import pdist, squareform
        from scipy.spatial import procrustes
        
        n_runs = len(tsne_results)
        stability_scores = []
        
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                try:
                    # Align the two embeddings using Procrustes analysis
                    _, aligned_j, disparity = procrustes(tsne_results[i], tsne_results[j])
                    stability_scores.append(1 - disparity)  # Higher score = more stable
                except Exception as e:
                    print_color(f"Procrustes analysis failed for runs {i} and {j}: {e}", "yellow")
        
        if stability_scores:
            mean_stability = np.mean(stability_scores)
            std_stability = np.std(stability_scores)
            
            print_color(f"t-SNE Stability Analysis:", "green")
            print_color(f"  Mean stability score: {mean_stability:.3f} ± {std_stability:.3f}", "green")
            print_color(f"  Range: [{min(stability_scores):.3f}, {max(stability_scores):.3f}]", "green")
            
            if mean_stability > 0.8:
                print_color("  ✓ High stability - clusters are consistent across runs", "green")
            elif mean_stability > 0.6:
                print_color("  ⚠ Moderate stability - some variation in cluster positions", "yellow")
            else:
                print_color("  ⚠ Low stability - consider using UMAP or increasing sample size", "yellow")
        else:
            print_color("Could not compute stability scores", "yellow")
    
    def _create_simple_visualization(self, X_embedded, viz_method):
        """Create a simple single visualization."""
        # Find the index of the base agent in the memory to highlight it
        base_agent_index = None
        for i, (_, candidate) in enumerate(self.memory.memory):
            if candidate == self.base_agent_ModuleCandidate:
                base_agent_index = i
                print_color(f"Base agent found at index {base_agent_index}", "green")
                break
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        
        # Plot all embeddings as children (blue dots)
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c='lightblue', alpha=0.6, s=50, label='Children candidates')
        
        # Highlight the base agent (parent) if found in memory
        if base_agent_index is not None:
            plt.scatter(X_embedded[base_agent_index, 0], X_embedded[base_agent_index, 1], 
                       c='red', s=200, marker='*', edgecolors='black', linewidth=2, 
                       label='Base agent (parent)', zorder=5)
        
        plt.title(f'{viz_method} Visualization of Candidate Embeddings\n(Red star shows the base agent - parent of all children)')
        plt.xlabel(f'{viz_method} Component 1')
        plt.ylabel(f'{viz_method} Component 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('embeddings_tsne.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print_color(f"Saved {viz_method} visualization with {len(self.embeddings_array)} candidates to 'embeddings_tsne.png'", "green")

    def explore(self, verbose: bool = False, **kwargs):
        """ Overwrite the explore method, always pulling the original candidate.
        
        This method always returns the base agent candidate to ensure we're always
        exploring from the same starting point, generating children from the same parent.
        
        Args:
            verbose (bool): Whether to print verbose output
            **kwargs: Additional keyword arguments
            
        Returns:
            tuple: (top_candidates, priorities, info_dict)
                - top_candidates: List containing only the base agent candidate
                - priorities: List containing the priority of the base agent
                - info_dict: Dictionary with exploration statistics
        """
        print_color(f"--- Always using the base agent candidate for exploration...", "blue") if verbose else None
        
        # Always return the base agent candidate
        top_candidates = [self.base_agent_ModuleCandidate]* self.num_candidates
        
        # Get the priority of the base agent
        base_priority = self.compute_exploration_priority(self.base_agent_ModuleCandidate)
        priorities = [base_priority] * self.num_candidates
        
        # Create info dictionary for logging
        info_dict = {
            'num_exploration_candidates': 1,
            'exploration_candidates_mean_priority': base_priority,
            'exploration_candidates_mean_score': self.base_agent_ModuleCandidate.mean_score() or 0.0,
            'exploration_candidates_average_num_rollouts': self.base_agent_ModuleCandidate.num_rollouts,
        }
        
        if verbose:
            print_color(f"Base agent priority: {base_priority:.4f}", "blue")
            print_color(f"Base agent mean score: {self.base_agent_ModuleCandidate.mean_score()}", "blue")
            print_color(f"Base agent num rollouts: {self.base_agent_ModuleCandidate.num_rollouts}", "blue")
        
        return top_candidates, priorities, info_dict