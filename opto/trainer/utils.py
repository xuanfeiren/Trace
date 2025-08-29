import asyncio
import functools
import warnings
from concurrent.futures import ThreadPoolExecutor
from tqdm.asyncio import tqdm_asyncio
from opto.trace.bundle import ALLOW_EXTERNAL_DEPENDENCIES
from opto.trace.modules import Module
from opto.trainer.guide import AutoGuide
from typing import Union, List, Tuple, Dict, Any, Optional
from opto.optimizers.utils import print_color

from opto.trace.nodes import ParameterNode
from black import format_str, FileMode

def async_run(runs, args_list = None, kwargs_list = None, max_workers = None, description = None, allow_sequential_run=True):
    """Run multiple functions in asynchronously.

    Args:
        runs (list): list of functions to run
        args_list (list): list of arguments for each function
        kwargs_list (list): list of keyword arguments for each function
        max_workers (int, optional): maximum number of worker threads to use.
            If None, the default ThreadPoolExecutor behavior is used.
        description (str, optional): description to display in the progress bar.
            This can indicate the current stage (e.g., "Evaluating", "Training", "Optimizing").
        allow_sequential_run (bool, optional): if True, runs the functions sequentially if max_workers is 1.
    """
    # if ALLOW_EXTERNAL_DEPENDENCIES is not False:
    #     warnings.warn(
    #         "Running async_run with external dependencies check enabled. "
    #         "This may lead to false positive errors. "
    #         "If such error happens, call disable_external_dependencies_check(True) before running async_run.",
    #         UserWarning,
    #     )

    if args_list is None:
        args_list = [[]] * len(runs)
    if kwargs_list is None:
        kwargs_list = [{}] * len(runs)

    if (max_workers == 1) and allow_sequential_run: # run without asyncio
        print(f"{description} (Running sequentially).")
        return [run(*args, **kwargs) for run, args, kwargs in zip(runs, args_list, kwargs_list)]
    else: 
        async def _run():
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                tasks = [loop.run_in_executor(executor, functools.partial(run, *args, **kwargs)) 
                        for run, args, kwargs, in zip(runs, args_list, kwargs_list)]
                
                # Use the description in the tqdm progress bar if provided
                if description:
                    return await tqdm_asyncio.gather(*tasks, desc=description)
                else:
                    return await tqdm_asyncio.gather(*tasks)
        return asyncio.run(_run())


def batch_run(max_workers=None, description=None):
    """
    Create a function that runs in parallel using asyncio, with support for batching.
    The batch size is inferred as the length of the longest argument or keyword argument.            

    Args:
        fun (callable): The function to run.
        
        max_workers (int, optional): Maximum number of worker threads to use.
            If None, the default ThreadPoolExecutor behavior is used.
        description (str, optional): Description to display in the progress bar.

    Returns:
        callable: A new function that processes batches of inputs.

    NOTE: 
        If fun takes input that has __len__ (like lists or arrays), they won't be broadcasted. 
        When using batch_run, be sure to pass list of such arguments of the same length.       

    Example:
        >>> @batch_run(max_workers=4, description="Processing batch")
        >>> def my_function(x, y):
        >>>     return x + y
        >>> x = [1, 2, 3, 4, 5]
        >>> y = 10
        >>> outputs = my_function(x, y)
        >>> # outputs will be [11, 12, 13, 14, 15]
        >>> # This will run the function in asynchronously with 4 threads   
    """
    
    def decorator(fun):
        """
        Decorator to create a function that runs in parallel using asyncio, with support for batching.
        
        Args:
            fun (callable): The function to run.
            
            max_workers (int, optional): Maximum number of worker threads to use.
                If None, the default ThreadPoolExecutor behavior is used.
            description (str, optional): Description to display in the progress bar.

        Returns:
            callable: A new function that processes batches of inputs.
        """    
        def _fun(*args, **kwargs):
            
            # We try to infer the batch size from the args
            all_args = args + tuple(kwargs.values())
            # find all list or array-like arguments and use their length as batch size
            batch_size = max(len(arg) for arg in all_args if hasattr(arg, '__len__'))
            # broadcast the batch size to all args and record the indices that are broadcasted
            args = [arg if hasattr(arg, '__len__') else [arg] * batch_size for arg in args]
            kwargs = {k: v if hasattr(v, '__len__') else [v] * batch_size for k, v in kwargs.items()}   

            # assert that all args and kwargs have the same length
            lengths = [len(arg) for arg in args] + [len(v) for v in kwargs.values()]
            if len(set(lengths)) != 1:
                raise ValueError("All arguments and keyword arguments must have the same length.")

            # deepcopy if it is a trace.Module (as they may have mutable state)
            # Module.copy() is used to create a new instance with the same parameters
            _args = [[a.copy() if isinstance(a, (Module, AutoGuide)) else a for a in arg ] for arg in args ]
            _kwargs = {k: [a.copy() if isinstance(a, (Module, AutoGuide)) else a  for a in v ] for k, v in kwargs.items() }

            # Run the forward function in parallel using asyncio with the same parameters. 
            # Since trace.Node is treated as immutable, we can safely use the same instance.
            # The resultant graph will be the same as if we had called the function with the original arguments.

            # convert _args and _kwargs (args, kwargs of list) to lists of args and kwargs

            args_list = [tuple(aa[i] for aa in _args) for i in range(batch_size)]
            kwargs_list = [{k: _kwargs[k][i] for k in _kwargs} for i in range(batch_size)]

            outputs = async_run([fun] * batch_size, args_list=args_list, kwargs_list=kwargs_list,
                                max_workers=max_workers, description=description)
            return outputs

        return _fun

    return decorator

if __name__ == "__main__":

    def tester(t):  # regular time-consuming function
        import time
        print(t)
        time.sleep(t)
        return t, 2

    runs = [tester] * 10  # 10 tasks to demonstrate threading
    args_list = [(3,), (3,), (2,), (3,), (3,), (2,), (2,), (3,), (2,), (3,)]
    kwargs_list = [{}] * 10
    import time
    
    # Example with 1 thread (runs sequentially)
    print("Running with 1 thread (sequential):")
    start = time.time()
    output = async_run(runs, args_list, kwargs_list, max_workers=1)
    print(f"Time with 1 thread: {time.time()-start:.2f} seconds")
    
    # Example with limited workers (2 threads)
    print("\nRunning with 2 threads (parallel):")
    start = time.time()
    output = async_run(runs, args_list, kwargs_list, max_workers=2)
    print(f"Time with 2 threads: {time.time()-start:.2f} seconds")
    
    # Example with limited workers (4 threads)
    print("\nRunning with 4 threads (parallel):")
    start = time.time()
    output = async_run(runs, args_list, kwargs_list, max_workers=4)
    print(f"Time with 4 threads: {time.time()-start:.2f} seconds")
    
    # Example with default number of workers
    print("\nRunning with default number of threads:")
    start = time.time()
    output = async_run(runs, args_list, kwargs_list)
    print(f"Time with default threads: {time.time()-start:.2f} seconds")
import numpy as np
def sample_minibatch( dataset: Dict[str, List[Any]], batch_size: int = None) -> Tuple[List[Any], List[Any]]:
        """Sample a minibatch from the dataset."""
        if not dataset or not dataset.get('inputs') or not dataset.get('infos'):
            print_color("Warning: Attempted to sample from an empty or malformed dataset.", color='yellow')
            return [], []
        if batch_size is None: # return the whole dataset, if batch_size is not provided
            return dataset['inputs'], dataset['infos']
        dataset_size = len(dataset['inputs'])
        if dataset_size == 0:
            print_color("Warning: Dataset is empty, cannot sample minibatch.", color='yellow')
            return [], []

        actual_batch_size = min(batch_size, dataset_size)
        indices = np.random.choice(dataset_size, actual_batch_size, replace=False)
        xs = [dataset['inputs'][i] for i in indices]
        infos = [dataset['infos'][i] for i in indices]
        return xs, infos
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
    import time

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
                raise RuntimeError(f"{operation_name}: Failed after {max_retries} attempts. Error: {e}")
                
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

def evaluate_agent(agent, guide, dataset,min_score=0,num_threads=20,num_eval_times=5):
    """Evaluate an agent."""
    from opto.trainer.evaluators import evaluate
    eval_scores = evaluate(agent,guide, dataset['inputs'],dataset['infos'],
                                        min_score=min_score,
                                        num_threads=num_threads,
                                        num_samples=num_eval_times,
                                        description=f"Evaluating agent")
    # Create table with explicit column names
    if eval_scores.ndim >1:
        columns = [f'Eval_{i+1}' for i in range(eval_scores.shape[1])]
        all_valid_scores = [score for row in eval_scores for score in row if score is not None]
    else:
        all_valid_scores = [score for score in eval_scores if score is not None]
    test_score = np.mean(all_valid_scores) if all_valid_scores else 0
    return test_score

