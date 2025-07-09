# Repeat the agent on the first task multiple times until it succeeds
from tau_bench.envs import get_env
from tau_bench.types import RunConfig


from opto import trace
from opto.optimizers import OptoPrime 
from opto.trace.nodes import GRAPH
from opto.trace.modules import Module 

# Copyright Sierra

import json
from litellm import completion
from typing import List, Optional, Dict, Any
import argparse

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME
from opto.trainer.guide import AutoGuide

import litellm 
litellm.drop_params = True
import numpy as np

from typing import  List, Tuple, Dict, Any, Optional
from opto import trace
from opto.trainer.utils import async_run # Assuming print_color is in utils
from opto.trainer.algorithms.basic_algorithms import batchify # evaluate and batchify might be useful

import json

from black import format_str, FileMode

import time

@trace.model
class ToolCallingAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
    ):
        super().__init__()
        self.tools_info = trace.node(tools_info, trainable=True)
        self.wiki = wiki
        self.additional_instructions = trace.node("Here are the additional instructions to help the agent solve the task: ", trainable=True)
        self.model = model
        self.provider = provider
        self.temperature = temperature

    @trace.bundle()
    def solve(self, tools_info, additional_instructions, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30):
        """Agent solves the task with the given tools_info."""
        total_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "system", "content": additional_instructions},
            {"role": "user", "content": obs},
        ]
        
        for step in range(max_num_steps):
            completion_kwargs = {
                "messages": messages,
                "model": self.model,
                "custom_llm_provider": self.provider,
                "tools": tools_info,
                "temperature": self.temperature,
            }
            
            # Retry logic with exponential backoff for entire interaction
            max_retries = 10
            base_delay = 1.0
            step_successful = False
            
            for retry_attempt in range(max_retries):
                try:
                    # Step 1: Get completion from API
                    res = completion(**completion_kwargs)
                    
                    # Step 2: Process the response
                    next_message = res.choices[0].message.model_dump()
                    if retry_attempt >= 1: #debug
                        print("Completion succeeded.")
                    cost = res._hidden_params.get("response_cost")
                    if cost is not None:
                        total_cost += cost
                    
                    # Step 3: Convert message to action
                    action = message_to_action(next_message)
                    
                    # Step 4: Execute action in environment
                    env_response = env.step(action)
                    
                    # If we get here, everything succeeded
                    step_successful = True
                    break
                    
                except Exception as e:
                    # print(f"Step {step}: Error: {e}, tring to retry...")
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
                        print(f"Step {step}: Failed after {max_retries} attempts. Error: {e}")
                        break
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
                        print(f"Step {step}: {error_type_desc} - Retry {retry_attempt + 1}/{max_retries} after {delay:.1f}s. Error: {e}")
                        time.sleep(delay)
                    else:
                        # Non-retryable error
                        print(f"Step {step}: Non-retryable error: {e}")
                        return 0, [], {}
            
            if not step_successful:
                print(f"Step {step}: Skipping step due to interaction failure")
            else:
                # Only process results if step was successful
                reward = env_response.reward
                info = {**info, **env_response.info.model_dump()}
                
                if action.name != RESPOND_ACTION_NAME:
                    next_message["tool_calls"] = next_message["tool_calls"][:1]
                    messages.extend([
                        next_message,
                        {
                            "role": "tool",
                            "tool_call_id": next_message["tool_calls"][0]["id"],
                            "name": next_message["tool_calls"][0]["function"]["name"],
                            "content": env_response.observation,
                        },
                    ])
                else:
                    messages.extend([
                        next_message,
                        {"role": "user", "content": env_response.observation},
                    ])
                    
                if env_response.done:
                    break
                
        result = SolveResult(reward=reward, info=info, messages=messages, total_cost=total_cost)
        
        if result.reward == 1:
            return result.reward, "Correct", "Correct"
        else:
            return result.reward, result.messages, result.info
        
    def forward(self, task_input):
        """Forward pass of the agent for trainer compatibility."""
        env = getattr(self, '_env', None)
        if env is None:
            raise ValueError("Environment not set. Call set_env() before forward pass.")
        
        return self.solve(self.tools_info, self.additional_instructions, env, task_input)
    
    def set_env(self, env):
        """Set the environment for this agent."""
        self._env = env

def message_to_action(message: Dict[str, Any]) -> Action:
    """Convert message to action."""
    if "tool_calls" in message and message["tool_calls"] is not None and len(message["tool_calls"]) > 0 and message["tool_calls"][0]["function"] is not None:
        tool_call = message["tool_calls"][0]
        return Action(
            name=tool_call["function"]["name"],
            kwargs=json.loads(tool_call["function"]["arguments"]),
        )
    else:
        return Action(name=RESPOND_ACTION_NAME, kwargs={"content": message["content"]})

class TeacherGuide(AutoGuide):
    """Guide that extract reward and feedback from the agent's output."""
    def __init__(self, env: Env, config: RunConfig):
        super().__init__()
        self.env = env
        self.config = config
        
    def get_feedback(self, task, output: SolveResult, info):   
        """Get feedback from the agent's output."""
        reward, messages, info = output
        if reward == 1:
            feedback = "Correct"
        else:
            conversation_parts = []
            for msg in messages:
                msg_str = f"{msg['role']}: {msg.get('content', '')}"
                
                if 'tool_calls' in msg and msg['tool_calls']:
                    tool_calls_str = []
                    for tool_call in msg['tool_calls']:
                        if 'function' in tool_call:
                            func_name = tool_call['function'].get('name', '')
                            func_args = tool_call['function'].get('arguments', '')
                            tool_calls_str.append(f"Tool: {func_name}({func_args})")
                    if tool_calls_str:
                        msg_str += f" [Tool Calls: {'; '.join(tool_calls_str)}]"
                
                if msg['role'] == 'tool':
                    tool_name = msg.get('name', '')
                    tool_call_id = msg.get('tool_call_id', '')
                    msg_str = f"tool ({tool_name}, ID: {tool_call_id}): {msg.get('content', '')}"
                
                conversation_parts.append(msg_str)
            
            feedback = "The agent failed to solve the task. Here is the conversation history: " + "\n".join(conversation_parts)
        return reward, feedback
        
    def metric(self, task, output: SolveResult, info):
        """Metric for the agent's performance."""
        reward, messages, info = output
        return reward

def create_retail_dataset(env, num_tasks=10):
    """Create dataset from retail environment tasks."""
    inputs = []
    infos = []
    
    for task_id in range(num_tasks):
        inputs.append(task_id)
        infos.append(task_id)
    
    return {'inputs': inputs, 'infos': infos}

from opto.trainer.evaluators import evaluate
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
def main():
    parser = argparse.ArgumentParser(description='Train agent using search algorithms')
    parser.add_argument('--num_train_samples', type=int, default=50,
                       help='Number of test samples')
    
    parser.add_argument('--num_validate_samples', type=int, default=50,
                       help='Number of test samples')
    parser.add_argument('--num_test_samples', type=int, default=50,
                       help='Number of test samples')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='gemini-2.0-flash',
                       help='Model to use for the agent')
    parser.add_argument('--user_model', type=str, default='gemini-2.0-flash',
                       help='Model to use for the user')
    args = parser.parse_args()
    config = RunConfig(
            model_provider="gemini",
            user_model_provider="gemini",
            model=args.model,
            user_model=args.user_model,
            num_trials=1,
            env="retail",
            agent_strategy="tool-calling",
            temperature=0.0,
            task_split="test",
            task_ids=list(range(max(args.num_train_samples, args.num_validate_samples, args.num_test_samples))),
            log_dir="results",
            max_concurrency=1,
            seed=10,
            shuffle=0,
            user_strategy="llm",
            few_shot_displays_path=None
        )
        
    env = get_env(config.env,user_strategy=config.user_strategy,user_model=config.user_model,user_provider=config.user_model_provider,task_split=config.task_split,task_index=0)
    test_dataset = create_retail_dataset(env, num_tasks=args.num_test_samples)
    
    
    agent = ToolCallingAgent(
            tools_info=env.tools_info,
            wiki=env.wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature
        )
    agent.set_env(env)
        
    guide = TeacherGuide(env, config)
    eval_xs, eval_infos = test_dataset['inputs'], test_dataset['infos']
    num_eval_times = 1

    evaluate(agent,guide,eval_xs,eval_infos,num_samples=num_eval_times,num_threads=10,description=f"Evaluating candidate")
if __name__ == "__main__":
    main() 