from opto.optimizers.utils import print_color
from opto.utils.llm import LLM # For the selector LLM
import random
import re


def get_trajectory_of_one_rollout(rollout):
    """
    Convert a rollout into a structured markdown trajectory for optimization.

    This function extracts the trainable parameters and formats the trajectory 
    to guide the optimizer in improving the module's performance.

    Parameters
    ----------
    rollout : dict
        A rollout dictionary containing:
        - 'module': trace.Module - the agent module with trainable parameters
        - 'x': Any - the input data
        - 'info': Any - additional information about the input
        - 'target': Any - the generated output
        - 'score': float - evaluation score (0 = failed, 1 = success)
        - 'feedback': Any - detailed feedback from the evaluation

    Returns
    -------
    str
        A markdown-formatted trajectory string for optimizer guidance.
    """
    assert rollout['module'] is not None, "rollout['module'] is None."
    assert rollout['x'] is not None, "rollout['x'] is None."
    assert rollout['target'] is not None, "rollout['target'] is None."
    assert rollout['score'] is not None, "rollout['score'] is None."
    assert rollout['feedback'] is not None, "rollout['feedback'] is None."
    
    # Extract trainable parameters
    # For debug
    # print_color(f"Rollout feedback: {rollout['feedback']}", "yellow")
    parameters = rollout['module'].parameters()
    parameters_dict = {p.py_name: p.data for p in parameters}
    
    # Construct structured markdown trajectory
#     trajectory = f"""## Task Trajectory

# ## Module Parameters
# {parameters_dict}

# ## Input 
# {rollout['x']}

# ## Output
# {rollout['target']}

# ## Result
# - **Score:** {rollout['score']} 
# - **Feedback:** {rollout['feedback']}

# ## Optimization Note
# Analyze what parameter patterns lead to successful vs. failed outputs.
# """
    trajectory = f"""## Task Trajectory

## Best circles (for this centers configuration)
{rollout['target']}

## Result
- **Score:** {rollout['score']} 
- **Feedback:** {rollout['feedback']}

"""
    return trajectory




class Summarizer:
    """A class which use LLM to summarize the trajectories of the memory. It should be able to learn the patterns of the trajectories. Generate a summary to guide the optimizer to generate better candidates.
    """
    def __init__(self, model_name: str = "gemini/gemini-2.0-flash"):
        self.llm = LLM() # use the default model
        self.max_candidates_in_prompt = 50

    def _get_trajecories_for_memory(self, memory):
        """
        Get trajectories for the memory. Memory is a list of (neg_score, candidate) tuples.
        We first collect rollouts from the each candidate, and then get the trajectories for each rollout.

        Return one single string of all trajectories.
        """
        trajectories = []
        print_color(f"Getting trajectories from {len(memory)} candidates.", "blue")
        # copy a random shuffle of the memory
        memory_with_rollouts = [(neg_score, candidate) for neg_score, candidate in memory if len([rollout for rollout in candidate.rollouts if rollout['score'] is not None]) > 0]
        temporary_memory = random.sample(memory_with_rollouts, k=min(self.max_candidates_in_prompt, len(memory_with_rollouts)))
        for _, candidate in temporary_memory:
            rollouts = [rollout for rollout in candidate.rollouts if rollout['score'] is not None]
            if len(rollouts) == 0:
                continue
            # For each candidate, add one (if exists) successful_rollout and one (if exists) failed_rollout.
            candidate_update_dict = candidate.update_dict.values()
            # print_color(f"Candidate pamameters: {candidate_update_dict}", "blue")# For debugging
            prompt = f"Candidate pamameters: {candidate_update_dict}."
            successful_rollouts = [rollout for rollout in rollouts if rollout['score'] > 0]
            failed_rollouts = [rollout for rollout in rollouts if rollout['score'] == 0]
            if len(successful_rollouts) > 0: 
                random_successful_rollout = random.choice(successful_rollouts)
                prompt += f"\nSuccessful trajectory: {get_trajectory_of_one_rollout(random_successful_rollout)}."
            if len(failed_rollouts) > 0:
                random_failed_rollout = random.choice(failed_rollouts)
                prompt += f"\nFailed trajectory: {get_trajectory_of_one_rollout(random_failed_rollout)}."
            
            trajectories.append(prompt)
        
        print_color(f"Generated trajectories from {len(trajectories)} candidates.", "green")
        return '\n'.join(trajectories)

    def summarize(self, memory) -> str:
        """Summarize the trajectories using the LLM.
        Args:
            memory: The memory containing trajectories to summarize.
        Returns:
            str: The summary.
        """

        history_trajectories = self._get_trajecories_for_memory(memory)

        # print_color(f"History trajectories: {history_trajectories}", "green")

        if len(history_trajectories) == 0:
            return "No successful trajectories found for the memory."
        
        system_prompt = "You are an expert at analyzing agent behavior patterns and providing actionable guidance for parameter optimization."
        
        user_prompt = f"""Analyze the following agent rollout trajectories and extract insights for optimization.

        Trajectories:
        {history_trajectories}

        Provide your analysis in XML format:
        <reasoning>
        Analyze the key patterns and strategies that led to success or failure in these trajectories.
        </reasoning>
        <summary>
        Concrete recommendations for improving output quality based on successful or failed patterns observed in the trajectories.
        </summary>"""

        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]


        # print_color(f"Prompt messages: {prompt_messages}", "yellow")
        response = self.llm(messages=prompt_messages)
        response = response.choices[0].message.content
        # print_color(f"Response: {response}", "cyan")
        # breakpoint()
        # Extract summary using XML regex
        summary_match = re.search(r'<summary>(.*?)</summary>', response, re.DOTALL)

        return summary_match.group(1).strip()