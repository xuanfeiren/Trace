from opto.optimizers.utils import print_color
from opto.utils.llm import LLM # For the selector LLM
import json
import random
import re

def get_trajectory_from_output(output):
    """Get trajectory from the agent's output."""
    reward, messages = output
    conversation_parts = []
    # assert messages is a list of dicts
    try:
        assert isinstance(messages, list) and all(isinstance(msg, dict) for msg in messages), "messages must be a list of dicts."
    except AssertionError as e:
        print_color(f"Error: {e}", "red")
        print_color(f"messages: {messages}", "blue")
        breakpoint()
        return None
    # delete the first message if it is the system message. That's the wiki prompt.
    if messages[0]['role'] == 'system':
        # print_color(f"Deleting the first message if it is the system message.", "red")
        messages.pop(0)
    # else:
    #     print_color(f"The first message: {messages[0]}", "red")
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
    return '\n'.join(conversation_parts)

def get_trajectory_of_one_rollout(rollout):
    """Get trajectories of one rollout."""
    target = rollout['target']
    # reward, messages = target
    conversation_str = get_trajectory_from_output(target)
    # print two versions of the conversation.
    # print_color(f"Conversation: {conversation_str}", "green")
    # breakpoint()
    return conversation_str

class Summarizer:
    """A class which use LLM to summarize the trajectories of the memory. It should be able to learn the patterns of the trajectories. Generate a summary to guide the optimizer to generate better candidates.
    """
    def __init__(self, model_name: str = "gemini/gemini-2.0-flash"):
        self.llm = LLM(model=model_name)
        self.max_candidates_in_prompt = 20

    def _get_trajecories_for_memory(self, memory):
        """
        Get trajectories for the memory. Memory is a list of (neg_score, candidate) tuples.
        We first collect rollouts from the each candidate, and then get the trajectories for each rollout.

        Return one single string of all trajectories.
        """
        trajectories = []
        # Here we use one heuristic: for each candidate, randomly select one trajectory to put into trajectories.
        print_color(f"Getting trajectories from {len(memory)} candidates.", "blue")
        # copy a random shuffle of the memory
        temporary_memory = random.sample(memory, k=min(self.max_candidates_in_prompt, len(memory)))
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
                prompt += f"Successful trajectory: {get_trajectory_of_one_rollout(random_successful_rollout)}."
            if len(failed_rollouts) > 0:
                random_failed_rollout = random.choice(failed_rollouts)
                prompt += f"Failed trajectory: {get_trajectory_of_one_rollout(random_failed_rollout)}."
            
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
        
        user_prompt = f"""Analyze the following agent conversation trajectories and extract insights for optimization.

        Trajectories:
        {history_trajectories}

        Provide your analysis in JSON format:
        1. First, reason about what made these trajectories successful or failed
        2. Then, provide concrete guidance for the optimizer

        Output format:
        {{
            "reasoning": "Analyze the key patterns and strategies that led to success or failure in these trajectories",
            "summary": "Concrete recommendations for generating better agent parameters based on successful or failed patterns observed in the trajectories"
        }}"""

        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response_format = {"type": "json_object"}
        # print_color(f"History trajectories: {history_trajectories}", "blue")
        # print_color(f"Prompt messages: {prompt_messages}", "blue")
        response = self.llm(prompt_messages, response_format=response_format)

        response = response.choices[0].message.content
        # print_color(f"Response: {response}", "yellow")
        
        # Extract summary field directly using regex, avoiding JSON parsing issues
        summary_match = re.search(r'"summary"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', response, re.DOTALL)
        
        if summary_match:
            summary = summary_match.group(1)
            # Unescape basic JSON escape sequences if needed
            summary = summary.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
            return str(summary)
        else:
            # Fallback to JSON parsing if regex doesn't match
            try:
                summary_json = json.loads(response)
                summary = summary_json.get('summary', '')
                # Handle both string and array formats
                if isinstance(summary, list):
                    summary = '\n'.join(str(item) for item in summary)
                return str(summary)
            except Exception as e:
                print_color(f"Unable to extract summary from response: {e}", "red")
                print_color(f"Response: {response}", "blue")
                return "Unable to extract summary from LLM response."