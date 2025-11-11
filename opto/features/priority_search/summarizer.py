from opto.optimizers.utils import print_color
from opto.utils.llm import LLM # For the selector LLM
import json
import random

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
        messages.pop(0)
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

def get_trajectories_of_one_rollout(rollout):
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

    def _get_trajecories_for_memory(self, memory):
        """
        Get trajectories for the memory. Memory is a list of (neg_score, candidate) tuples.
        We first collect rollouts from the each candidate, and then get the trajectories for each rollout.

        Return one single string of all trajectories.
        """
        trajectories = []
        # Here we use one heuristic: for each candidate, randomly select one trajectory to put into trajectories.
        print_color(f"Getting trajectories from {len(memory)} candidates.", "blue")
        for _, candidate in memory:
            rollouts = candidate.rollouts
            # only learn from successful rollouts.
            successful_rollouts = [rollout for rollout in rollouts if rollout['score'] > 0]
            if len(successful_rollouts) == 0:
                continue
            random_rollout = random.choice(successful_rollouts)
            trajectories.append(get_trajectories_of_one_rollout(random_rollout))
        
        print_color(f"Generated {len(trajectories)} trajectories.", "green")

        # only use the first 10 trajectories.
        trajectories = trajectories[:10]
        
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
        
        user_prompt = f"""Analyze the following successful agent conversation trajectories and extract insights for optimization.

        Trajectories:
        {history_trajectories}

        Provide your analysis in JSON format:
        1. First, reason about what made these trajectories successful
        2. Then, provide concrete guidance for the optimizer

        Output format:
        {{
            "reasoning": "Analyze the key patterns and strategies that led to success in these trajectories",
            "summary": "[Concrete recommendations for generating better agent parameters based on successful patterns observed in the trajectories]"
        }}"""

        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response_format = {"type": "json_object"}
        # print_color(f"Prompt messages: {prompt_messages}", "blue")
        response = self.llm(prompt_messages, response_format=response_format)

        response = response.choices[0].message.content
        # print_color(f"Response: {response}", "yellow")

        summary_json = json.loads(response)
        summary = summary_json['summary']
        
        return str(summary)