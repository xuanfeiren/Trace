from opto.optimizers.utils import print_color
from opto.utils.llm import LLM # For the selector LLM
import json
import random

def get_trajectory_from_output(output):
    """Get trajectory from the agent's output."""
    reward, messages, info = output
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
    return '\n'.join(conversation_parts)

def get_trajectories_of_one_rollout(rollout):
    """Get trajectories of one rollout."""
    target = rollout['target']
    reward, messages, info = target
    conversation_str = get_trajectory_from_output(target)
    # print two versions of the conversation.
    print_color(f"Conversation: {conversation_str}", "green")
    breakpoint()
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
        for _, candidate in memory:
            rollouts = candidate.rollouts
            random_rollout = random.choice(rollouts)
            trajectories.append(get_trajectories_of_one_rollout(random_rollout))
        return '\n'.join(trajectories)

    def summarize(self, memory) -> str:
        """Summarize the trajectories using the LLM.
        Args:
            memory: The memory containing trajectories to summarize.
        Returns:
            str: The summary.
        """

        history_trajectories = self._get_trajecories_for_memory(memory)
        
        system_prompt = "You are an expert at analyzing agent behavior patterns and extracting insights to improve agent performance."
        
        user_prompt = f"""Analyze the following agent conversation trajectories and provide insights.

        Trajectories:
        {history_trajectories}

        Please provide your analysis in JSON format with the following structure:
        1. First, provide your reasoning about patterns you observe in these trajectories
        2. Then, provide a summary with key insights

        Output format:
        {{
            "reasoning": "Your step-by-step analysis of patterns, successful strategies, common failures, and important observations from the trajectories",
            "summary": "A concise summary of key insights that can guide the optimizer to generate better agent parameters. Focus on what works well and what should be improved."
        }}"""

        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response_format = {"type": "json_object"}
        response = self.llm(prompt_messages, response_format=response_format)

        summary = response.choices[0].message.content
        summary_json = json.loads(summary)
        return summary_json['summary']