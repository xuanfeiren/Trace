from opto.optimizers.utils import print_color
from opto.utils.llm import LLM # For the selector LLM
import json
import random
import re
choices = ["tau-bench", "veribench", "generic"]
DOMAIN = "generic" # or "tau-bench" or "veribench"
# from system_prompts import SYSTEM_PROMPT, EXAMPLES

def get_tau_bench_trajectory_from_output(output):
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

def get_tau_bench_trajectory_of_one_rollout(rollout):
    """Get trajectories of one rollout."""
    target = rollout['target']
    # reward, messages = target
    conversation_str = get_tau_bench_trajectory_from_output(target)
    # print two versions of the conversation.
    # print_color(f"Conversation: {conversation_str}", "green")
    return conversation_str

def get_veribench_trajectory_of_one_rollout(rollout):
    """
    Convert a rollout into a structured markdown trajectory for Veribench optimization.

    The agent's system prompt structure is:
        SYSTEM_PROMPT (fixed) + additional_instructions (trainable) + EXAMPLES (fixed)
    
    This function extracts the trainable `additional_instructions` parameter and formats
    the trajectory to guide the optimizer in improving this component.

    Parameters
    ----------
    rollout : dict
        A rollout dictionary containing:
        - 'module': trace.Module - the agent module with trainable additional_instructions
        - 'x': Any - the input (Python code to be translated)
        - 'info': Any - additional information about the input
        - 'target': Any - the generated Lean 4 code output
        - 'score': float - evaluation score (0 = failed, 1 = success)
        - 'feedback': Any - detailed compilation feedback from the guide

    Returns
    -------
    str
        A markdown-formatted trajectory string for optimizer guidance.
    """
    assert DOMAIN == "veribench", "This function is only for Veribench."
    assert rollout['module'] is not None, "rollout['module'] is None."
    assert rollout['x'] is not None, "rollout['x'] is None."
    assert rollout['target'] is not None, "rollout['target'] is None."
    assert rollout['score'] is not None, "rollout['score'] is None."
    assert rollout['feedback'] is not None, "rollout['feedback'] is None."
    
    # Extract trainable parameters (additional_instructions)
    parameters = rollout['module'].parameters()
    parameters_dict = {p.py_name: p.data for p in parameters}
    
    # Extract rollout components
    python_code = rollout['x'] # input
    lean_output = rollout['target']
    score = rollout['score']
    feedback = rollout['feedback']
    
    # Construct structured markdown trajectory
    trajectory = f"""## Task: Python → Lean 4 Translation

## Input (Python Code)
{python_code}

## Generated Lean 4 Code (Trainable Parameter)
{lean_output}

## Result
- **Score:** {score} (0 = failed, 1 = success)
- **Compilation Feedback, contains the error message for the failed lean 4 code:** {feedback}

## Optimization Note
The Lean 4 code above is the trainable parameter. Analyze what code patterns lead to successful compilation vs. failure.
"""
    # print_color(f"Trajectory: {trajectory}", "green")
    # breakpoint()
    return trajectory

def get_generic_trajectory_of_one_rollout(rollout):
    """
    Get trajectory of one rollout for the generic domain.
    """
    assert DOMAIN == "generic", "This function is only for generic domain."
    assert rollout['module'] is not None, "rollout['module'] is None."
    assert rollout['x'] is not None, "rollout['x'] is None."
    assert rollout['target'] is not None, "rollout['target'] is None."
    assert rollout['score'] is not None, "rollout['score'] is None."
    assert rollout['feedback'] is not None, "rollout['feedback'] is None."
    
    # Extract trainable parameters
    parameters = rollout['module'].parameters()
    parameters_dict = {p.py_name: p.data for p in parameters}
    
    # Construct structured markdown trajectory
    trajectory = f"""## Task parameters: {parameters_dict}
## Input: {rollout['x']}
## Output: {rollout['target']}
## Result
- **Score:** {rollout['score']}
- **Feedback:** {rollout['feedback']}"""
    return trajectory

if DOMAIN == "tau-bench":
    get_trajectory_of_one_rollout = get_tau_bench_trajectory_of_one_rollout
elif DOMAIN == "veribench":
    get_trajectory_of_one_rollout = get_veribench_trajectory_of_one_rollout
elif DOMAIN == "generic":
    get_trajectory_of_one_rollout = get_generic_trajectory_of_one_rollout
else:
    raise ValueError(f"Invalid domain: {DOMAIN}")

class Summarizer:
    """A class which use LLM to summarize the trajectories of the memory. It should be able to learn the patterns of the trajectories. Generate a summary to guide the optimizer to generate better candidates.
    """
    def __init__(self, model_name: str = "claude-3.5-sonnet"):
        self.llm = LLM() # use the default model
        self.max_candidates_in_prompt = 5
        self.current_summary = "Concrete recommendations for generating better agent parameters based on successful patterns observed in the trajectories: "
        self.used_candidates = set()  # Track candidates that have been summarized

    def _get_trajectories_for_memory(self, memory):
        """
        Get trajectories for the memory. Memory is a list of (neg_score, candidate) tuples.
        We first collect rollouts from the each candidate, and then get the trajectories for each rollout.

        Return one single string of all trajectories.
        """
        trajectories = []
        print_color(f"Getting trajectories from {len(memory)} candidates.", "blue")
        # Filter out candidates that have already been used and have rollouts
        memory_with_rollouts = [(neg_score, candidate) for neg_score, candidate in memory
                                if len([rollout for rollout in candidate.rollouts if rollout['score'] is not None]) > 0
                                and id(candidate) not in self.used_candidates]
        print_color(f"Memory (unseen candidates) with rollouts: {len(memory_with_rollouts)}", "blue")
        # Sample 5 candidates (or fewer if not enough available)
        num_to_sample = min(5, len(memory_with_rollouts))
        temporary_memory = random.sample(memory_with_rollouts, k=num_to_sample)
        # Mark sampled candidates as used
        for _, candidate in temporary_memory:
            self.used_candidates.add(id(candidate))
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

        history_trajectories = self._get_trajectories_for_memory(memory)

        # print_color(f"History trajectories: {history_trajectories}", "green")

        if len(history_trajectories) == 0:
            return "No successful trajectories found for the memory."
        
        system_prompt = "You are an expert at analyzing agent behavior patterns and providing actionable guidance for parameter optimization."
        
        user_prompt = f"""Analyze the following agent conversation trajectories and extract insights for optimization.

        Current Summary (from previous analysis):
        {self.current_summary}

        New Trajectories to Analyze:
        {history_trajectories}

        Instructions:
        - Keep all insights from the Current Summary above
        - Analyze the new trajectories and identify any new patterns
        - Add new insights to the summary while preserving existing ones
        - Build upon and refine the current recommendations

        Provide your analysis in XML format:
        <reasoning>
        Analyze the key patterns and strategies that led to success or failure in these trajectories.
        </reasoning>
        <summary>
        Concrete recommendations for generating better parameters based on successful or failed patterns observed in the trajectories. Keep the current summary and add new insights from the new trajectories. Write the entire modified summary here.
        </summary>"""

        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # print_color(f"User prompt: {user_prompt}", "blue")
        
        # print_color(f"System prompt: {system_prompt}", "blue")
        # print_color(f"User prompt: {user_prompt}", "blue")
        
        response = self.llm(messages=prompt_messages)
        response = response.choices[0].message.content
        # print_color(f"Response: {response}", "yellow")
        
        # Extract summary using XML regex
        summary_match = re.search(r'<summary>(.*?)</summary>', response, re.DOTALL)

        self.current_summary = summary_match.group(1).strip()

        return self.current_summary

from opto.trainer.utils import async_run

class DetailedSummarizer:
    """A class which use LLM to summarize the trajectories of the memory. It should be able to learn the patterns of the trajectories. Generate a summary to guide the optimizer to generate better candidates.
    This version generates summaries for each (candidate, task) pair. Then it will be combined to a context, or call LLM for a final summary.
    """
    def __init__(self, model_name: str = "gemini/gemini-2.0-flash"):
        self.llm = LLM(model=model_name)
        self.long_summary = None

    def subsummarize(self, candidate, x):
        """
        Generate a summary for a specific (candidate, task) pair across multiple trajectories.
        
        Calls the LLM to analyze the candidate's performance on task x and generate a structured response containing:
        - <reasoning>: Analysis of the candidate's approach and decision-making process
        - <summary>: Description of how the candidate behaves when solving task x across observed trajectories
        - <insights>: Identification of success patterns and failure modes for the candidate on task x
        
        Args:
            candidate: The candidate agent/parameters being evaluated.
            x: The specific task or input on which the candidate was tested.
            
        Returns:
            A structured summary of the candidate's behavior on the given task.
        """
        # Extract the trajectories for the (candidate, x) pair.
        rollouts = [rollout for rollout in candidate.rollouts if rollout['x'] == x and rollout['score'] is not None]
        if len(rollouts) == 0:
            return None
        # Get the trajectories for the rollouts.
        # trajectories = [get_trajectory_of_one_rollout(rollout) for rollout in rollouts]
        # Call LLM to generate structured response.
        
        # Categorize trajectories by success/failure
        successful_trajectories = [get_trajectory_of_one_rollout(r) for r in rollouts if r['score'] > 0]
        failed_trajectories = [get_trajectory_of_one_rollout(r) for r in rollouts if r['score'] == 0]
        
        # Build trajectory summary
        trajectory_summary = f"Task: {x}\n"
        trajectory_summary += f"Candidate parameters: {candidate.update_dict.values()}\n\n"
        
        if successful_trajectories:
            trajectory_summary += f"Successful trajectories ({len(successful_trajectories)}):\n"
            trajectory_summary += "\n---\n".join(successful_trajectories[:3])  # Limit to 3 examples
            trajectory_summary += "\n\n"
        
        if failed_trajectories:
            trajectory_summary += f"Failed trajectories ({len(failed_trajectories)}):\n"
            trajectory_summary += "\n---\n".join(failed_trajectories[:3])  # Limit to 3 examples
        
        system_prompt = "You are an expert at analyzing agent behavior and extracting actionable insights from execution traces."
        
        user_prompt = f"""Analyze the following trajectories for a candidate agent attempting to solve a specific task.

            {trajectory_summary}

            Provide a detailed analysis in the following XML format:
            <reasoning>Analyze the candidate's approach, decision-making process, and execution patterns across these trajectories</reasoning>
            <summary>Describe the candidate's overall behavior when solving this task, including strategies employed and common patterns</summary>
            <insights>Identify specific success patterns (what works) and failure modes (what doesn't work) for this candidate on this task</insights>

            Focus on actionable insights that can guide parameter optimization."""

        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.llm(messages=prompt_messages)
            response_content = response.choices[0].message.content
            
            # Extract summary and insights using regex
            summary_match = re.search(r'<summary>(.*?)</summary>', response_content, re.DOTALL)
            insights_match = re.search(r'<insights>(.*?)</insights>', response_content, re.DOTALL)
            
            # Format as XML with task wrapper using actual x value
            result = f"<task_{x}>\n"
            if summary_match:
                result += f"<summary>{summary_match.group(1).strip()}</summary>\n"
            if insights_match:
                result += f"<insights>{insights_match.group(1).strip()}</insights>\n"
            result += f"</task_{x}>"
            
            return result 
        except Exception as e:
            print_color(f"Error generating subsummary: {e}", "red")
            return None

    def structured_summary(self, memory):
        """
        Generate comprehensive summaries for all candidates in memory across their evaluated tasks.
        
        This method processes the memory by:
        1. Extracting all unique (candidate, task) pairs from candidate rollouts
        2. Calling subsummarize asynchronously for each (candidate, task) pair
        3. Aggregating task-level summaries for each candidate
        4. Formatting results as structured XML
        
        Args:
            memory: List of (neg_score, candidate) tuples containing evaluation history.
            
        Returns:
            str: XML-formatted string with structure:
                <candidate>
                    <parameters>{candidate.update_dict.values()}</parameters>
                    <task_1><summary>...</summary><insights>...</insights></task_1>
                    <task_2><summary>...</summary><insights>...</insights></task_2>
                    ...
                </candidate>
                Multiple candidate blocks are concatenated with newlines.
        """
        # Collect all (candidate, task) pairs from memory
        candidate_task_pairs = []
        candidate_map = {}  # Map to track which tasks belong to which candidate
        
        for idx, (_, candidate) in enumerate(memory):
            # Get unique tasks (x values) for this candidate
            tasks = set()
            for rollout in candidate.rollouts:
                if rollout['score'] is not None and 'x' in rollout:
                    tasks.add(rollout['x'])
            
            candidate_map[idx] = {
                'candidate': candidate,
                'tasks': list(tasks)
            }
            
            # Create (candidate, task) pairs for async processing
            for task in tasks:
                candidate_task_pairs.append((candidate, task))
        
        if len(candidate_task_pairs) == 0:
            return "No candidate-task pairs found in memory."
        
        print_color(f"Processing {len(candidate_task_pairs)} (candidate, task) pairs from {len(memory)} candidates.", "blue")
        
        # Prepare async execution
        runs = [self.subsummarize] * len(candidate_task_pairs)
        args_list = [[candidate, task] for candidate, task in candidate_task_pairs]
        
        # Run subsummarize asynchronously for all (candidate, task) pairs
        subsummaries = async_run(
            runs, 
            max_workers=100,
            args_list=args_list, 
            description="Generating task summaries"
        )
        
        # Organize results by candidate
        candidate_summaries = {}
        pair_idx = 0
        for idx, info in candidate_map.items():
            candidate = info['candidate']
            tasks = info['tasks']
            
            # Collect summaries for this candidate's tasks
            task_summaries = []
            for task in tasks:
                if pair_idx < len(subsummaries) and subsummaries[pair_idx] is not None:
                    task_summaries.append(subsummaries[pair_idx])
                pair_idx += 1
            
            candidate_summaries[idx] = {
                'candidate': candidate,
                'summaries': task_summaries
            }
        
        # Format as XML
        result_blocks = []
        for idx, info in candidate_summaries.items():
            candidate = info['candidate']
            summaries = info['summaries']
            
            if len(summaries) == 0:
                continue
            
            candidate_block = "<candidate>\n"
            candidate_block += f"<parameters>{list(candidate.update_dict.values())}</parameters>\n"
            
            for summary in summaries:
                if summary:
                    candidate_block += summary + "\n"
            
            candidate_block += "</candidate>"
            result_blocks.append(candidate_block)
        
        print_color(f"Generated summaries for {len(result_blocks)} candidates.", "green")
        
        detailed_xml = "\n\n".join(result_blocks) if result_blocks else ""
        # Store the structured summary for later use
        self.long_summary = detailed_xml
        return detailed_xml

    def summarize(self, memory):
        """
        Generate a concise actionable summary for the optimizer.
        """
        detailed_xml = self.structured_summary(memory)
        
        if not detailed_xml or detailed_xml == "No candidate-task pairs found in memory.":
            return "No summaries generated."
            
        # Final summarization step to condense XML into actionable insights
        print_color("Synthesizing final recommendations from detailed summaries...", "blue")
        
        system_prompt = "You are an expert meta-optimizer. Your goal is to synthesize detailed performance logs into actionable guidance for parameter optimization."
        
        user_prompt = f"""Analyze the following detailed performance summaries of different candidate parameters.

        The data below contains structured XML entries where each <candidate> block represents a different parameter configuration tested on multiple tasks. Each candidate includes:
        - <parameters>: The specific parameter values that were used
        - <task_X>: Performance analysis for task X, containing:
        - <summary>: Description of the candidate's behavior and strategies on this task
        - <insights>: Specific success patterns and failure modes observed

        Here is the detailed analysis:

        {detailed_xml}

        Important: The detailed XML contains many parameters with their summaries. There may be common success or failure patterns across multiple candidates. Your goal is to identify these common patterns and synthesize them into actionable guidance that will be fed into the optimizer to generate better parameters.

        Provide your response in XML format:
        <reasoning>
        Identify key success/failure patterns across candidates and tasks. Explain what works, what fails, and why.
        </reasoning>
        <summary>
        Concrete recommendations for the optimizer: what parameter qualities to pursue, what patterns to avoid, and specific guidance for generating improved candidates. This will be used directly as optimizer context.
        </summary>"""

        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.llm(messages=prompt_messages)
        content = response.choices[0].message.content
        
        # Extract summary using regex
        summary_match = re.search(r'<summary>(.*?)</summary>', content, re.DOTALL)
        
        final_summary = summary_match.group(1).strip()
        return final_summary
    
    def select_parameter(self,memory):
        """
        This is to improve the test-time performance of the search algorithm. We store the structured summary in the summarizer class. When we need to select a parameter to test, we can select one from the current memory, based on the information provided by the structured summary.
        Args:
            memory: The memory containing the candidates to select from.
        Returns:
            priority, ModuleCandidate: The priority and the selected candidate.
        """
        # construct a prompt contain the candidates in the memory with their parameters.
        candidates_prompt = "<candidates>\n"
        for idx, (_, candidate) in enumerate(memory):
            candidates_prompt += f"<candidate id=\"{idx}\">{candidate.update_dict.values()}</candidate>\n"
        candidates_prompt += "</candidates>"
        
        
        system_prompt = "You are an expert meta-optimizer responsible for selecting the most promising parameter candidate for testing based on historical performance patterns."
        
        user_prompt = f"""
        ## Task
        Select the candidate most likely to succeed based on historical performance patterns.

        ## Historical Performance Analysis
        {self.long_summary}

        ## Current Candidates to Choose From
        {candidates_prompt}

        ## Selection Criteria
        Based on the historical performance patterns above, consider:
        1. Which candidate parameters align best with successful patterns identified in the historical analysis?
        2. Which candidate most effectively avoids known failure patterns?

        ## Output Format
        Provide your response in the following XML format:
        <reasoning>
        Explain which historical patterns you're considering and why this candidate is most promising.
        </reasoning>
        <selection>
        <candidate_id>ID_OF_SELECTED_CANDIDATE</candidate_id>
        </selection>"""

        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.llm(messages=prompt_messages)
        content = response.choices[0].message.content
        print_color(f"Select parameter response: {content}", "blue")
        # Extract candidate_id using regex
        candidate_id_match = re.search(r'<candidate_id>(\d+)</candidate_id>', content, re.DOTALL)
        
        
        selected_idx = int(candidate_id_match.group(1))
        
        
        neg_priority, selected_candidate = memory[selected_idx]
        print_color(f"Selected candidate {selected_idx}", "green")
        
        # Return negative of neg_priority to get the original priority
        return -neg_priority, selected_candidate
        
