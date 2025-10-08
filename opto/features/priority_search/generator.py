from opto.utils.llm import LLM
from opto.features.priority_search.priority_search import ModuleCandidate

def get_parameter_text(candidate):
        """Get the parameter text for a ModuleCandidate."""
        if not candidate.update_dict:
            return "base_module_parameters"
        # Convert parameter nodes to readable names for deterministic embedding
        params_with_names = {k.py_name: v for k, v in candidate.update_dict.items()}
        return str(params_with_names)

class LLMCandidateGenerator:
    """Generate new candidates using LLM with OptoPrimeV2-style prompts."""
    
    def __init__(self, model_name="gemini/gemini-2.0-flash", temperature=0.0, verbose=False, max_candidates_in_prompt=20):
        self.llm = LLM(model=model_name)
        self.temperature = temperature
        # In priority search we store negative scores in the memory
        self.negative_score = True
        self.verbose = verbose
        self.max_candidates_in_prompt = max_candidates_in_prompt
        if verbose:
            print(f"LLMCandidateGenerator initialized with model {model_name} and temperature {temperature}")
    
    def generate_candidates(self, base_module, optimizer, memory, num_candidates=5):
        """Generate new candidates using LLM based on memory of past candidates."""
        if self.verbose:
            print(f"Generating {num_candidates} candidates using LLM generator.")
        # Create prompt based on OptoPrimeV2 structure
        # NOTE a heuristic for now
        memory_subset = memory[:self.max_candidates_in_prompt]
        # Use the max_candidates_in_prompt to limit the number of candidates in the prompt
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_user_prompt(base_module, memory_subset, num_candidates)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        # print("system_prompt: ", system_prompt)
        # print("user_prompt: ", user_prompt)
        if self.verbose:
            print("Prompt: ", messages)
        try:
            response = self.llm(messages=messages, temperature=self.temperature, max_tokens=4000)
            response_text = response.choices[0].message.content
            if self.verbose:
                print("Response: ", response_text)
            # Parse candidates from response
            candidates = self._parse_candidates(response_text, base_module, optimizer)
            return candidates
            
        except Exception as e:
            print(f"Error generating candidates: {e}")
            return []
    
    def _create_system_prompt(self):
        """Create system prompt for candidate generation."""
        return """You are an AI optimization assistant tasked with generating improved parameter configurations for a system.

Your goal is to analyze past performance data and generate new parameter configurations that are likely to achieve higher performance scores.

You will receive information about previous parameter configurations and their performance scores (0.0 to 1.0, where 1.0 is perfect)."""
    
    def _create_user_prompt(self, base_module, memory, num_candidates):
        """Create user prompt with memory context and generation request."""
        
        # Format memory examples
        memory_text = ""
        if memory:
            memory_text = "## Previous Configurations and Scores\n\n"
            # Sort memory by score (descending)
            sorted_memory = sorted(memory, key=lambda x: x[0], reverse=True)
            
            for i, (score, candidate) in enumerate(sorted_memory[:10]):  # Show top 10
                display_score = -score if self.negative_score else score
                memory_text += f"### Configuration {i+1} (Score: {display_score:.3f})\n"
                # Format parameters with proper names
                params_display = get_parameter_text(candidate)
                memory_text += f"Parameters: {params_display}\n\n"
        
        # Get base parameters with proper names
        base_params = {p.py_name if hasattr(p, 'py_name') else str(p): p.data for p in base_module.parameters()}
        
        prompt = f"""## Current Task
Generate {num_candidates} new parameter configurations to improve system performance.

## Base Configuration
{base_params}

{memory_text}

## Task
Generate {num_candidates} new parameter configurations that improve upon the best previous results by:
1. Analyzing what made successful configurations work well
2. Identifying weaknesses in poor-performing configurations  
3. Creating variations that combine the best aspects while addressing weaknesses
4. Exploring promising new directions based on the patterns you observe

## Output Format
Provide your response in the following XML format:

<reasoning>
[Your analysis of what makes configurations successful and your strategy for improvement]
</reasoning>

<candidates>
<candidate index="1">
<reasoning>
[Brief explanation of why this configuration should perform well]
</reasoning>
<parameters>
[Complete parameter dictionary for this candidate]
</parameters>
</candidate>

<candidate index="2">
<reasoning>
[Brief explanation of why this configuration should perform well]
</reasoning>
<parameters>
[Complete parameter dictionary for this candidate]
</parameters>
</candidate>

... (continue for all {num_candidates} candidates)
</candidates>

Generate diverse candidates that explore different promising directions while building on successful patterns."""   
        
        return prompt
    
    def _parse_candidates(self, response_text, base_module, optimizer):
        """Parse candidate configurations from LLM response."""
        candidates = []
        
        try:
            # Extract candidates section
            import re
            candidates_match = re.search(r'<candidates>(.*?)</candidates>', response_text, re.DOTALL)
            if not candidates_match:
                print("No candidates section found in response")
                return candidates
            
            candidates_section = candidates_match.group(1)
            
            # Extract individual candidates
            candidate_pattern = r'<candidate[^>]*index=["\'](\d+)["\'][^>]*>(.*?)</candidate>'
            candidate_matches = re.finditer(candidate_pattern, candidates_section, re.DOTALL)
            
            for match in candidate_matches:
                candidate_content = match.group(2)
                
                # Extract parameters
                params_match = re.search(r'<parameters>(.*?)</parameters>', candidate_content, re.DOTALL)
                if params_match:
                    params_text = params_match.group(1).strip()
                    
                    try:
                        # Try to evaluate as Python dict with string keys
                        params_dict = eval(params_text)
                        
                        # Map string parameter names back to ParameterNode objects
                        update_dict = {}
                        param_name_to_node = {p.py_name if hasattr(p, 'py_name') else str(p): p for p in base_module.parameters()}
                        
                        for param_name, value in params_dict.items():
                            if param_name in param_name_to_node:
                                update_dict[param_name_to_node[param_name]] = value
                            else:
                                print(f"Warning: Parameter '{param_name}' not found in base module parameters")
                        
                        # Create ModuleCandidate with ParameterNode keys
                        candidate = ModuleCandidate(
                            base_module=base_module,
                            update_dict=update_dict,
                            optimizer=optimizer
                        )
                        candidates.append(candidate)
                        
                    except Exception as e:
                        print(f"Error parsing candidate parameters: {e}")
                        continue
        
        except Exception as e:
            print(f"Error parsing candidates: {e}")
        
        return candidates