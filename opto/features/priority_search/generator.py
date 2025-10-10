from opto.utils.llm import LLM
from opto.features.priority_search.priority_search import ModuleCandidate
from opto.optimizers.utils import print_color
import ast
import re

def get_parameter_text(candidate):
        """Get the parameter text for a ModuleCandidate."""
        if not candidate.update_dict:
            return "base_module_parameters"
        # Convert parameter nodes to readable names for deterministic embedding
        params_with_names = {k.py_name: v for k, v in candidate.update_dict.items()}
        return str(params_with_names)

class LLMCandidateGenerator:
    """Generate new candidates using LLM with OptoPrimeV2-style prompts."""
    
    def __init__(self, model_name="gemini/gemini-2.0-flash", temperature=0.0, verbose=False,num_threads=None, max_candidates_in_prompt=20):
        self.llm = LLM(model=model_name)
        self.temperature = temperature
        # In priority search we store negative scores in the memory
        self.negative_score = True
        self.verbose = verbose
        self.num_threads = num_threads
        self.max_candidates_in_prompt = max_candidates_in_prompt
        if verbose:
            print(f"LLMCandidateGenerator initialized with model {model_name} and temperature {temperature}")
    
    def generate_candidates(self, base_module, optimizer, memory, num_candidates=5):
        """Generate new candidates using LLM based on memory of past candidates.
        memory: a list of candidate
        """
        
        # Generate 1 candidate per batch for maximum reliability
        if self.verbose:
            print(f"Generating {num_candidates} candidates, 1 candidate per batch")
        
        # Create memory subset for prompts
        # memory_subset = memory[:self.max_candidates_in_prompt]
        
        # Create a single generation function and replicate it
        def generate_single_candidate():
            return self._generate_single_batch(base_module, optimizer, memory, 1)
        
        generation_functions = [generate_single_candidate] * num_candidates
        
        # Use async_run if num_threads > 1, otherwise run sequentially
        if self.num_threads and self.num_threads > 1 and num_candidates > 1:
            from opto.trainer.utils import async_run
            batch_results = async_run(
                generation_functions,
                max_workers=self.num_threads,
                description=f"Generating {num_candidates} candidates (1 per batch)"
            )
        else:
            # Sequential execution
            batch_results = [func() for func in generation_functions]
        
        # Flatten all candidates from all batches
        all_candidates = []
        for batch_candidates in batch_results:
            if batch_candidates:  # Handle None/empty results
                all_candidates.extend(batch_candidates)
        
        if self.verbose:
            print(f"Successfully generated {len(all_candidates)} candidates total using the generator")
        
        return all_candidates
    
    def _generate_single_batch(self, base_module, optimizer, memory_subset, num_candidates):
        """Generate a single batch of candidates."""
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_user_prompt(base_module, memory_subset, num_candidates)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        if self.verbose:
            print(f"User prompt: {user_prompt}")
        
        try:
            response = self.llm(messages=messages, temperature=self.temperature, max_tokens=8192)
            response_text = response.choices[0].message.content
            if self.verbose:
                print(f"Response text: {response_text}")
            # Parse candidates from response
            candidates = self._parse_candidates(response_text, base_module, optimizer)
            return candidates
            
        except Exception as e:
            print_color(f"Error generating batch: {e}", "red")
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
            sorted_memory = sorted(memory, key=lambda x: x.predicted_score, reverse=True)
            
            for i, candidate in enumerate(sorted_memory[:10]):  # Show top 10
                display_score = candidate.predicted_score
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

<candidates>"""

        # Add candidate examples based on the number requested
        for i in range(min(num_candidates, 2)):  # Show at most 2 examples
            prompt += f"""
<candidate index="{i+1}">
<reasoning>
[Brief explanation of why this configuration should perform well]
</reasoning>
<parameters>"""
            # Add example parameter format
            for param_name in base_params.keys():
                prompt += f"""
  <parameter name='{param_name}'><![CDATA[new_value_for_{param_name}]]></parameter>"""
            prompt += """
</parameters>
</candidate>"""
        
        if num_candidates > 2:
            prompt += f"""

... (continue for all {num_candidates} candidates)"""
        
        prompt += """
</candidates>

Generate diverse candidates that explore different promising directions while building on successful patterns."""   
        
        return prompt
    
    def _parse_candidates(self, response_text, base_module, optimizer):
        """Parse candidate configurations from LLM response."""
        candidates = []
        
        try:
            # Extract candidates section
            candidates_match = re.search(r'<candidates>(.*?)</candidates>', response_text, re.DOTALL)
            if not candidates_match:
                print_color("No candidates section found in response", "red")
                return candidates
            
            candidates_section = candidates_match.group(1)
            
            # Extract individual candidates
            candidate_pattern = r'<candidate[^>]*index=["\'](\d+)["\'][^>]*>(.*?)</candidate>'
            candidate_matches = re.finditer(candidate_pattern, candidates_section, re.DOTALL)
            
            for match in candidate_matches:
                candidate_content = match.group(2)
                
                # Extract parameters using XML format
                params_match = re.search(r'<parameters>(.*?)</parameters>', candidate_content, re.DOTALL)
                if params_match:
                    params_section = params_match.group(1).strip()
                    
                    try:
                        # Parse XML parameters similar to regressor.py
                        params_dict = {}
                        
                        # Extract individual parameter elements
                        param_pattern = r'<parameter\s+name=["\']([^"\']+)["\'][^>]*>(.*?)</parameter>'
                        param_matches = re.finditer(param_pattern, params_section, re.DOTALL)
                        
                        for param_match in param_matches:
                            param_name = param_match.group(1)
                            param_content = param_match.group(2).strip()
                            
                            # Handle CDATA sections
                            cdata_match = re.search(r'<!\[CDATA\[(.*?)\]\]>', param_content, re.DOTALL)
                            if cdata_match:
                                param_value = cdata_match.group(1)
                            else:
                                # Handle regular text content, unescaping XML entities
                                param_value = param_content.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"').replace('&apos;', "'")
                            
                            # Try to convert to appropriate Python type
                            try:
                                # Try to evaluate as Python literal (safer than eval)
                                param_value = ast.literal_eval(param_value)
                            except (ValueError, SyntaxError):
                                # Keep as string if not a valid Python literal
                                pass
                            
                            params_dict[param_name] = param_value
                        
                        # Map string parameter names back to ParameterNode objects
                        update_dict = {}
                        param_name_to_node = {p.py_name if hasattr(p, 'py_name') else str(p): p for p in base_module.parameters()}
                        
                        for param_name, value in params_dict.items():
                            if param_name in param_name_to_node:
                                update_dict[param_name_to_node[param_name]] = value
                            else:
                                print_color(f"Warning: Parameter '{param_name}' not found in base module parameters", "red")
                        
                        # Create ModuleCandidate with ParameterNode keys
                        candidate = ModuleCandidate(
                            base_module=base_module,
                            update_dict=update_dict,
                            optimizer=optimizer
                        )
                        candidates.append(candidate)
                        
                    except Exception as e:
                        print_color(f"Error parsing candidate parameters: {e}", "red")
                        # if self.verbose:
                        #     print("response_text: ", response_text)
                            # print("params_section: ", params_section)

                        continue
        
        except Exception as e:
            print_color(f"Error parsing candidates: {e}", "red")
        
        return candidates