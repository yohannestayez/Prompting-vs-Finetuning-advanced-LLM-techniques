import autogen
import json
import re


def _extract_json_from_codeblock(content: str) -> str:
    # Find start and end of the ```json code block
    start = content.find("```json")
    end = content.rfind("```")
    if start != -1 and end != -1:
        # Extract and clean the JSON part
        json_content = content[start + 7:end].strip()
        return json_content
    else:
        # Return original content if no code block found
        return content

class Instructor(autogen.AssistantAgent):
    """
    Uses Role Prompting + Instruction Prompting
    """
    def __init__(self, llm_config):
        # Define the expert role in system message
        system_message = """You are a senior performance optimization engineer with 15 years of experience. 
        Your specialty is identifying optimization opportunities in Python code."""
        super().__init__(name="Instructor", llm_config=llm_config, system_message=system_message)

    def set_optimization_context(self, code: str) -> str:
        # Create prompt for optimization analysis
        prompt = f"""Analyze this code for optimization potential:
        {code}
        Consider: time complexity, memory usage, and API efficiency"""
        # Get agent’s reply
        response = self.generate_reply([{"content": prompt, "role": "user"}])
        return response

class Classifier(autogen.AssistantAgent):
    """
    Uses Few-Shot Prompting with structured examples
    System message contains classification demonstrations
    """
    def __init__(self, llm_config):
        system_message = """
            **Task:**
            Classify inefficiencies in the following code snippets. For each example, analyze the code and return a JSON object describing the inefficiency using the fields:

            * `type`: The broad category of inefficiency (e.g., `"Algorithm"`, `"Memory"`, `"I/O"`)
            * `category`: A more specific sub-type (e.g., `"Time Complexity"`, `"High Allocation"`, `"Network"`)
            * `label`: A concise description of the inefficiency (e.g., `"O(n²)"`, `"Linear Storage"`, `"Unbatched Requests"`)

            **Constraints:**

            * Output **JSON only** for each example
            * Format strictly as:
            `{"type": "...", "category": "...", "label": "..."}`
            * No explanations or commentary

            **Examples:**

            ```
            [Example 1]  
            Code:  
            for i in range(n):  
                for j in range(n):  
                    process(i, j)  
            Response:  
            {"type": "Algorithm", "category": "Time Complexity", "label": "O(n²)"}

            [Example 2]  
            Code:  
            data = [x for x in range(10^6)]  
            Response:  
            {"type": "Memory", "category": "High Allocation", "label": "Linear Storage"}

            [Example 3]  
            Code:  
            result = requests.get(url)  
            result.json()  
            Response:  
            {"type": "I/O", "category": "Network", "label": "Unbatched Requests"}
            ```

            """
        
        super().__init__(name="Classifier", llm_config=llm_config, system_message=system_message)

    def classify_inefficiency(self, code: str) -> dict:
        """Structured Output Prompting: Forces JSON format classification"""
        prompt = f"Classify optimization potential for:\n```python\n{code}\n```"
        response = str(self.generate_reply([{"content": prompt, "role": "user"}])['content'])
        response=_extract_json_from_codeblock(response)
        
        try:
            if isinstance(response, str):
                return json.loads(response)
            else:
                return response
        except json.JSONDecodeError:
            return {"type": "Unknown", "category": "Unclassified", "label": "Needs Review"}

class Optimizer(autogen.AssistantAgent):
    """
    Implements Chain-of-Thought (CoT) with Self-Consistency
    Generates multiple analysis paths then synthesizes them
    """
    def __init__(self, llm_config):
        # Define system message with 3-step optimization strategy
        system_message = """Break down optimizations in 3 steps: 
        1. Identify bottlenecks 2. Compare approaches 3. Select best method"""
        super().__init__(name="Optimizer", llm_config=llm_config, system_message=system_message)

    def generate_optimization_plan(self, code: str) -> str:
        """Self-Consistency Prompting: Two analysis paths + synthesis"""
        # Analyze algorithmic complexity
        prompt1 = f"""Analyze code:\n{code}\nFocus on algorithmic complexity"""
        analysis1 = self.generate_reply([{"content": prompt1, "role": "user"}])['content']
        
        # Analyze memory usage patterns
        prompt2 = f"""Analyze code:\n{code}\nFocus on memory patterns"""
        analysis2 = self.generate_reply([{"content": prompt2, "role": "user"}])['content']
        
        # Combine both analyses into a unified plan
        final_prompt = f"""Combine these analyses:
        Path 1: {analysis1}
        Path 2: {analysis2}
        Create unified optimization plan"""
        return self.generate_reply([{"content": final_prompt, "role": "user"}])['content']

class Implementer(autogen.AssistantAgent):
    """
    Uses Code Generation + Program-Aided Language (PAL) techniques
    Structured output prompting with code block enforcement
    """
    def __init__(self, llm_config):
        # Set system message to enforce code blocks and explanations
        system_message = """Generate optimized Python code with explanations.
        ALWAYS include code in Markdown blocks."""
        super().__init__(name="Implementer", llm_config=llm_config, system_message=system_message)

    def implement_optimizations(self, code: str, analysis: str) -> str:
        """Structured Output + Code Generation Prompting"""
        # Build the prompt with analysis, code, and requested output format
        prompt = f"""Based on this analysis:
        {analysis}
        Optimize this code:
        ```python
        {code}
        ```
        Provide:
        1. Optimized code in Python block
        2. Complexity comparison table
        3. Memory usage explanation"""
        
        # Get the agent’s response
        response = self.generate_reply([{"content": prompt, "role": "user"}])['content']
        # Extract the Python code block from the response
        match = re.search(r'```python(.*?)```', response, re.DOTALL)
        result = match.group(1).strip() if match else "No valid code generated"
        return result

class Insighter(autogen.AssistantAgent):
    """
    Implements Generated Knowledge Prompting
    Focuses on extracting optimization patterns and best practices
    """
    def __init__(self, llm_config):
        system_message = """Explain optimization patterns and language-specific best practices.
        Connect to Computer Science fundamentals."""
        super().__init__(name="Insighter", llm_config=llm_config, system_message=system_message)

    def generate_optimization_insights(self, analysis: str) -> str:
        """Generated Knowledge Prompting: Extract principles from analysis"""
        prompt = f"""
                    Transform the following code optimization analysis into a set of **generalized optimization guidelines**.

                    Input:
                    {analysis}

                    Format the output strictly as:
                    1. **Core Principle** – Abstract, language-agnostic performance insight.
                    2. **Language-Specific Tip** – Practical advice for Python developers.
                    3. **Tradeoff Consideration** – What might be lost or compromised by applying this optimization.

                    Instructions:
                    - Do **not** include any code from the original input.
                    - Do **not** output code blocks.
                    - Focus exclusively on distilled conceptual and practical insights.
                    - Keep the tone concise, formal, and technical.
                    - Output only the transformed optimization rules — no additional commentary.
                    """

        return self.generate_reply([{"content": prompt, "role": "user"}])['content']

class Refiner(autogen.AssistantAgent):
    """
    Uses ReAct (Reasoning + Acting) Prompting
    Implements iterative refinement based on benchmark results
    """
    def __init__(self, llm_config):
        system_message = """Iteratively refine optimizations using benchmark data.
        Consider time/memory tradeoffs."""
        super().__init__(name="Refiner", llm_config=llm_config, system_message=system_message)

    def refine_implementation(self, code: str, optimized: str, benchmark: dict) -> str:
        """ReAct Prompting: Reason about metrics, act with new implementation"""
        prompt = f"""        
        Current optimization achieved:
            - Time: {benchmark['optimized_time']} (Δ{benchmark['time_improvement']}%)
            - Memory: {benchmark['optimized_memory']} (Δ{benchmark['memory_change']}%)

        Please refine the following code further based on the performance metrics above.

        Requirements:
        - Only return an improved Python code block.
        - The code must be functionally equivalent to the original but optimized for performance.
        - Avoid unnecessary comments or explanations before the code block.
        - Maintain readability and Pythonic style.

        Improved version:
        ```python
        # Your optimized version here
        ````

        Original code (for reference):

        ```python
        {code}
        ```

        """
        
        response = self.generate_reply([{"content": prompt, "role": "user"}])['content']
        match = re.search(r'```python(.*?)```', response, re.DOTALL)
        return match.group(1).strip() if match else optimized