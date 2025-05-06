import autogen
import json
import re

class Instructor(autogen.AssistantAgent):
    """
    Uses Role Prompting + Instruction Prompting
    System message sets expert role, method provides specific optimization context
    """
    def __init__(self, llm_config):
        system_message = """You are a senior performance optimization engineer with 15 years of experience. 
        Your specialty is identifying optimization opportunities in Python code."""
        super().__init__(name="Instructor", llm_config=llm_config, system_message=system_message)

    def set_optimization_context(self, code: str) -> str:
        """Instruction Prompting: Directs analysis of specific optimization aspects"""
        prompt = f"""Analyze this code for optimization potential:
        {code}
        Consider: time complexity, memory usage, and API efficiency"""
        response = self.generate_reply([{"content": prompt, "role": "user"}])
        return response

class Classifier(autogen.AssistantAgent):
    """
    Uses Few-Shot Prompting with structured examples
    System message contains classification demonstrations
    """
    def __init__(self, llm_config):
        system_message = """Classify code inefficiencies using these examples:
        
        [Example 1]
        Code: for i in range(n):\n    for j in range(n):\n        process(i,j)
        Response: {"type": "Algorithm", "category": "Time Complexity", "label": "O(n²)"}

        [Example 2] 
        Code: data = [x for x in range(10^6)]
        Response: {"type": "Memory", "category": "High Allocation", "label": "Linear Storage"}

        [Example 3]
        Code: result = requests.get(url)\nresult.json()
        Response: {"type": "I/O", "category": "Network", "label": "Unbatched Requests"}
        
        Return JSON ONLY with type, category, and label fields."""
        
        super().__init__(name="Classifier", llm_config=llm_config, system_message=system_message)

    def classify_inefficiency(self, code: str) -> dict:
        """Structured Output Prompting: Forces JSON format classification"""
        prompt = f"Classify optimization potential for:\n```python\n{code}\n```"
        response = str(self.generate_reply([{"content": prompt, "role": "user"}])['content'])
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"type": "Unknown", "category": "Unclassified", "label": "Needs Review"}

class Optimizer(autogen.AssistantAgent):
    """
    Implements Chain-of-Thought (CoT) with Self-Consistency
    Generates multiple analysis paths then synthesizes them
    """
    def __init__(self, llm_config):
        system_message = """Break down optimizations in 3 steps: 
        1. Identify bottlenecks 2. Compare approaches 3. Select best method"""
        super().__init__(name="Optimizer", llm_config=llm_config, system_message=system_message)

    def generate_optimization_plan(self, code: str) -> str:
        """Self-Consistency Prompting: Two analysis paths + synthesis"""
        # First analysis path
        prompt1 = f"""Analyze code:\n{code}\nFocus on algorithmic complexity"""
        analysis1 = self.generate_reply([{"content": prompt1, "role": "user"}])['content']
        
        # Second analysis path 
        prompt2 = f"""Analyze same code:\n{code}\nFocus on memory patterns"""
        analysis2 = self.generate_reply([{"content": prompt2, "role": "user"}])['content']
        
        # Synthesis prompt
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
        system_message = """Generate optimized Python code with explanations.
        ALWAYS include code in Markdown blocks."""
        super().__init__(name="Implementer", llm_config=llm_config, system_message=system_message)

    def implement_optimizations(self, code: str, analysis: str) -> str:
        """Structured Output + Code Generation Prompting"""
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
        
        response = self.generate_reply([{"content": prompt, "role": "user"}])['content']
        match = re.search(r'```python(.*?)```', response, re.DOTALL)
        return match.group(1).strip() if match else "No valid code generated"

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
        prompt = f"""Transform this analysis into general optimization rules:
        {analysis}
        Present as:
        1. Core Principle
        2. Language-Specific Tip
        3. Tradeoff Consideration"""
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
        prompt = f"""Current optimization achieved:
        - Time: {benchmark['time']}s (Δ{benchmark['time_diff']}%)
        - Memory: {benchmark['memory']}MB (Δ{benchmark['mem_diff']}%)
        
        Refine this code:
        ```python
        {optimized}
        ```
        Original:
        ```python
        {code}
        ```
        Suggest better approach considering metrics."""
        
        response = self.generate_reply([{"content": prompt, "role": "user"}])['content']
        match = re.search(r'```python(.*?)```', response, re.DOTALL)
        return match.group(1).strip() if match else optimized