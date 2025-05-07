# Project: AI-Powered Python Code Optimization

This project leverages a multi-agent system built with AutoGen to automatically analyze, optimize, and provide insights on Python code. It employs a sophisticated pipeline of specialized AI agents, each utilizing advanced prompting techniques to achieve high-quality code optimization.

## Core Idea

The system takes Python code as input and processes it through a series of agents:
1.  An **Instructor** agent first analyzes the code for general optimization potential.
2.  A **Classifier** agent then identifies and categorizes specific inefficiencies.
3.  An **Optimizer** agent devises a strategic plan to address these inefficiencies.
4.  An **Implementer** agent generates the optimized Python code.
5.  An **Insighter** agent distills generalized optimization principles and best practices from the process.
6.  A **Refiner** agent iteratively improves the optimized code based on (simulated or actual) benchmark feedback.

The collaboration between these agents, orchestrated through carefully designed prompts, allows for a comprehensive and nuanced approach to code optimization.

## Folder Structure

```
.
├── main.py             # Main script to run the optimization pipeline
├── optimizers/
│   ├── agents.py       # Defines the specialized AI agents and their prompting strategies
│   ├── pipeline.py     # Orchestrates the flow of code and information between agents
│   └── utils.py        # Utility functions (e.g., for JSON extraction)
└── README.md           # This file
```

## Prompting Techniques Employed

This project places a strong emphasis on sophisticated prompting techniques to guide the Large Language Models (LLMs) effectively within each agent. Here's a breakdown of the techniques used in `optimizers/agents.py`:

### 1. Role Prompting
*   **What it is**: Assigning a specific persona or role to the LLM to frame its responses and behavior.
*   **How it's used**:
    *   The **`Instructor`** agent is given the system message: `"You are a senior performance optimization engineer with 15 years of experience. Your specialty is identifying optimization opportunities in Python code."` This primes the LLM to adopt an expert mindset for code analysis.

### 2. Instruction Prompting
*   **What it is**: Providing clear, direct instructions to the LLM on the task to be performed and the expected output.
*   **How it's used**:
    *   The **`Instructor`** agent's `set_optimization_context` method uses a prompt like: `"Analyze this code for optimization potential:\n{code}\nConsider: time complexity, memory usage, and API efficiency"`. This gives a direct command and specifies areas of focus.

### 3. Few-Shot Prompting
*   **What it is**: Providing a few examples (shots) of the task and the desired output format within the prompt to help the LLM understand patterns and expectations.
*   **How it's used**:
    *   The **`Classifier`** agent's system message includes multiple examples of code snippets and their corresponding JSON classification output (e.g., `[Example 1] Code: ... Response: {"type": "Algorithm", ...}`). This helps the LLM learn the desired input-output mapping for inefficiency classification.

### 4. Structured Output Prompting
*   **What it is**: Guiding the LLM to produce output in a specific, often machine-readable, format (like JSON or XML). This often involves explicitly stating the desired structure, fields, and constraints.
*   **How it's used**:
    *   The **`Classifier`** agent's system message details the exact JSON structure required (`{"type": "...", "category": "...", "label": "..."}`) and includes constraints like "Output JSON only" and "No explanations or commentary."
    *   The **`Implementer`** agent's system message commands: `"Generate optimized Python code with explanations. ALWAYS include code in Markdown blocks."` Its `implement_optimizations` method further specifies a multi-part output: optimized code, complexity table, and memory explanation.
    *   The **`Insighter`** agent's `generate_optimization_insights` prompt specifies a strict output format: "1. **Core Principle**...", "2. **Language-Specific Tip**...", "3. **Tradeoff Consideration**...".

### 5. Chain-of-Thought (CoT) Prompting
*   **What it is**: Encouraging the LLM to generate a series of intermediate reasoning steps before arriving at a final answer, often improving performance on complex tasks.
*   **How it's used**:
    *   The **`Optimizer`** agent's system message: `"Break down optimizations in 3 steps: 1. Identify bottlenecks 2. Compare approaches 3. Select best method"` guides the LLM to follow a structured thought process for creating an optimization plan.

### 6. Self-Consistency
*   **What it is**: Generating multiple diverse reasoning paths (or solutions) for a problem and then selecting the most consistent or best one, often leading to more robust results.
*   **How it's used**:
    *   The **`Optimizer`** agent's `generate_optimization_plan` method first prompts for two separate analyses of the same code (one focusing on algorithmic complexity, the other on memory patterns). It then uses a final prompt to `"Combine these analyses ... Create unified optimization plan"`, effectively synthesizing multiple "thoughts" into a final output.

### 7. Code Generation
*   **What it is**: Prompting the LLM to generate executable code in a specific programming language.
*   **How it's used**:
    *   The **`Implementer`** agent is explicitly designed to generate Python code. Its `implement_optimizations` method prompts for "Optimized code in Python block."

### 8. Program-Aided Language (PAL) Techniques
*   **What it is**: Using an LLM to generate code (the "program") that can then be executed by an interpreter to solve a problem or produce a result. The LLM's output is a step in a larger computational process.
*   **How it's used**:
    *   The **`Implementer`** agent generates Python code which is intended to be executable. The `_extract_json_from_codeblock` utility function and the regex in `Implementer` to extract Python code from markdown are also examples of programmatically processing LLM output.

### 9. Generated Knowledge Prompting
*   **What it is**: Prompting an LLM to synthesize its existing knowledge and the provided context to generate new insights, principles, or explanations that were not explicitly stated in the input.
*   **How it's used**:
    *   The **`Insighter`** agent's `generate_optimization_insights` method takes a code optimization analysis and prompts the LLM to `"Transform the following code optimization analysis into a set of generalized optimization guidelines"`, asking for abstract core principles, language-specific tips, and tradeoff considerations, thus generating new, distilled knowledge.

### 10. ReAct (Reason + Act) Prompting
*   **What it is**: A paradigm where the LLM iteratively generates reasoning traces and task-specific actions. It reasons about what to do, performs an action (which can involve using external tools or generating text/code), observes the result, and then reasons again to decide the next action.
*   **How it's used**:
    *   The **`Refiner`** agent's `refine_implementation` method exemplifies this. It's given benchmark data (the "observation" from a previous "action" of running the code). The prompt then asks it to `"refine the following code further based on the performance metrics"` (reasoning and generating new code - the next "action"). The system message `"Iteratively refine optimizations using benchmark data"` also sets up this iterative improvement loop.