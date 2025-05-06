from optimizers.agents import (
    Instructor,
    Classifier,
    Optimizer,
    Implementer,
    Insighter,
    Refiner
)
from optimizers.utils import benchmark_code

class OptimizationPipeline:
    """
    Orchestrates the code optimization workflow using a sequence of specialized agents
    Implements a 6-stage pipeline with feedback loops
    """
    
    def __init__(self, llm_config):
        """
        Initialize all optimization agents with their respective prompting strategies
        """
        self.agents = {
            # Role + Instruction Prompting
            "instructor": Instructor(llm_config),
            
            # Few-Shot + Structured Output
            "classifier": Classifier(llm_config),
            
            # Chain-of-Thought + Self-Consistency
            "optimizer": Optimizer(llm_config),
            
            # Code Generation + PAL
            "implementer": Implementer(llm_config),
            
            # Generated Knowledge
            "insighter": Insighter(llm_config),
            
            # ReAct + Iterative Refinement
            "refiner": Refiner(llm_config)
        }

    def optimize(self, original_code: str) -> dict:
        """
        Execute full optimization pipeline:
        1. Context Setup → 2. Classification → 3. Analysis → 
        4. Implementation → 5. Refinement → 6. Insights
        """
        results = {}
        
        # Stage 1: Context Establishment
        results["context"] = self.agents["instructor"].set_optimization_context(original_code)
        
        # Stage 2: Issue Classification
        results["classification"] = self.agents["classifier"].classify_inefficiency(original_code)
        
        # Stage 3: CoT Analysis
        results["analysis"] = self.agents["optimizer"].generate_optimization_plan(original_code)
        
        # Stage 4: Code Implementation
        results["optimized_code"] = self.agents["implementer"].implement_optimizations(
            original_code, results["analysis"]
        )
        
        # Stage 5: Benchmark & Refinement
        original_benchmark = benchmark_code(original_code)
        optimized_benchmark = benchmark_code(results["optimized_code"])
        
        results["benchmark"] = self._calculate_improvements(
            original_benchmark, optimized_benchmark
        )
        
        # Refinement Condition: <20% time improvement or memory increase
        if (results["benchmark"]["time_improvement"] < 20 or 
            results["benchmark"]["memory_change"] > 0):
            results["optimized_code"] = self.agents["refiner"].refine_implementation(
                original_code,
                results["optimized_code"],
                results["benchmark"]
            )
            # Re-benchmark after refinement
            results["benchmark"] = self._calculate_improvements(
                original_benchmark, 
                benchmark_code(results["optimized_code"])
            )
        
        # Stage 6: Knowledge Generation
        results["insights"] = self.agents["insighter"].generate_optimization_insights(
            results["analysis"]
        )
        
        return results

    def _calculate_improvements(self, original: dict, optimized: dict) -> dict:
        """Calculate performance metrics between original and optimized code"""
        return {
            "original_time": original["time"],
            "optimized_time": optimized["time"],
            "time_improvement": round(
                (original["time"] - optimized["time"]) / original["time"] * 100, 2
            ),
            "original_memory": original["memory"],
            "optimized_memory": optimized["memory"],
            "memory_change": round(
                (optimized["memory"] - original["memory"]) / original["memory"] * 100, 2
            )
        }