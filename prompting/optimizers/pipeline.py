from agents import (
    Instructor,
    Classifier,
    Optimizer,
    Implementer,
    Insighter,
    Refiner
)
from utils import benchmark_code
# Add to top of file
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimizer.log'),
        logging.StreamHandler()
    ]
)


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

    import logging

    def optimize(self, original_code: str) -> dict:
        """
        Execute full optimization pipeline with error tracking:
        1. Context Setup → 2. Classification → 3. Analysis → 
        4. Implementation → 5. Refinement → 6. Insights
        """
        logger = logging.getLogger(self.__class__.__name__)
        results = {}
        
        try:
            logger.info("Starting optimization pipeline")
            logger.debug(f"Original code:\n{original_code}")

            # Stage 1: Context Establishment
            try:
                logger.info("Stage 1/6: Setting optimization context")
                results["context"] = self.agents["instructor"].set_optimization_context(original_code)
                logger.debug(f"Context established: {results['context']}...")
            except Exception as e:
                logger.error("Failed to establish optimization context")
                raise RuntimeError(f"Context setup failed: {str(e)}") from e

            # Stage 2: Issue Classification
            try:
                logger.info("Stage 2/6: Classifying code inefficiencies")
                results["classification"] = self.agents["classifier"].classify_inefficiency(original_code)
                logger.debug(f"Classification results: {results['classification']}")
            except Exception as e:
                logger.error("Classification failed", exc_info=True)
                raise RuntimeError(f"Classification error: {str(e)}") from e

            # Stage 3: CoT Analysis
            try:
                logger.info("Stage 3/6: Generating optimization plan")
                results["analysis"] = self.agents["optimizer"].generate_optimization_plan(original_code)
                logger.debug(f"Analysis summary: {results['analysis']}...")
            except Exception as e:
                logger.error("Optimization analysis failed")
                raise RuntimeError(f"Analysis error: {str(e)}") from e

            # Stage 4: Code Implementation
            try:
                logger.info("Stage 4/6: Generating optimized code")
                results["optimized_code"] = self.agents["implementer"].implement_optimizations(
                    original_code, results["analysis"]
                )
                logger.debug(f"Optimized code generated ({len(results['optimized_code'])} chars)")
            except Exception as e:
                logger.error("Code implementation failed")
                raise RuntimeError(f"Implementation error: {str(e)}") from e

            # Stage 5: Benchmark & Refinement
            try:
                logger.info("Stage 5/6: Benchmarking and refinement")
                original_benchmark = benchmark_code(original_code)
                optimized_benchmark = benchmark_code(results["optimized_code"])
                results["benchmark"] = self._calculate_improvements(original_benchmark, optimized_benchmark)
                
                logger.info(f"Initial benchmark: {results['benchmark']}")
                
                if (results["benchmark"]["time_improvement"] < 20):
                    logger.warning("Optimization targets not met - starting refinement")
                    results["optimized_code"] = self.agents["refiner"].refine_implementation(
                        original_code,
                        results["optimized_code"],
                        results["benchmark"]
                    )
                    results["benchmark"] = self._calculate_improvements(
                        original_benchmark, 
                        benchmark_code(results["optimized_code"])
                    )
                    logger.info(f"Post-refinement benchmark: {results['benchmark']}")
            except Exception as e:
                logger.error("Benchmarking/refinement failed", exc_info=True)
                raise RuntimeError(f"Benchmark error: {str(e)}") from e

            # Stage 6: Knowledge Generation
            try:
                logger.info("Stage 6/6: Generating insights")
                results["insights"] = self.agents["insighter"].generate_optimization_insights(
                    results["analysis"]
                )
                logger.debug(f"Insights generated: {results['insights'][:200]}...")
            except Exception as e:
                logger.error("Insight generation failed")
                raise RuntimeError(f"Insight error: {str(e)}") from e

            logger.info("Optimization pipeline completed successfully")
            return results

        except Exception as e:
            logger.critical("Optimization pipeline failed", exc_info=True)
            raise  # Re-raise for external handling

    def _calculate_improvements(self, original: dict, optimized: dict) -> dict:
        """Calculate performance metrics with safe division handling"""
        original_time = original["time"]
        optimized_time = optimized["time"]
        original_mem = original["memory"]
        optimized_mem = optimized["memory"]

        # Calculate time improvement with zero handling
        if original_time <= 0:
            time_improvement = 100.0 if optimized_time > 0 else 0.0
        else:
            time_improvement = ((original_time - optimized_time) / original_time) * 100

        # Calculate memory change with zero handling
        if original_mem <= 0:
            mem_change = -100.0 if optimized_mem > 0 else 0.0
        else:
            mem_change = ((optimized_mem - original_mem) / original_mem) * 100

        return {
            "original_time": round(original_time, 4),
            "optimized_time": round(optimized_time, 4),
            "time_improvement": round(time_improvement, 2),
            "original_memory": original_mem,
            "optimized_memory": optimized_mem,
            "memory_change": round(mem_change, 2)
    }
        
# Test Code
if __name__ == "__main__":
    from sys import path
    path.append('.')
    from config import LLM_CONFIG as llm_config
    # Sample inefficient code to test
    test_code = """
            def calculate_pairs(nums):
                result = []
                for i in range(len(nums)):
                    for j in range(len(nums)):
                        if i != j:
                            result.append((nums[i], nums[j]))
                return result

            # Test case
            print(len(calculate_pairs([1, 2, 3, 4])))
                """

    # Initialize pipeline
    pipeline = OptimizationPipeline(llm_config)
    
    try:
        # Run optimization
        results = pipeline.optimize(test_code)
        
        # Print results
        print("\n=== Optimization Results ===")
        print(f"Time Improvement: {results['benchmark']['time_improvement']}%")
        print(f"Memory Change: {results['benchmark']['memory_change']}%")
        
        print("\n=== Optimized Code ===")
        print(results["optimized_code"])
        
        print("\n=== Insights ===")
        print(results["insights"])
        
    except Exception as e:
        print(f"Optimization failed: {str(e)}")