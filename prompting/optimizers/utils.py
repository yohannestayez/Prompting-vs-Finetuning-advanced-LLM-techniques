import time
import tracemalloc

def benchmark_code(code: str) -> dict:
    """Measures execution time and memory usage"""
    tracemalloc.start()
    start_time = time.time()
    
    # Execute code
    try:
        exec(code, {})
    except:
        pass
    
    return {
        "time": time.time() - start_time,
        "memory": tracemalloc.get_traced_memory()[1]
    }