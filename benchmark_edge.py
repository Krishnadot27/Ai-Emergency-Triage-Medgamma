import time
import psutil
from inference import gemma_model
from inference_edge import gemma_edge

# Test case
test_case = {
    'age': 65,
    'gender': 'Male',
    'symptoms': ['chest pain'],
    'clinical_notes': 'Crushing chest pain for 30 min'
}

print("="*60)
print("EDGE vs STANDARD BENCHMARK")
print("="*60)

# Standard version
print("\n[STANDARD VERSION]")
process = psutil.Process()
mem_before = process.memory_info().rss / 1024 / 1024

gemma_model.load_model()
start = time.time()
result_std = gemma_model.analyze_patient(test_case)
time_std = time.time() - start

mem_after = process.memory_info().rss / 1024 / 1024
mem_std = mem_after - mem_before

print(f"Time: {time_std:.2f}s")
print(f"Memory: {mem_std:.0f} MB")
print(f"Result: {result_std['risk_level']}")

# Edge version
print("\n[EDGE VERSION]")
mem_before = process.memory_info().rss / 1024 / 1024

gemma_edge.load_model()
start = time.time()
result_edge = gemma_edge.analyze_patient(test_case)
time_edge = time.time() - start

mem_after = process.memory_info().rss / 1024 / 1024
mem_edge = mem_after - mem_before

print(f"Time: {time_edge:.2f}s")
print(f"Memory: {mem_edge:.0f} MB")
print(f"Result: {result_edge['risk_level']}")

# Comparison
print("\n" + "="*60)
print("IMPROVEMENT")
print("="*60)
print(f"Speed: {(time_std/time_edge):.1f}x faster")
print(f"Memory: {(mem_std-mem_edge):.0f} MB saved")
print("="*60)