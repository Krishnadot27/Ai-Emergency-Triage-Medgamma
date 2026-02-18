"""
Performance Diagnostic Script
==============================
Run this to find what's causing slowness
"""

import time
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("="*70)
print("PERFORMANCE DIAGNOSTIC")
print("="*70)

# System info
print("\n[SYSTEM INFO]")
print(f"CPU cores: {psutil.cpu_count()}")
print(f"RAM available: {psutil.virtual_memory().available / 1024**3:.1f} GB")
print(f"RAM total: {psutil.virtual_memory().total / 1024**3:.1f} GB")
print(f"CPU usage: {psutil.cpu_percent(interval=1)}%")

# Check torch
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test model loading
print("\n" + "="*70)
print("[TEST 1] Model Loading Speed")
print("="*70)

model_name = "google/gemma-2b-it"

print(f"\nLoading {model_name}...")
start = time.time()

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    load_time_1 = time.time() - start
    print(f"✓ Tokenizer loaded in {load_time_1:.1f} seconds")
    
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    load_time_2 = time.time() - start
    print(f"✓ Model loaded in {load_time_2:.1f} seconds")
    
    total_load = load_time_1 + load_time_2
    
    if total_load > 180:
        print(f"⚠️  SLOW: {total_load:.1f}s (should be 60-120s)")
        print("   Possible causes: Slow disk, low RAM, downloading model")
    else:
        print(f"✓ Normal loading time: {total_load:.1f}s")
    
    # Test inference
    print("\n" + "="*70)
    print("[TEST 2] Inference Speed")
    print("="*70)
    
    test_prompt = "Patient: chest pain. Triage level:"
    
    print("\nGenerating response (this is the slow part)...")
    start = time.time()
    
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    inference_time = time.time() - start
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"✓ Inference completed in {inference_time:.1f} seconds")
    print(f"\nResponse length: {len(response)} characters")
    
    if inference_time > 60:
        print(f"\n⚠️  VERY SLOW: {inference_time:.1f}s (should be 10-30s)")
        print("\n   LIKELY CAUSES:")
        print("   1. max_new_tokens too high (should be 100-150)")
        print("   2. do_sample=True (should be False for speed)")
        print("   3. CPU is slow or overloaded")
        print("   4. Thermal throttling (computer too hot)")
    elif inference_time > 30:
        print(f"\n⚠️  SLOW: {inference_time:.1f}s (should be 10-30s)")
        print("   Try: Reduce max_new_tokens, use do_sample=False")
    else:
        print(f"\n✓ Normal inference time")
    
    # Memory check
    print("\n" + "="*70)
    print("[TEST 3] Memory Usage")
    print("="*70)
    
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024 / 1024
    
    print(f"\nCurrent memory usage: {mem_mb:.0f} MB")
    
    if mem_mb > 8000:
        print("⚠️  HIGH: May be swapping to disk (very slow)")
        print("   Solution: Close other apps, restart computer")
    else:
        print("✓ Memory usage acceptable")
    
    # Summary
    print("\n" + "="*70)
    print("DIAGNOSIS SUMMARY")
    print("="*70)
    
    print(f"\nTotal time for analysis: ~{total_load + inference_time:.1f} seconds")
    
    if total_load + inference_time > 200:
        print("\n❌ PROBLEM IDENTIFIED: System is too slow")
        print("\nRECOMMENDED FIXES (in order):")
        print("1. Apply speed optimization script")
        print("2. Close other applications")
        print("3. Restart computer")
        print("4. Check CPU isn't thermal throttling")
        print("5. Use edge version (inference_edge.py)")
    else:
        print("\n✓ System performance is acceptable")
        print("\nIf still slow, check:")
        print("1. Model reloading on every request")
        print("2. Running out of RAM (swapping)")
        print("3. Other processes using CPU")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\nTroubleshooting:")
    print("1. Check internet connection (first time downloads model)")
    print("2. Check disk space (need 10GB free)")
    print("3. Check if model_name is correct")

print("\n" + "="*70)