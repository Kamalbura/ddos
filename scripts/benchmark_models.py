#!/usr/bin/env python3
"""
TST Model Complexity Comparison Script
Compare Research (Heavy) vs Production (Light) models
"""
import os
import sys
import time
import torch
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'tst_app'))

def benchmark_model_configs():
    """Compare different model configurations"""
    print("🔬 TST Model Complexity Comparison")
    print("=" * 60)
    
    # Import after path setup
    from tstplus import TSTPlus
    
    # Research Configuration (Heavy)
    research_config = {
        'c_in': 1,
        'c_out': 2,
        'seq_len': 400,
        'd_model': 128,      # Heavy: 128 vs 64
        'n_heads': 16,       # Heavy: 16 vs 8
        'n_layers': 3,       # Heavy: 3 vs 2
        'd_ff': 512,         # Heavy: 512 vs 256
        'dropout': 0.1,
        'max_seq_len': 512,
        'norm': 'BatchNorm',
        'attn_dropout': 0.1,
        'res_attention': True,
        'pre_norm': False,
        'pe': 'zeros',
        'learn_pe': True
    }
    
    # Production Configuration (Light)
    production_config = {
        'c_in': 1,
        'c_out': 2,
        'seq_len': 400,
        'd_model': 64,       # Light: 64
        'n_heads': 8,        # Light: 8
        'n_layers': 2,       # Light: 2
        'd_ff': 256,         # Light: 256
        'dropout': 0.1
    }
    
    configs = [
        ("🔬 RESEARCH (Heavy)", research_config),
        ("⚡ PRODUCTION (Light)", production_config)
    ]
    
    results = []
    
    for config_name, config in configs:
        print(f"\n{config_name} Model:")
        print("-" * 30)
        
        try:
            # Create model
            model = TSTPlus(**config)
            model.eval()
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            param_memory_mb = total_params * 4 / (1024 * 1024)
            
            print(f"Architecture:")
            print(f"  - d_model: {config['d_model']}")
            print(f"  - n_heads: {config['n_heads']}")
            print(f"  - n_layers: {config['n_layers']}")
            print(f"  - d_ff: {config.get('d_ff', 256)}")
            print(f"Parameters:")
            print(f"  - Total: {total_params:,}")
            print(f"  - Memory: ~{param_memory_mb:.1f} MB")
            
            # Benchmark inference
            dummy_input = torch.randn(1, 1, 400)  # [batch, features, sequence]
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = model(dummy_input)
            
            # Actual benchmark
            times = []
            with torch.no_grad():
                for _ in range(100):
                    start_time = time.time()
                    output = model(dummy_input)
                    end_time = time.time()
                    times.append((end_time - start_time) * 1000)  # ms
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            print(f"Performance:")
            print(f"  - Avg inference: {avg_time:.2f} ± {std_time:.2f} ms")
            print(f"  - Output shape: {output.shape}")
            
            results.append({
                'name': config_name,
                'params': total_params,
                'memory_mb': param_memory_mb,
                'avg_time_ms': avg_time,
                'config': config
            })
            
        except Exception as e:
            print(f"❌ Error testing {config_name}: {e}")
    
    # Comparison
    if len(results) == 2:
        print("\n" + "=" * 60)
        print("📊 COMPARISON")
        print("=" * 60)
        
        research = results[0]
        production = results[1]
        
        param_ratio = research['params'] / production['params']
        memory_ratio = research['memory_mb'] / production['memory_mb']
        time_ratio = research['avg_time_ms'] / production['avg_time_ms']
        
        print(f"Parameter Difference: {param_ratio:.1f}x more in Research")
        print(f"Memory Difference: {memory_ratio:.1f}x more in Research") 
        print(f"Speed Difference: {time_ratio:.1f}x slower in Research")
        
        print(f"\n🔬 Research Model: {research['params']:,} params, {research['avg_time_ms']:.1f}ms")
        print(f"⚡ Production Model: {production['params']:,} params, {production['avg_time_ms']:.1f}ms")
        
        print(f"\n💡 Trade-off:")
        print(f"  - Research: Higher accuracy, {param_ratio:.1f}x more parameters, {time_ratio:.1f}x slower")
        print(f"  - Production: Good accuracy, {param_ratio:.1f}x fewer parameters, {time_ratio:.1f}x faster")

if __name__ == "__main__":
    benchmark_model_configs()