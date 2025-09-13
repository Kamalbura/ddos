#!/usr/bin/env python3
"""
Test Ultra-Heavy Processing
Quick test to verify ultra profile loads and enables maximum CPU usage
"""
import os
import sys
import time
import threading
import psutil

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from common.performance_config import PerformanceConfig

def monitor_cpu_usage(duration=30):
    """Monitor CPU usage for specified duration"""
    print(f"📊 Monitoring CPU usage for {duration} seconds...")
    start_time = time.time()
    cpu_readings = []
    
    while time.time() - start_time < duration:
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_readings.append(cpu_percent)
        print(f"CPU: {cpu_percent:5.1f}% | Avg: {sum(cpu_readings)/len(cpu_readings):5.1f}%", end='\r')
    
    print()  # New line after monitoring
    avg_cpu = sum(cpu_readings) / len(cpu_readings)
    max_cpu = max(cpu_readings)
    
    print("=" * 50)
    print("📊 CPU Usage Results:")
    print(f"Average CPU: {avg_cpu:.1f}%")
    print(f"Maximum CPU: {max_cpu:.1f}%")
    print(f"Target for ultra: 80-95%")
    
    if avg_cpu >= 80:
        print("✅ SUCCESS: Ultra-heavy processing achieved target CPU usage!")
    elif avg_cpu >= 60:
        print("⚠️ PARTIAL: Heavy processing working but not ultra level")
    elif avg_cpu >= 30:
        print("⚠️ MEDIUM: Moderate CPU usage detected")
    else:
        print("❌ LOW: CPU usage still too low")
    print("=" * 50)
    
    return avg_cpu, max_cpu

def ultra_cpu_test():
    """Test ultra-heavy CPU processing similar to TST app"""
    import numpy as np
    
    print("🔥 Starting ULTRA CPU Test...")
    
    try:
        while True:
            # Massive CPU-intensive operations (same as TST app)
            for _ in range(500):  # 500 operations per cycle
                # Heavy matrix operations
                matrix_a = np.random.random((100, 100))
                matrix_b = np.random.random((100, 100))
                
                # Multiple expensive operations
                _ = np.linalg.inv(matrix_a + np.eye(100) * 0.01)  # Matrix inversion
                _ = np.dot(matrix_a, matrix_b)  # Matrix multiplication
                _ = np.fft.fft2(matrix_a)  # 2D FFT
                _ = np.linalg.svd(matrix_b)  # SVD decomposition
                _ = np.convolve(matrix_a.flatten(), matrix_b.flatten()[:50], mode='valid')
                
                # Additional CPU-intensive computations
                _ = np.power(matrix_a, 3)  # Element-wise power
                _ = np.exp(matrix_a * 0.1)  # Exponential
                _ = np.log(np.abs(matrix_a) + 1e-10)  # Logarithm
                
            time.sleep(0.001)  # Tiny sleep to prevent complete system freeze
    except KeyboardInterrupt:
        print("\n🛑 Ultra CPU test stopped")

def main():
    print("🧪 Ultra-Heavy Processing Test")
    print("=" * 50)
    
    # Load performance configuration
    config = PerformanceConfig()
    
    print(f"Current profile: {config.current_profile}")
    config.print_profile_info()
    
    # Test ultra profile specifically
    if config.current_profile != 'ultra':
        print(f"\n⚠️ Current profile is '{config.current_profile}', switching to 'ultra'...")
        if config.set_profile('ultra'):
            print("✅ Switched to ultra profile")
            config.print_profile_info()
        else:
            print("❌ Failed to switch to ultra profile")
            return
    
    # Get processing config
    processing = config.get_processing_config()
    print(f"\n🔧 Processing Configuration:")
    print(f"Heavy features enabled: {processing.get('enable_heavy_features', False)}")
    print(f"Augmentation count: {processing.get('augmentation_count', 1)}")
    print(f"Ensemble passes: {processing.get('ensemble_passes', 1)}")
    
    input("\nPress Enter to start ultra-heavy CPU test (30 seconds)...")
    
    # Start CPU monitoring in a separate thread
    monitor_thread = threading.Thread(target=lambda: monitor_cpu_usage(30), daemon=True)
    monitor_thread.start()
    
    # Start ultra-heavy processing
    cpu_test_thread = threading.Thread(target=ultra_cpu_test, daemon=True)
    cpu_test_thread.start()
    
    # Wait for monitoring to complete
    monitor_thread.join()
    
    print("\n✅ Ultra-heavy processing test completed!")
    print("If CPU usage was below 80%, the issue may be:")
    print("1. System limitations (CPU cores, cooling)")
    print("2. Python GIL limitations (try multiprocessing)")
    print("3. NumPy using optimized BLAS (may be more efficient)")

if __name__ == "__main__":
    main()