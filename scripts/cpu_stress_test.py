#!/usr/bin/env python3
"""
Ultra Heavy Model CPU Test
Test the 60%+ CPU usage configuration
"""
import os
import sys
import time
import psutil
import threading

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'tst_app'))

def monitor_cpu(duration=10):
    """Monitor CPU usage during test"""
    cpu_readings = []
    start_time = time.time()
    
    print("🔥 Monitoring CPU usage...")
    while time.time() - start_time < duration:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_readings.append(cpu_percent)
        if len(cpu_readings) % 10 == 0:
            print(f"  Current CPU: {cpu_percent:.1f}%")
    
    avg_cpu = sum(cpu_readings) / len(cpu_readings)
    max_cpu = max(cpu_readings)
    
    print(f"\n📊 CPU Usage Results:")
    print(f"  - Average CPU: {avg_cpu:.1f}%")
    print(f"  - Peak CPU: {max_cpu:.1f}%")
    print(f"  - Target: 60%+ (Research Grade)")
    
    if avg_cpu >= 60:
        print(f"  ✅ SUCCESS: Reached research-grade CPU usage!")
    else:
        print(f"  ⚠️  Need higher: {60 - avg_cpu:.1f}% more CPU needed")

def stress_test_heavy_model():
    """Test ultra-heavy model configuration"""
    print("🔬 Ultra Heavy TST Model CPU Stress Test")
    print("=" * 50)
    
    # Import after path setup
    from main import TST_Detector
    
    # Model path
    model_path = os.path.join(project_root, 'models', 'tst_model_fp32.pth')
    
    print("\n🚀 Loading Ultra Heavy Research Model...")
    detector = TST_Detector(model_path, research_mode=True)
    load_success = detector.load_model()
    
    if not load_success:
        print("❌ Model loading failed")
        return
    
    print(f"✅ Model loaded successfully")
    print(f"🔬 Research mode: {detector.research_mode}")
    
    # Generate test data
    import numpy as np
    test_features = np.random.randn(400).reshape(1, -1)
    
    print(f"\n🔥 Starting heavy computation test...")
    print(f"This should push CPU to 60%+ like your original research!")
    
    # Start CPU monitoring in background
    monitor_thread = threading.Thread(target=monitor_cpu, args=(15,))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Heavy computation loop
    start_time = time.time()
    predictions = []
    
    for i in range(50):  # 50 heavy predictions
        start_pred = time.time()
        pred, conf = detector.detect_ddos(test_features)
        pred_time = (time.time() - start_pred) * 1000
        
        predictions.append((pred, conf, pred_time))
        
        if i % 10 == 0:
            print(f"  Prediction {i+1}: {pred} (conf: {conf:.3f}) in {pred_time:.1f}ms")
    
    total_time = time.time() - start_time
    avg_pred_time = sum(p[2] for p in predictions) / len(predictions)
    
    print(f"\n📈 Performance Results:")
    print(f"  - 50 predictions in {total_time:.1f} seconds")
    print(f"  - Average per prediction: {avg_pred_time:.1f}ms")
    print(f"  - Research mode overhead: 5x data augmentation + ensemble")
    
    # Wait for CPU monitoring to finish
    monitor_thread.join()
    
    print(f"\n💡 Model Configuration:")
    print(f"  - d_model: 256 (16x production)")
    print(f"  - n_heads: 32 (4x production)")  
    print(f"  - n_layers: 6 (3x production)")
    print(f"  - d_ff: 1024 (4x production)")
    print(f"  - Heavy processing: Data augmentation + ensemble voting")

if __name__ == "__main__":
    stress_test_heavy_model()