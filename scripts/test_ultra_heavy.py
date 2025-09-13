#!/usr/bin/env python3
"""
Ultra Heavy CPU Test for Raspberry Pi
Test the maximum CPU consumption features
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

def monitor_cpu_continuous(duration=30):
    """Continuous CPU monitoring"""
    cpu_readings = []
    start_time = time.time()
    
    print(f"🔥 Monitoring CPU for {duration} seconds...")
    while time.time() - start_time < duration:
        cpu_percent = psutil.cpu_percent(interval=0.5)
        cpu_readings.append(cpu_percent)
        print(f"  Current CPU: {cpu_percent:.1f}%")
        
        # Check if we're hitting the target
        if cpu_percent >= 80:
            print(f"  🎯 TARGET REACHED: {cpu_percent:.1f}% CPU!")
    
    if cpu_readings:
        avg_cpu = sum(cpu_readings) / len(cpu_readings)
        max_cpu = max(cpu_readings)
        min_cpu = min(cpu_readings)
        
        print(f"\n📊 CPU Usage Results:")
        print(f"  - Average: {avg_cpu:.1f}%")
        print(f"  - Peak: {max_cpu:.1f}%")
        print(f"  - Minimum: {min_cpu:.1f}%")
        print(f"  - Target: 80-95% (Ultra Research)")
        
        if avg_cpu >= 80:
            print(f"  ✅ SUCCESS: Ultra-heavy processing working!")
        elif avg_cpu >= 60:
            print(f"  ⚠️  GOOD: Heavy processing active, can boost more")
        else:
            print(f"  ❌ ISSUE: CPU usage still too low")

def test_ultra_heavy_processing():
    """Test ultra-heavy processing with realistic data flow"""
    print("🔬 Ultra Heavy Processing Test for Raspberry Pi")
    print("=" * 55)
    
    # Import after path setup
    from main import TST_Detector
    
    # Model path
    model_path = os.path.join(project_root, 'models', 'tst_model_fp32.pth')
    
    print("\n🚀 Loading Ultra Heavy Model...")
    detector = TST_Detector(model_path)
    load_success = detector.load_model()
    
    if not load_success:
        print("❌ Model loading failed")
        return
    
    print(f"✅ Model loaded successfully")
    print(f"📊 Profile: {detector.processing_config}")
    
    # Start CPU monitoring in background
    monitor_thread = threading.Thread(target=monitor_cpu_continuous, args=(30,))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Generate realistic continuous data stream
    import numpy as np
    
    print(f"\n🔥 Starting ultra-heavy continuous processing...")
    print(f"Simulating real network data processing...")
    
    start_time = time.time()
    total_predictions = 0
    
    try:
        while time.time() - start_time < 25:  # Run for 25 seconds
            # Simulate realistic network data
            test_features = np.random.randn(400) * 100  # Network packet counts
            
            # Heavy processing detection
            pred, conf = detector.detect_ddos(test_features)
            total_predictions += 1
            
            if total_predictions % 5 == 0:
                elapsed = time.time() - start_time
                rate = total_predictions / elapsed
                print(f"  Processed {total_predictions} predictions at {rate:.1f}/sec")
            
            # Small delay to simulate realistic timing
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Test stopped by user")
    
    # Wait for monitoring to complete
    monitor_thread.join()
    
    total_time = time.time() - start_time
    print(f"\n📈 Performance Summary:")
    print(f"  - Total predictions: {total_predictions}")
    print(f"  - Processing rate: {total_predictions/total_time:.1f} predictions/sec")
    print(f"  - Total runtime: {total_time:.1f} seconds")
    print(f"  - Expected CPU: 80-95% for ultra profile")
    
    print(f"\n💡 Configuration Details:")
    config = detector.processing_config
    print(f"  - Augmentation: {config.get('augmentation_count')}x")
    print(f"  - Ensemble passes: {config.get('ensemble_passes')}x") 
    print(f"  - Heavy features: {'✅' if config.get('enable_heavy_features') else '❌'}")
    print(f"  - Statistics: {'✅' if config.get('enable_statistics') else '❌'}")

if __name__ == "__main__":
    test_ultra_heavy_processing()