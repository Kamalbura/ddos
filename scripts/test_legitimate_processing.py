#!/usr/bin/env python3
"""
Test Legitimate Heavy Processing
Test the research-grade processing without wasteful CPU spinning
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

def monitor_cpu_usage(duration=20):
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
    
    if avg_cpu >= 90:
        print("🔥 EXTREME: Over 90% CPU - may be excessive")
    elif avg_cpu >= 60:
        print("✅ OPTIMAL: Good research-grade CPU usage")
    elif avg_cpu >= 30:
        print("⚠️ MEDIUM: Moderate CPU usage")
    else:
        print("❌ LOW: CPU usage too low for research mode")
    print("=" * 50)
    
    return avg_cpu, max_cpu

def legitimate_research_processing():
    """Simulate legitimate research-grade processing like the TST app"""
    import numpy as np
    
    print("🔬 Starting LEGITIMATE research processing...")
    
    # Simulate feature history and statistical cache (like TST app)
    feature_history = list(np.random.random(30))  # 30 historical features
    statistical_cache = {}
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
            
            # 1. Advanced feature engineering (legitimate)
            if len(feature_history) > 10:
                recent_features = np.array(feature_history[-10:])
                statistical_cache['rolling_mean'] = np.mean(recent_features)
                statistical_cache['rolling_std'] = np.std(recent_features)
                statistical_cache['trend'] = np.polyfit(range(len(recent_features)), recent_features, 1)[0]
                
                # Frequency domain analysis
                if len(recent_features) >= 8:
                    fft_features = np.fft.fft(recent_features)
                    statistical_cache['dominant_freq'] = np.argmax(np.abs(fft_features[1:len(recent_features)//2])) + 1
                    statistical_cache['spectral_energy'] = np.sum(np.abs(fft_features)**2)
            
            # 2. Multi-model ensemble processing (legitimate)
            predictions = []
            confidences = []
            
            # Simulate research feature data
            current_feature = np.random.random()
            feature_history.append(current_feature)
            if len(feature_history) > 50:
                feature_history = feature_history[-50:]
                
            # Enhanced features with temporal analysis
            enhanced_features = np.array(feature_history[-20:])  # Last 20 features
            
            # Multi-scale moving averages (research technique)
            for window in [3, 5, 10, 15]:
                if len(feature_history) >= window:
                    ma = np.convolve(feature_history[-window:], np.ones(window)/window, mode='valid')[-1]
                    enhanced_features = enhanced_features * (1 + ma * 0.1)
            
            # 3. Multiple detection passes with feature variations (legitimate)
            for variation in range(3):  # 3 feature variations
                if variation == 0:
                    test_features = enhanced_features
                elif variation == 1:
                    # Frequency domain emphasis
                    fft_features = np.fft.fft(enhanced_features).real
                    test_features = enhanced_features * 0.7 + fft_features * 0.3
                else:
                    # Statistical normalization
                    if 'rolling_std' in statistical_cache and statistical_cache['rolling_std'] > 0:
                        test_features = (enhanced_features - statistical_cache.get('rolling_mean', 0)) / statistical_cache['rolling_std']
                    else:
                        test_features = enhanced_features
                
                # Simulate heavy model inference (legitimate computation)
                for _ in range(5):  # 5 ensemble passes per variation
                    # Legitimate model operations (matrix multiplications, etc.)
                    weights = np.random.random((20, 10))
                    hidden = np.dot(test_features.reshape(1, -1)[:, :20], weights)
                    hidden = np.tanh(hidden)  # Activation
                    output = np.mean(hidden)
                    
                    pred = output > 0.5
                    conf = abs(output - 0.5) + 0.5
                    
                    predictions.append(pred)
                    confidences.append(conf)
            
            # 4. Advanced ensemble decision (legitimate)
            attack_votes = sum(predictions)
            base_confidence = np.mean(confidences)
            
            if iteration % 100 == 0:
                print(f"Processed {iteration} samples | Accuracy features computed | CPU load: research-grade")
            
            # Small sleep to allow monitoring (realistic processing rate)
            time.sleep(0.01)  # 100 samples per second
            
    except KeyboardInterrupt:
        print(f"\n🛑 Legitimate processing stopped after {iteration} samples")

def main():
    print("🧪 Legitimate Heavy Processing Test")
    print("=" * 50)
    
    # Load performance configuration
    config = PerformanceConfig()
    
    print(f"Current profile: {config.current_profile}")
    config.print_profile_info()
    
    print("\nThis test simulates LEGITIMATE research-grade processing:")
    print("✅ Advanced feature engineering")
    print("✅ Multi-scale temporal analysis")  
    print("✅ Ensemble model inference")
    print("✅ Statistical pattern detection")
    print("❌ NO wasteful CPU spinning")
    
    input("\nPress Enter to start legitimate heavy processing test (20 seconds)...")
    
    # Start CPU monitoring
    monitor_thread = threading.Thread(target=lambda: monitor_cpu_usage(20), daemon=True)
    monitor_thread.start()
    
    # Start legitimate processing
    research_thread = threading.Thread(target=legitimate_research_processing, daemon=True)
    research_thread.start()
    
    # Wait for monitoring to complete
    monitor_thread.join()
    
    print("\n✅ Legitimate heavy processing test completed!")
    print("\nExpected results:")
    print("- 60-80% CPU: ✅ Perfect research-grade processing")
    print("- 40-60% CPU: ⚡ Good processing, could be heavier")
    print("- 90%+ CPU: ⚠️ May be too aggressive")
    print("- <40% CPU: 💡 Light processing mode")

if __name__ == "__main__":
    main()