#!/usr/bin/env python3
"""
Test Research Mode TST Configuration
Demonstrates heavy model vs light model performance
"""
import os
import sys
import time

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'tst_app'))

def test_research_mode():
    """Test the research mode configuration"""
    print("🧪 Testing TST Research Mode Configuration")
    print("=" * 50)
    
    # Import after path setup
    from main import TST_Detector
    
    # Model path
    model_path = os.path.join(project_root, 'models', 'tst_model_fp32.pth')
    
    # Test Production Mode (Default)
    print("\n⚡ Testing PRODUCTION Mode:")
    print("-" * 30)
    
    detector_prod = TST_Detector(model_path, research_mode=False)
    load_success = detector_prod.load_model()
    print(f"Model loaded: {load_success}")
    print(f"Research mode: {detector_prod.research_mode}")
    
    # Test Research Mode
    print("\n🔬 Testing RESEARCH Mode:")
    print("-" * 30)
    
    detector_research = TST_Detector(model_path, research_mode=True)
    load_success = detector_research.load_model()
    print(f"Model loaded: {load_success}")
    print(f"Research mode: {detector_research.research_mode}")
    
    # Simulate detection
    print("\n📊 Performance Comparison:")
    print("-" * 30)
    
    import numpy as np
    dummy_features = np.random.randn(400).reshape(1, -1)
    
    # Production mode detection
    start_time = time.time()
    try:
        pred_prod, conf_prod = detector_prod.detect_ddos(dummy_features)
        prod_time = (time.time() - start_time) * 1000
        print(f"Production: {pred_prod} (conf: {conf_prod:.3f}) in {prod_time:.1f}ms")
    except Exception as e:
        print(f"Production error: {e}")
    
    # Research mode detection
    start_time = time.time()
    try:
        pred_research, conf_research = detector_research.detect_ddos(dummy_features)
        research_time = (time.time() - start_time) * 1000
        print(f"Research: {pred_research} (conf: {conf_research:.3f}) in {research_time:.1f}ms")
    except Exception as e:
        print(f"Research error: {e}")
    
    if 'prod_time' in locals() and 'research_time' in locals():
        print(f"\n💡 Speed difference: Research is ~{research_time/prod_time:.1f}x slower")
        print("   But provides higher accuracy with more complex analysis")

if __name__ == "__main__":
    test_research_mode()