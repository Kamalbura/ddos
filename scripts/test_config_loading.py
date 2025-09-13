#!/usr/bin/env python3
"""
Test TST Configuration Loading
Quick test to verify configuration system works
"""
import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'tst_app'))

def test_config_loading():
    print("🧪 Testing TST Configuration System")
    print("=" * 50)
    
    # Test configuration manager
    from common.performance_config import PerformanceConfig
    
    config = PerformanceConfig()
    print(f"✅ Configuration loaded")
    print(f"📊 Current profile: {config.current_profile}")
    
    # Test TST detector initialization
    print(f"\n🔬 Testing TST Detector initialization...")
    from main import TST_Detector
    
    model_path = os.path.join(project_root, 'models', 'tst_model_fp32.pth')
    detector = TST_Detector(model_path)
    
    print(f"✅ TST Detector initialized")
    print(f"📊 Profile: {detector.processing_config}")
    print(f"🧠 Model config: d_model={detector.config['d_model']}, n_heads={detector.config['n_heads']}")
    
    print(f"\n🎯 Configuration system working properly!")

if __name__ == "__main__":
    test_config_loading()