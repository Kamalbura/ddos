#!/usr/bin/env python3
"""
Test TST_Detector model loading directly
"""
import os
import sys

# Add paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'tst_app'))

from main import TST_Detector

def test_tst_detector():
    print("🔍 Testing TST_Detector model loading...")
    
    model_path = os.path.join(project_root, 'models', 'tst_model_fp32.pth')
    detector = TST_Detector(model_path)
    
    print("✅ TST_Detector instantiated successfully")
    
    result = detector.load_model()
    print(f"Model loading result: {result}")
    
    if result:
        print("🎉 TST_Detector model loading test passed!")
    else:
        print("❌ TST_Detector model loading test failed!")
    
    return result

if __name__ == "__main__":
    test_tst_detector()