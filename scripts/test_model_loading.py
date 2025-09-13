#!/usr/bin/env python3
"""
Test script to verify TST model loading
"""
import os
import sys
import torch

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'tst_app'))

def test_model_loading():
    print("🔍 Testing TST model loading...")
    
    # Check if model file exists
    model_path = os.path.join(project_root, 'models', 'tst_model_fp32.pth')
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("   Available models:")
        models_dir = os.path.join(project_root, 'models')
        if os.path.exists(models_dir):
            for f in os.listdir(models_dir):
                if f.endswith('.pth'):
                    print(f"   - {f}")
        return False
    
    print(f"✅ Model file found: {model_path}")
    
    try:
        # Import classes
        from tstplus import TSTPlus, _TSTBackbone, _TSTEncoderLayer
        print("✅ TST classes imported successfully")
        
        # Add to namespace for pickle
        sys.modules[__name__]._TSTBackbone = _TSTBackbone
        sys.modules[__name__]._TSTEncoderLayer = _TSTEncoderLayer
        sys.modules[__name__].TSTPlus = TSTPlus
        
        # Try loading
        print("🔄 Loading model...")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print("✅ Model loaded successfully!")
        
        # Check model type
        if hasattr(checkpoint, 'eval'):
            print("✅ Model is a complete PyTorch model")
            model = checkpoint
        else:
            print(f"ℹ️  Checkpoint type: {type(checkpoint)}")
            if isinstance(checkpoint, dict):
                print(f"ℹ️  Checkpoint keys: {list(checkpoint.keys())}")
        
        print("🎉 Model loading test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

if __name__ == "__main__":
    test_model_loading()