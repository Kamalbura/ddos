#!/usr/bin/env python3
"""
TST Model Quantization Script
Converts FP32 TST model to optimized INT8 model for better performance on Raspberry Pi
"""

import torch
import os
import sys
import logging

# Add project root to path to allow importing from tst_app
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

try:
    from tst_app.tstplus import TSTPlus
except ImportError:
    print("Error: Could not import TSTPlus. Please ensure tstplus.py is in the tst_app directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quantize_tst_model():
    """
    Load FP32 TST model, apply dynamic quantization, and save optimized INT8 model
    """
    models_dir = os.path.join(project_root, 'models')
    fp32_path = os.path.join(models_dir, 'tst_model_fp32.pth')
    int8_path = os.path.join(models_dir, 'tst_model_int8.pth')

    if not os.path.exists(fp32_path):
        logger.error(f"FP32 model not found at {fp32_path}")
        logger.info("Please ensure the TST model file is placed in the models/ directory")
        return False

    try:
        logger.info("Loading FP32 TST model...")
        
        # Load the model
        model_fp32 = torch.load(fp32_path, map_location=torch.device('cpu'))
        
        # Ensure model is in evaluation mode
        model_fp32.eval()
        
        logger.info(f"Model loaded successfully. Type: {type(model_fp32)}")
        
        # Print model structure for debugging
        if hasattr(model_fp32, 'state_dict'):
            state_dict = model_fp32.state_dict()
            logger.info(f"Model has {len(state_dict)} parameters")
            
            # Print first few parameter names
            param_names = list(state_dict.keys())[:5]
            logger.debug(f"First few parameters: {param_names}")

        logger.info("Applying INT8 dynamic quantization...")
        
        # Apply dynamic quantization
        # This quantizes linear layers to INT8 while keeping activations in FP32
        model_int8 = torch.quantization.quantize_dynamic(
            model_fp32,  # model
            {torch.nn.Linear},  # a set of layers to dynamically quantize
            dtype=torch.qint8  # the target dtype for quantized weights
        )

        logger.info(f"Quantization complete. Saving to: {int8_path}")
        
        # Save the quantized model
        torch.save(model_int8, int8_path)
        
        logger.info("✅ TST model quantization successful!")
        
        # Compare model sizes
        if os.path.exists(fp32_path) and os.path.exists(int8_path):
            fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)  # MB
            int8_size = os.path.getsize(int8_path) / (1024 * 1024)  # MB
            compression_ratio = fp32_size / int8_size if int8_size > 0 else 1
            
            logger.info(f"Model size comparison:")
            logger.info(f"  FP32 model: {fp32_size:.2f} MB")
            logger.info(f"  INT8 model: {int8_size:.2f} MB")
            logger.info(f"  Compression ratio: {compression_ratio:.2f}x")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during quantization: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quantized_model():
    """
    Test the quantized model with dummy data to ensure it works
    """
    logger.info("Testing quantized model...")
    
    int8_path = os.path.join(project_root, 'models', 'tst_model_int8.pth')
    
    if not os.path.exists(int8_path):
        logger.error("Quantized model not found. Please run quantization first.")
        return False
    
    try:
        # Load quantized model
        model_int8 = torch.load(int8_path, map_location=torch.device('cpu'))
        model_int8.eval()
        
        # Create dummy input data
        # TST expects [batch_size, n_vars, seq_len]
        batch_size = 1
        n_vars = 1  # Single feature (packet count)
        seq_len = 400  # LOOKBACK value for TST
        
        dummy_input = torch.randn(batch_size, n_vars, seq_len)
        
        logger.info(f"Testing with input shape: {dummy_input.shape}")
        
        # Test inference
        with torch.no_grad():
            start_time = torch.utils.benchmark.Timer(
                stmt='model_int8(dummy_input)',
                globals={'model_int8': model_int8, 'dummy_input': dummy_input}
            ).blocked_autorange()
            
            output = model_int8(dummy_input)
            
        logger.info(f"✅ Quantized model test successful!")
        logger.info(f"Output shape: {output.shape}")
        logger.info(f"Average inference time: {start_time.mean * 1000:.2f} ms")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing quantized model: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main function to run quantization process
    """
    print("🚀 TST Model Quantization Tool")
    print("=" * 50)
    
    # Check if models directory exists
    models_dir = os.path.join(project_root, 'models')
    if not os.path.exists(models_dir):
        print(f"Error: Models directory not found at {models_dir}")
        print("Please run this script from the project root directory")
        sys.exit(1)
    
    # Check PyTorch availability
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
    except ImportError:
        print("Error: PyTorch not installed. Please install it first.")
        sys.exit(1)
    
    # Run quantization
    logger.info("Starting TST model quantization...")
    
    if quantize_tst_model():
        logger.info("Quantization completed successfully!")
        
        # Test the quantized model
        if test_quantized_model():
            logger.info("🎉 All operations completed successfully!")
            logger.info("The optimized INT8 model is ready for deployment on Raspberry Pi")
        else:
            logger.warning("Quantization succeeded but model testing failed")
    else:
        logger.error("❌ Quantization failed")
        sys.exit(1)

if __name__ == '__main__':
    main()