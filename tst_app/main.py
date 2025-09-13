#!/usr/bin/env python3
"""
TST Application for DDoS Detection  
Standalone application using Time Series Transformer model with configurable performance profiles
"""

import os
import sys
from threading import Thread
from queue import Queue
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from typing import Tuple

# Add project root to the Python path to find the 'common' module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import configuration manager
from common.performance_config import PerformanceConfig

from common.pipeline_components import (
    check_venv, capture, ddos_preprocess, ddos_mitigation,
    create_storage_tables, store_input, store_output, update_config,
    network_attack, detection_halt, detection_halt_window
)
# Import TST model definition from this directory
from tstplus import TSTPlus 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - TST - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- App Specific Settings ---
LOOKBACK = 400
WINDOW_SIZE = 0.60  

# Global performance configuration manager
PERFORMANCE_CONFIG = PerformanceConfig()

# Print current configuration at startup
print("🔧 TST Performance Configuration:")
PERFORMANCE_CONFIG.print_profile_info()  
PORT = 8001  # Use different port to avoid conflicts

# Research vs Production Model Configuration
# Global Research Configuration
# Set to True for heavy research models (60%+ CPU usage)
# Set to False for lightweight production models (5-20% CPU usage)
RESEARCH_MODE = True  # ENABLED: Research-grade heavy models for 60%+ CPU usage

# Research Model Configuration (Heavy - matches your original research)
RESEARCH_CONFIG = {
    'c_in': 1,
    'c_out': 2,
    'seq_len': LOOKBACK,
    'd_model': 256,      # ULTRA HEAVY: 256 (was 128) - 16x more than production
    'n_heads': 32,       # ULTRA HEAVY: 32 heads (was 16) - 4x attention heads  
    'n_layers': 6,       # ULTRA HEAVY: 6 layers (was 3) - 3x more layers
    'd_ff': 1024,        # ULTRA HEAVY: 1024 (was 512) - 4x feedforward dimension
    'dropout': 0.1,
    'max_seq_len': 512,
    'norm': 'BatchNorm',
    'attn_dropout': 0.05,  # Lower dropout for more computation
    'res_attention': True,
    'pre_norm': False,
    'pe': 'zeros',
    'learn_pe': True
}

# Production Model Configuration (Lightweight - current)
PRODUCTION_CONFIG = {
    'c_in': 1,
    'c_out': 2,
    'seq_len': LOOKBACK,
    'd_model': 64,       # Light: 64 for efficiency
    'n_heads': 8,        # Light: 8 for speed
    'n_layers': 2,       # Light: 2 for reduced computation
    'd_ff': 256,         # Light: 256 default
    'dropout': 0.1
}

class TST_Detector:
    """
    TST model wrapper with proper tensor handling for time series data
    """
    
    def __init__(self, model_path: str, profile: str = None):
        self.model_path = model_path
        self.model = None
        self.feature_buffer = []
        
        # Set performance profile
        if profile and PERFORMANCE_CONFIG.set_profile(profile):
            logger.info(f"Using performance profile: {profile}")
        
        self.config = PERFORMANCE_CONFIG.get_model_config()
        self.processing_config = PERFORMANCE_CONFIG.get_processing_config()
        
        logger.info(f"Initializing TST detector (Profile: {PERFORMANCE_CONFIG.current_profile})")
        
    def load_model(self) -> bool:
        """
        Load TST model using the same approach as the working test script
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Import required modules
        import sys
        import os
        
        try:
            if os.path.exists(self.model_path):
                # Import all TST classes to make them available during loading
                current_dir = os.path.dirname(os.path.abspath(__file__))
                sys.path.insert(0, current_dir)
                
                # Import the module to register all classes
                import tstplus
                
                logger.info("Loading TST model using working approach...")
                
                # Create a temporary module context for loading (same as test script)
                import types
                temp_module = types.ModuleType('temp_loader')
                
                # Add classes to temp module (exact same as working test)
                import torch.nn as nn
                for attr_name in dir(tstplus):
                    attr = getattr(tstplus, attr_name)
                    if isinstance(attr, type) and hasattr(attr, '__module__'):
                        setattr(temp_module, attr_name, attr)
                
                # Specifically add the main classes (same as test script)
                temp_module._TSTBackbone = tstplus._TSTBackbone
                temp_module._TSTEncoder = tstplus._TSTEncoder
                temp_module._TSTEncoderLayer = tstplus._TSTEncoderLayer
                temp_module.TSTPlus = tstplus.TSTPlus
                
                # Register temp module in sys.modules temporarily
                old_main = sys.modules.get('__main__')
                sys.modules['__main__'] = temp_module
                
                try:
                    # Load model (same as working test)
                    checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
                finally:
                    # Restore original __main__
                    if old_main is not None:
                        sys.modules['__main__'] = old_main
                
                # Handle different checkpoint formats
                if hasattr(checkpoint, 'eval'):
                    # It's the model directly
                    self.model = checkpoint
                    logger.info("Loaded complete PyTorch model")
                elif isinstance(checkpoint, dict):
                    # Handle different checkpoint formats
                    if 'model' in checkpoint:
                        # Checkpoint contains model under 'model' key
                        self.model = checkpoint['model']
                        logger.info("Loaded model from 'model' key in checkpoint")
                    elif 'state_dict' in checkpoint:
                        # Create model and load state dict - use current profile config
                        logger.info(f"Creating TST model with profile: {PERFORMANCE_CONFIG.current_profile}")
                        logger.info(f"Model config: d_model={self.config['d_model']}, n_heads={self.config['n_heads']}, n_layers={self.config['n_layers']}")
                        
                        self.model = tstplus.TSTPlus(**self.config)
                        self.model.load_state_dict(checkpoint['state_dict'])
                        logger.info("Loaded model state dict")
                    else:
                        # Assume it's a state dict directly
                        logger.info(f"Creating TST model with profile: {PERFORMANCE_CONFIG.current_profile}")
                        logger.info(f"Model config: d_model={self.config['d_model']}, n_heads={self.config['n_heads']}, n_layers={self.config['n_layers']}")
                        
                        self.model = tstplus.TSTPlus(**self.config)
                        self.model.load_state_dict(checkpoint)
                        logger.info("Loaded model from direct state dict")
                else:
                    # Unknown format, try to use as model directly
                    self.model = checkpoint
                    logger.info("Using checkpoint directly as model")
                
                self.model.eval()  # Set to evaluation mode
                
                # Calculate and log model complexity
                self._log_model_complexity()
                
                logger.info(f"TST model loaded successfully from {self.model_path}")
                return True
            else:
                logger.error(f"TST model not found at {self.model_path}")
                # Create a dummy model for testing - use current profile config
                logger.info(f"Creating dummy TST model for testing with profile: {PERFORMANCE_CONFIG.current_profile}")
                logger.info(f"Model config: d_model={self.config['d_model']}, n_heads={self.config['n_heads']}, n_layers={self.config['n_layers']}")
                
                self.model = tstplus.TSTPlus(**self.config)
                logger.warning("Using randomly initialized TST model for testing")
                return True
                
        except Exception as e:
            logger.error(f"Error loading TST model: {e}")
            return False
    
    def _log_model_complexity(self):
        """Log detailed model complexity information"""
        if self.model is None:
            return
            
        try:
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            # Estimate memory usage (rough calculation)
            param_memory_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
            
            # Get current profile info
            profile_info = PERFORMANCE_CONFIG.get_current_profile_info()
            profile_name = profile_info.get('name', PERFORMANCE_CONFIG.current_profile)
            
            logger.info("=" * 60)
            logger.info(f"🧠 TST MODEL COMPLEXITY ANALYSIS")
            logger.info("=" * 60)
            logger.info(f"Mode: {profile_name} ({PERFORMANCE_CONFIG.current_profile})")
            logger.info(f"Architecture:")
            logger.info(f"  - d_model: {self.config.get('d_model', 'N/A')} (embedding dimension)")
            logger.info(f"  - n_heads: {self.config.get('n_heads', 'N/A')} (attention heads)")
            logger.info(f"  - n_layers: {self.config.get('n_layers', 'N/A')} (transformer layers)")
            logger.info(f"  - d_ff: {self.config.get('d_ff', 'N/A')} (feedforward dimension)")
            logger.info(f"  - seq_len: {self.config.get('seq_len', 'N/A')} (sequence length)")
            logger.info(f"Parameters:")
            logger.info(f"  - Total: {total_params:,} parameters")
            logger.info(f"  - Trainable: {trainable_params:,} parameters")
            logger.info(f"  - Memory: ~{param_memory_mb:.1f} MB")
            logger.info(f"Expected Performance:")
            # Get performance info from profile
            performance = profile_info.get('performance', {})
            cpu_target = profile_info.get('cpu_target', 'Unknown')
            
            logger.info(f"  - CPU Usage: {cpu_target}")
            logger.info(f"  - Inference: {performance.get('expected_inference_ms', 'Unknown')} ms")
            logger.info(f"  - Memory: {performance.get('memory_mb', 'Unknown')} MB") 
            logger.info(f"  - Accuracy: {performance.get('accuracy', 'Unknown')}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error analyzing model complexity: {e}")
    
    def prepare_sequence_data(self, data_points: list) -> torch.Tensor:
        """
        Prepare sequence data for TST model input
        
        Args:
            data_points: List of packet count values
            
        Returns:
            torch.Tensor: Properly formatted tensor for TST
        """
        try:
            # Add to buffer
            self.feature_buffer.extend(data_points if isinstance(data_points, list) else [data_points])
            
            # Keep only the required sequence length
            if len(self.feature_buffer) > LOOKBACK:
                self.feature_buffer = self.feature_buffer[-LOOKBACK:]
            
            # Pad with zeros if not enough data
            if len(self.feature_buffer) < LOOKBACK:
                padding_needed = LOOKBACK - len(self.feature_buffer)
                padded_sequence = [0.0] * padding_needed + self.feature_buffer
            else:
                padded_sequence = self.feature_buffer[-LOOKBACK:]
            
            # Create tensor with shape [batch, features, sequence]
            # TST expects [batch_size, n_vars, seq_len]
            sequence_array = np.array(padded_sequence, dtype=np.float32)
            
            # Reshape to [1, 1, seq_len] - batch_size=1, n_vars=1, seq_len=LOOKBACK
            tensor = torch.tensor(sequence_array).unsqueeze(0).unsqueeze(0)
            
            logger.debug(f"Prepared TST input tensor shape: {tensor.shape}")
            return tensor
            
        except Exception as e:
            logger.error(f"Error preparing sequence data: {e}")
            # Return dummy tensor with correct shape
            dummy_data = np.random.randn(1, 1, LOOKBACK).astype(np.float32)
            return torch.tensor(dummy_data)
    
    def predict(self, tensor: torch.Tensor) -> Tuple[bool, float]:
        """
        Make prediction using TST model
        
        Args:
            tensor: Input tensor with shape [batch, features, sequence]
            
        Returns:
            Tuple[bool, float]: (is_attack, confidence)
        """
        if self.model is None:
            logger.warning("Model not loaded, returning random prediction")
            return np.random.random() > 0.95, 0.5
        
        try:
            with torch.no_grad():
                # Forward pass
                output = self.model(tensor)
                
                # Handle different output formats
                if output.dim() > 1:
                    # If multi-class output, take softmax
                    probs = torch.softmax(output, dim=1)
                    attack_prob = probs[0, 1].item() if probs.shape[1] > 1 else probs[0, 0].item()
                else:
                    # Single output, apply sigmoid
                    attack_prob = torch.sigmoid(output).item()
                
                # Determine attack
                is_attack = attack_prob > 0.5
                confidence = max(attack_prob, 1 - attack_prob)  # Confidence is max probability
                
                return is_attack, confidence
                
        except Exception as e:
            logger.error(f"Error in TST prediction: {e}")
            return False, 0.5

    def detect_ddos(self, features: np.ndarray) -> Tuple[bool, float]:
        """
        Wrapper for detect_ddos compatibility with research-grade heavy processing
        
        Args:
            features: Input features as numpy array [batch, sequence] or [sequence]
            
        Returns:
            Tuple[bool, float]: (is_attack, confidence)
        """
        try:
            # Ensure proper shape
            if len(features.shape) == 1:
                features = features.reshape(1, -1)  # [sequence] -> [1, sequence]
            
            # Use processing configuration based on current profile
            processing = self.processing_config
            enable_heavy = processing.get('enable_heavy_features', False)
            augmentation_count = processing.get('augmentation_count', 1)
            ensemble_passes = processing.get('ensemble_passes', 1)
            enable_statistics = processing.get('enable_statistics', False)
            
            if enable_heavy and augmentation_count > 1:
                # ULTRA HEAVY processing mode - maximum CPU consumption
                augmented_features = []
                
                # 1. Intensive data augmentation with complex transformations
                for i in range(augmentation_count):  # 5x data augmentation
                    # Multiple types of noise and transformations
                    noise = np.random.normal(0, 0.01, features.shape)
                    augmented = features + noise
                    
                    # Additional heavy transformations for ultra profile
                    if PERFORMANCE_CONFIG.current_profile == 'ultra':
                        # Apply heavy mathematical operations
                        augmented = np.convolve(augmented.flatten(), np.ones(5)/5, mode='same').reshape(augmented.shape)
                        augmented = np.fft.fft(augmented.flatten()).real.reshape(augmented.shape)
                        augmented = np.tanh(augmented * 2)  # Non-linear transformation
                    
                    augmented_features.append(augmented)
                
                # 2. Massive ensemble with heavy computation
                predictions = []
                confidences = []
                
                for aug_features in augmented_features:
                    tensor = torch.FloatTensor(aug_features).unsqueeze(1)  # [batch, 1, sequence]
                    
                    # Ultra-heavy: Multiple passes with different processing
                    for pass_num in range(ensemble_passes):  # 3 passes per augmented sample
                        # Add CPU-intensive preprocessing for each pass
                        if PERFORMANCE_CONFIG.current_profile == 'ultra':
                            # Heavy matrix operations
                            processed_tensor = tensor.clone()
                            for _ in range(10):  # 10 matrix operations per pass
                                processed_tensor = torch.mm(processed_tensor.view(-1, 1), processed_tensor.view(1, -1)).view(tensor.shape)
                                processed_tensor = torch.sigmoid(processed_tensor)
                                processed_tensor = processed_tensor / processed_tensor.norm()
                            tensor = processed_tensor
                        
                        pred, conf = self.predict(tensor)
                        predictions.append(pred)
                        confidences.append(conf)
                
                # Heavy ensemble voting
                attack_votes = sum(predictions)
                total_votes = len(predictions)
                avg_confidence = np.mean(confidences)
                
                # Ultra-intensive statistical analysis
                if enable_statistics:
                    # Basic statistics
                    feature_stats = {
                        'mean': np.mean(features),
                        'std': np.std(features),
                        'skew': self._calculate_skewness(features),
                        'kurtosis': self._calculate_kurtosis(features)
                    }
                    
                    # Ultra profile: Add MASSIVE CPU-intensive statistical operations
                    if PERFORMANCE_CONFIG.current_profile == 'ultra':
                        # Heavy statistical computations that consume CPU
                        for _ in range(50):  # 50 iterations of heavy stats
                            # Compute complex statistics
                            feature_stats['var'] = np.var(features)
                            feature_stats['percentiles'] = np.percentile(features, [10, 25, 50, 75, 90])
                            feature_stats['entropy'] = -np.sum((features + 1e-10) * np.log(features + 1e-10))
                            
                            # Heavy correlation matrix computation
                            if features.size > 1:
                                correlation_matrix = np.corrcoef(features.reshape(-1, 1), features.reshape(-1, 1))
                                eigenvalues = np.linalg.eigvals(correlation_matrix)
                                feature_stats['dominant_eigenvalue'] = np.max(eigenvalues)
                                
                            # Heavy Fourier analysis
                            fft_result = np.fft.fft(features.flatten())
                            feature_stats['spectral_energy'] = np.sum(np.abs(fft_result)**2)
                            
                            # Heavy convolution operations
                            kernel = np.random.random(10)
                            convolved = np.convolve(features.flatten(), kernel, mode='valid')
                            feature_stats['conv_energy'] = np.sum(convolved**2)
                    
                    logger.debug(f"Ultra-heavy processing: {total_votes} predictions, intensive stats computed")
                
                # Final decision based on ensemble
                is_attack = attack_votes > (total_votes * 0.5)
                final_confidence = avg_confidence
                
                return is_attack, final_confidence
                
            else:
                # Production mode: lightweight processing
                tensor = torch.FloatTensor(features).unsqueeze(1)  # [batch, 1, sequence]
                return self.predict(tensor)
            
        except Exception as e:
            logger.error(f"Error in detect_ddos: {e}")
            return False, 0.5
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Heavy statistical computation for research mode"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        n = len(data.flatten())
        skew = np.sum(((data - mean) / std) ** 3) / n
        return float(skew)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Heavy statistical computation for research mode"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        n = len(data.flatten())
        kurt = np.sum(((data - mean) / std) ** 4) / n - 3
        return float(kurt)

def ddos_detection_tst(detection_queue_out, mitigation_queue_in, output_storage_queue_in):
    """
    TST-specific detection function
    
    Args:
        detection_queue_out: Queue to receive feature vectors from preprocessing
        mitigation_queue_in: Queue to send detection results to mitigation  
        output_storage_queue_in: Queue to send results for storage
    """
    global network_attack, detection_halt
    
    logger.info("Starting TST detection process")
    
    # Initialize TST detector
    model_path = os.path.join(project_root, 'models', 'tst_model_fp32.pth')
    detector = TST_Detector(model_path)
    
    # Load model
    if not detector.load_model():
        logger.error("Failed to load TST model, continuing with test predictions")
    
    # Log current configuration at startup
    logger.info(f"🔥 ACTIVE PROFILE: {PERFORMANCE_CONFIG.current_profile}")
    processing_config = PERFORMANCE_CONFIG.get_processing_config()
    logger.info(f"🔥 HEAVY FEATURES ENABLED: {processing_config.get('enable_heavy_features', False)}")
    logger.info(f"🔥 AUGMENTATION COUNT: {processing_config.get('augmentation_count', 1)}")
    logger.info(f"🔥 ENSEMBLE PASSES: {processing_config.get('ensemble_passes', 1)}")
    
    prev_time = time.time()
    inference_times = []
    
    # Ultra-aggressive CPU consumption thread for ultra profile
    def ultra_cpu_burner():
        """Dedicated thread for maximum CPU consumption"""
        while PERFORMANCE_CONFIG.current_profile == 'ultra':
            try:
                # Massive CPU-intensive operations
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
            except Exception as e:
                logger.debug(f"CPU burner thread error: {e}")
                time.sleep(0.1)
    
    # Start ultra CPU burner for ultra profile
    if PERFORMANCE_CONFIG.current_profile == 'ultra':
        logger.info("🔥 STARTING ULTRA CPU BURNER THREAD")
        from threading import Thread
        cpu_burner_thread = Thread(target=ultra_cpu_burner, daemon=True)
        cpu_burner_thread.start()
    
    while True:
        try:
            # Additional background processing based on profile
            current_profile = PERFORMANCE_CONFIG.current_profile
            
            if current_profile == 'ultra':
                # Ultra-heavy background processing for maximum CPU usage
                for _ in range(200):  # 200 operations per loop iteration
                    dummy_matrix = np.random.random((75, 75))
                    _ = np.linalg.inv(dummy_matrix + np.eye(75) * 0.01)  # Matrix inversion
                    _ = np.fft.fft2(dummy_matrix)  # 2D FFT
                    _ = np.convolve(dummy_matrix.flatten(), np.ones(20), mode='valid')
                    _ = np.power(dummy_matrix, 2.5)  # Power operations
            elif current_profile == 'heavy':
                # Heavy background processing
                for _ in range(50):
                    dummy_matrix = np.random.random((50, 50))
                    _ = np.dot(dummy_matrix, dummy_matrix.T)
                    _ = np.fft.fft(dummy_matrix.flatten())
            elif current_profile == 'medium':
                # Medium background processing
                for _ in range(10):
                    dummy_data = np.random.random(100)
                    _ = np.convolve(dummy_data, np.ones(5), mode='valid')
            # Light profile: no background processing
            
            # Get feature vector from preprocessing
            data_point = detection_queue_out.get()
            current_time = time.time()
            
            # Check if we're in detection halt period
            if network_attack and (current_time - detection_halt > detection_halt_window):
                network_attack = False
                logger.info("Detection halt period ended, resuming normal operation")
            
            if not network_attack:
                start_time = time.time()
                
                # Force ultra-heavy processing regardless of config flags
                if current_profile == 'ultra':
                    # GUARANTEED ULTRA-HEAVY PROCESSING
                    logger.debug("🔥 Executing ULTRA-HEAVY processing")
                    
                    # Prepare feature data with massive augmentation
                    tensor = detector.prepare_sequence_data(data_point)
                    features = tensor.squeeze().numpy()
                    
                    # Force ultra-heavy processing by overriding config
                    original_config = detector.processing_config.copy()
                    detector.processing_config.update({
                        'enable_heavy_features': True,
                        'augmentation_count': 8,  # Force 8x augmentation
                        'ensemble_passes': 5,     # Force 5x ensemble
                        'enable_statistics': True
                    })
                    
                    # Ultra-intensive pre-processing
                    for _ in range(25):  # 25 rounds of heavy preprocessing
                        # Heavy mathematical transformations
                        features = np.convolve(features.flatten(), np.random.random(10), mode='same').reshape(features.shape)
                        features = np.fft.fft(features.flatten()).real.reshape(features.shape)
                        features = np.tanh(features * np.random.random())
                        features = features / (np.linalg.norm(features) + 1e-10)
                    
                    # Call heavy processing method
                    is_attack, confidence = detector.detect_ddos(features)
                    
                    # Restore original config
                    detector.processing_config = original_config
                    
                    # Additional ultra-heavy post-processing
                    for _ in range(20):  # 20 rounds of post-processing
                        dummy_result = np.random.random((100, 100))
                        _ = np.linalg.eigvals(dummy_result)
                        _ = np.fft.fft2(dummy_result)
                        _ = np.convolve(dummy_result.flatten(), np.ones(20), mode='valid')
                        
                elif current_profile in ['heavy', 'medium']:
                    # Use configured heavy processing
                    tensor = detector.prepare_sequence_data(data_point)
                    features = tensor.squeeze().numpy()
                    is_attack, confidence = detector.detect_ddos(features)
                else:
                    # Light processing
                    tensor = detector.prepare_sequence_data(data_point)
                    is_attack, confidence = detector.predict(tensor)
                
                end_time = time.time()
                inference_time_ms = (end_time - start_time) * 1000
                inference_times.append(inference_time_ms)
                
                # Log performance stats every 30 seconds
                if current_time - prev_time > 30:
                    if inference_times:
                        avg_inference = np.mean(inference_times)
                        max_inference = np.max(inference_times)
                        min_inference = np.min(inference_times)
                        
                        profile_info = PERFORMANCE_CONFIG.get_current_profile_info()
                        profile_name = profile_info.get('name', current_profile)
                        cpu_target = profile_info.get('cpu_target', 'Unknown')
                        
                        logger.info("=" * 60)
                        logger.info(f"🔥 {profile_name.upper()} Performance Report")
                        logger.info("=" * 60)
                        logger.info(f"Active Profile: {current_profile}")
                        logger.info(f"CPU Target: {cpu_target}")
                        logger.info(f"Predictions: {len(inference_times)}")
                        logger.info(f"Avg inference: {avg_inference:.2f}ms")
                        logger.info(f"Min inference: {min_inference:.2f}ms")
                        logger.info(f"Max inference: {max_inference:.2f}ms")
                        
                        if current_profile == 'ultra':
                            logger.info("� ULTRA MODE: Maximum CPU consumption active")
                        elif current_profile == 'heavy':
                            logger.info("🔬 HEAVY MODE: Research-grade processing")
                        elif current_profile == 'medium':
                            logger.info("⚡ MEDIUM MODE: Balanced performance")
                        else:
                            logger.info("💡 LIGHT MODE: Efficient processing")
                        logger.info("=" * 60)
                        inference_times = []
                    prev_time = current_time
                
                # Log detection result with profile info
                profile_marker = "🔥" if current_profile == 'ultra' else "🔬" if current_profile == 'heavy' else "⚡" if current_profile == 'medium' else "💡"
                logger.debug(f"TST {profile_marker}: {inference_time_ms:.2f}ms | {'ATTACK' if is_attack else 'NORMAL'} | Confidence: {confidence*100:.1f}%")
                
                # Send results to mitigation and storage
                mitigation_queue_in.put(is_attack)
                output_storage_queue_in.put({
                    'is_attack': is_attack,
                    'confidence': confidence,
                    'prediction_time': inference_time_ms,
                    'algorithm': 'tst',
                    'sequence_length': LOOKBACK,
                    'profile': current_profile
                })
                
            else:
                logger.debug("Detection halted due to ongoing mitigation")
                # Send non-attack signal during mitigation
                mitigation_queue_in.put(False)
                
        except Exception as e:
            logger.error(f"Error in TST detection: {e}")
            # Send safe default rather than crashing
            mitigation_queue_in.put(False)
            time.sleep(0.1)

def run_tst_pipeline():
    """
    Main function to run the complete TST pipeline
    """
    logger.info("--- Starting TST DDoS Defender ---")
    
    # Check for root privileges on Unix systems
    try:
        if os.geteuid() != 0 and os.name != 'nt':  # Skip root check on Windows
            print("Error: This script requires root privileges on Unix systems.")
            print("Please run with: sudo python3 tst_app/main.py")
            sys.exit(1)
    except AttributeError:
        # os.geteuid() doesn't exist on Windows
        pass
        
    check_venv()
    
    # Check PyTorch availability
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
    except ImportError:
        print("Error: PyTorch not installed. Please install it with: pip install torch")
        sys.exit(1)

    # Create queues for inter-process communication
    q_capture = Queue()
    q_detection = Queue()
    q_mitigation = Queue()
    q_storage_in = Queue()
    q_storage_out = Queue()

    # Create database tables
    insert_query = create_storage_tables(LOOKBACK, WINDOW_SIZE)
    logger.info(f"Database tables created with lookback={LOOKBACK}, window_size={WINDOW_SIZE}")

    # Start all threads
    threads = [
        Thread(target=capture, args=(q_capture,), daemon=True),
        Thread(target=ddos_preprocess, args=(q_capture, q_detection, q_storage_in, LOOKBACK, WINDOW_SIZE), daemon=True),
        Thread(target=ddos_detection_tst, args=(q_detection, q_mitigation, q_storage_out), daemon=True),
        Thread(target=ddos_mitigation, args=(q_mitigation,), daemon=True),
        Thread(target=store_input, args=(insert_query, q_storage_in), daemon=True),
        Thread(target=store_output, args=(q_storage_out,), daemon=True),
        Thread(target=update_config, args=(PORT,), daemon=True)
    ]
    
    # Thread names for logging
    thread_names = [
        "capture", "ddos_preprocess", "ddos_detection_tst", 
        "ddos_mitigation", "store_input", "store_output", "update_config"
    ]
    
    # Start all threads
    for i, thread in enumerate(threads):
        thread.start()
        thread_name = thread_names[i] if i < len(thread_names) else f"thread_{i+1}"
        logger.info(f"Started thread {i+1}/7: {thread_name}")
        
    logger.info("All TST pipeline threads started successfully")
    logger.info(f"Configuration server available at: http://localhost:{PORT}/status")
    logger.info("Press Ctrl+C to stop the application")
    
    try:
        # Keep main process alive and monitor thread health
        while True:
            time.sleep(10)
            
            # Check if any thread has died
            for i, thread in enumerate(threads):
                if not thread.is_alive():
                    thread_name = thread_names[i] if i < len(thread_names) else f"thread_{i+1}"
                    logger.error(f"Thread {i+1} ({thread_name}) has died!")
            
    except KeyboardInterrupt:
        logger.info("Shutdown signal received...")
        logger.info("TST pipeline shutdown complete")

if __name__ == "__main__":
    run_tst_pipeline()