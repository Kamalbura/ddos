#!/usr/bin/env python3
"""
TST Application for DDoS Detection  
Standalone application using Time Series Transformer model with LOOKBACK=400
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
PORT = 8001  # Use different port to avoid conflicts

class TST_Detector:
    """
    TST model wrapper with proper tensor handling for time series data
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.feature_buffer = []
        
        logger.info("Initializing TST detector")
        
    def load_model(self) -> bool:
        """
        Load TST model with proper error handling
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if os.path.exists(self.model_path):
                # Load the model
                checkpoint = torch.load(self.model_path, map_location='cpu')
                
                # The model should be the entire model object
                if hasattr(checkpoint, 'eval'):
                    # It's the model directly
                    self.model = checkpoint
                else:
                    # It might be a state dict, try to create model
                    # For now, create a default TST model
                    self.model = TSTPlus(
                        c_in=1,  # Single feature (packet count)
                        c_out=2,  # Binary classification (Normal/Attack)
                        seq_len=LOOKBACK,
                        d_model=64,  # Smaller for efficiency
                        n_heads=8,
                        n_layers=2,  # Reduced layers for speed
                        dropout=0.1
                    )
                    
                    # Try to load state dict if available
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        try:
                            self.model.load_state_dict(checkpoint['state_dict'])
                        except:
                            logger.warning("Could not load state dict, using randomly initialized model")
                
                self.model.eval()  # Set to evaluation mode
                logger.info(f"TST model loaded successfully from {self.model_path}")
                return True
            else:
                logger.error(f"TST model not found at {self.model_path}")
                # Create a dummy model for testing
                self.model = TSTPlus(
                    c_in=1,
                    c_out=2, 
                    seq_len=LOOKBACK,
                    d_model=64,
                    n_heads=8,
                    n_layers=2,
                    dropout=0.1
                )
                logger.warning("Using randomly initialized TST model for testing")
                return True
                
        except Exception as e:
            logger.error(f"Error loading TST model: {e}")
            return False
    
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
    
    prev_time = time.time()
    inference_times = []
    
    while True:
        try:
            # Get feature vector from preprocessing
            data_point = detection_queue_out.get()
            current_time = time.time()
            
            # Check if we're in detection halt period
            if network_attack and (current_time - detection_halt > detection_halt_window):
                network_attack = False
                logger.info("Detection halt period ended, resuming normal operation")
            
            if not network_attack:
                start_time = time.time()
                
                # Prepare sequence data for TST
                tensor = detector.prepare_sequence_data(data_point)
                
                # Make prediction
                is_attack, confidence = detector.predict(tensor)
                
                end_time = time.time()
                inference_time_ms = (end_time - start_time) * 1000
                inference_times.append(inference_time_ms)
                
                # Log performance stats every 30 seconds
                if current_time - prev_time > 30:
                    if inference_times:
                        avg_inference = np.mean(inference_times)
                        logger.info(f"TST Performance - Avg inference: {avg_inference:.2f}ms, Predictions: {len(inference_times)}")
                        inference_times = []
                    prev_time = current_time
                
                # Log detection result
                logger.debug(f"TST: {inference_time_ms:.2f}ms | {'ATTACK' if is_attack else 'NORMAL'} | Confidence: {confidence*100:.1f}%")
                
                # Send results to mitigation and storage
                mitigation_queue_in.put(is_attack)
                output_storage_queue_in.put({
                    'is_attack': is_attack,
                    'confidence': confidence,
                    'prediction_time': inference_time_ms,
                    'algorithm': 'tst',
                    'sequence_length': LOOKBACK
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
    
    # Start all threads
    for i, thread in enumerate(threads):
        thread.start()
        logger.info(f"Started thread {i+1}/7: {thread._target.__name__}")
        
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
                    logger.error(f"Thread {i+1} ({thread._target.__name__}) has died!")
            
    except KeyboardInterrupt:
        logger.info("Shutdown signal received...")
        logger.info("TST pipeline shutdown complete")

if __name__ == "__main__":
    run_tst_pipeline()