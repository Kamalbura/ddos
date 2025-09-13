#!/usr/bin/env python3
"""
XGBoost Application for DDoS Detection
Standalone application using XGBoost model with LOOKBACK=5
"""

import os
import sys
from threading import Thread
from queue import Queue
import time
import xgboost as xgb
import numpy as np
import logging
from typing import Tuple

# Add project root to the Python path to find the 'common' module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from common.pipeline_components import (
    check_venv, check_packages, capture, ddos_preprocess, ddos_mitigation,
    create_storage_tables, store_input, store_output, update_config,
    network_attack, detection_halt, detection_halt_window
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - XGBoost - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- App Specific Settings ---
LOOKBACK = 5
WINDOW_SIZE = 0.60
PORT = 8000

def ddos_detection_xgb(detection_queue_out, mitigation_queue_in, output_storage_queue_in):
    """
    XGBoost-specific detection function
    
    Args:
        detection_queue_out: Queue to receive feature vectors from preprocessing
        mitigation_queue_in: Queue to send detection results to mitigation
        output_storage_queue_in: Queue to send results for storage
    """
    global network_attack, detection_halt
    
    logger.info("Starting XGBoost detection process")
    
    # Load XGBoost model
    model_path = os.path.join(project_root, 'models', 'xgboost_model.bin')
    model = None
    
    if os.path.exists(model_path):
        try:
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            logger.info(f"XGBoost model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}")
    else:
        logger.error(f"XGBoost model not found at {model_path}")
        logger.warning("Running in test mode with random predictions")
    
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
                
                if model is not None:
                    try:
                        # Prepare data for XGBoost
                        # XGBoost expects 2D array: (n_samples, n_features)
                        feature_array = np.array(data_point).reshape(1, -1)
                        
                        # Make prediction
                        prediction = model.predict(feature_array)[0]
                        
                        # Get prediction probability for confidence
                        try:
                            proba = model.predict_proba(feature_array)[0]
                            confidence = max(proba)  # Highest probability as confidence
                        except:
                            # If predict_proba fails, use distance from decision boundary
                            confidence = abs(prediction - 0.5) * 2
                        
                        is_attack = bool(prediction)
                        
                    except Exception as e:
                        logger.error(f"XGBoost prediction error: {e}")
                        is_attack = False
                        confidence = 0.5
                        
                else:
                    # Test mode - random predictions for development
                    is_attack = np.random.random() > 0.95  # 5% attack rate for testing
                    confidence = np.random.random() * 0.3 + 0.7  # High confidence
                
                end_time = time.time()
                inference_time_ms = (end_time - start_time) * 1000
                inference_times.append(inference_time_ms)
                
                # Log performance stats every 30 seconds
                if current_time - prev_time > 30:
                    if inference_times:
                        avg_inference = np.mean(inference_times)
                        logger.info(f"XGBoost Performance - Avg inference: {avg_inference:.2f}ms, Predictions: {len(inference_times)}")
                        inference_times = []
                    prev_time = current_time
                
                # Log detection result
                logger.debug(f"XGBoost: {inference_time_ms:.2f}ms | {'ATTACK' if is_attack else 'NORMAL'} | Confidence: {confidence*100:.1f}%")
                
                # Send results to mitigation and storage
                mitigation_queue_in.put(is_attack)
                output_storage_queue_in.put({
                    'is_attack': is_attack,
                    'confidence': confidence,
                    'prediction_time': inference_time_ms,
                    'algorithm': 'xgboost',
                    'feature_count': len(data_point)
                })
                
            else:
                logger.debug("Detection halted due to ongoing mitigation")
                # Send non-attack signal during mitigation
                mitigation_queue_in.put(False)
                
        except Exception as e:
            logger.error(f"Error in XGBoost detection: {e}")
            # Send safe default rather than crashing
            mitigation_queue_in.put(False)
            time.sleep(0.1)

def run_xgboost_pipeline():
    """
    Main function to run the complete XGBoost pipeline
    """
    logger.info("--- Starting XGBoost DDoS Defender ---")
    
    # Check for root privileges on Unix systems
    try:
        if os.geteuid() != 0 and os.name != 'nt':  # Skip root check on Windows
            print("Error: This script requires root privileges on Unix systems.")
            print("Please run with: sudo python3 xgboost_app/main.py")
            sys.exit(1)
    except AttributeError:
        # os.geteuid() doesn't exist on Windows
        pass
        
    check_venv()
    check_packages()

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
        Thread(target=ddos_detection_xgb, args=(q_detection, q_mitigation, q_storage_out), daemon=True),
        Thread(target=ddos_mitigation, args=(q_mitigation,), daemon=True),
        Thread(target=store_input, args=(insert_query, q_storage_in), daemon=True),
        Thread(target=store_output, args=(q_storage_out,), daemon=True),
        Thread(target=update_config, args=(PORT,), daemon=True)
    ]
    
    # Start all threads
    for i, thread in enumerate(threads):
        thread.start()
        logger.info(f"Started thread {i+1}/7: {thread._target.__name__}")
        
    logger.info("All XGBoost pipeline threads started successfully")
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
        logger.info("XGBoost pipeline shutdown complete")

if __name__ == "__main__":
    run_xgboost_pipeline()