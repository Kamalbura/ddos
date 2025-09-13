#!/usr/bin/env python3
"""
Common Pipeline Components for DDoS Detection System
Contains all shared, model-agnostic logic for both XGBoost and TST applications
"""

import os
import sys
import sqlite3
import json
import subprocess
import time
import socket
import struct
from time import sleep
import scapy.all as scapy
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer
from urllib.parse import urlparse, parse_qs
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables that might be modified by the apps
network_attack = False
detection_halt = time.time()
detection_halt_window = 30
flag = 0

# Configuration flags
NO_MITIGATE = True  # Set to False for production mitigation
FAST_MITIGATE = False

# --- Environment Checks ---
def check_venv():
    """
    Modified to check for a specific venv path
    Note: This will be overridden to check for /home/pi/nenv on Raspberry Pi
    """
    # On Windows during development, we'll be more lenient
    if os.name == 'nt':  # Windows
        logger.info("Development environment detected (Windows)")
        return True
    
    # On Linux/Pi, check for specific nenv path
    nenv_path = "/home/pi/nenv"
    in_venv = sys.prefix == nenv_path
    
    if not in_venv:
        print("Error: The correct virtual environment is not activated!")
        print(f"Please activate it using: source {nenv_path}/bin/activate")
        sys.exit(1)
        
    logger.info(f"Virtual environment verified: {nenv_path}")

def check_packages():
    """Check for base requirements"""
    required = ['scapy', 'xgboost', 'numpy'] # Check for base requirements
    missing = []
    
    for package in required:
        try:
            __import__(package)
            logger.debug(f"Package {package} is available")
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing required packages: {missing}")
        print("Please install them using: pip install " + " ".join(missing))
        sys.exit(1)
    
    logger.info("All required packages are available")

# --- Network Interface Detection ---
def get_network_interface():
    """
    Detect the appropriate network interface for packet capture
    
    Returns:
        str: Network interface name
    """
    if os.name == 'nt':  # Windows
        # On Windows, use the first available interface or default to monitoring mode
        try:
            interfaces = scapy.get_if_list()
            # Try to find wireless or ethernet interface
            for iface in interfaces:
                if any(keyword in iface.lower() for keyword in ['wireless', 'wifi', 'ethernet', 'local']):
                    logger.info(f"Using network interface: {iface}")
                    return iface
            # Fallback to first interface
            if interfaces:
                logger.info(f"Using default interface: {interfaces[0]}")
                return interfaces[0]
        except:
            pass
        return None  # Let scapy auto-detect
    else:  # Linux/Pi
        # On Pi, default to wlan0 for wireless monitoring
        return "wlan0"

# --- Capture Component ---
def capture(capture_queue_in):
    """
    Network packet capture component
    Sniffs for specific UDP packets (MAVLink) and puts timestamps into queue
    
    Args:
        capture_queue_in: Queue to put captured packet timestamps
    """
    logger.info("Starting network packet capture")
    
    def packet_callback(packet):
        try:
            # Look for MAVLink packets (UDP with specific payload signature)
            if scapy.IP in packet and scapy.UDP in packet and scapy.Raw in packet:
                # MAVLink packets start with 0xfd (v2.0) or 0xfe (v1.0)
                raw_data = packet[scapy.Raw].load
                if len(raw_data) > 0 and raw_data[0:1] in [b'\xfd', b'\xfe']:
                    timestamp = time.time()
                    capture_queue_in.put(timestamp)
                    logger.debug(f"Captured MAVLink packet at {timestamp}")
        except Exception as e:
            logger.error(f"Error in packet_callback: {e}")
    
    try:
        interface = get_network_interface()
        logger.info(f"Starting packet capture on interface: {interface}")
        
        # Start packet capture
        scapy.sniff(
            prn=packet_callback, 
            store=0,  # Don't store packets in memory
            iface=interface,
            filter="udp"  # Only capture UDP packets for efficiency
        )
    except Exception as e:
        logger.error(f"Error in packet capture: {e}")
        # For testing, generate synthetic data
        logger.warning("Generating synthetic packet data for testing")
        while True:
            capture_queue_in.put(time.time())
            sleep(0.01)  # 100 packets per second

# --- Pre-Process Component ---
def ddos_preprocess(capture_queue_out, detection_queue_in, input_storage_queue_in, lookback, window_size):
    """
    Preprocessing component that converts packet timestamps to feature vectors
    
    Args:
        capture_queue_out: Queue to receive packet timestamps from
        detection_queue_in: Queue to send feature vectors to detection
        input_storage_queue_in: Queue to send data for storage
        lookback: Number of time windows to include in feature vector
        window_size: Duration of each time window in seconds
    """
    logger.info(f"Starting preprocessing with lookback={lookback}, window_size={window_size}")
    
    previous_processed_time = time.time()
    data = [0] * lookback
    
    # Initial fill of lookback windows
    for i in range(lookback):
        count = 0
        start_time = time.time()
        
        # Count packets in this time window
        while (time.time() - start_time) < window_size:
            try:
                capture_queue_out.get_nowait()
                count += 1
            except:
                sleep(0.001)  # Small sleep to prevent busy waiting
        
        data[i] = count
        previous_processed_time = time.time()
        logger.debug(f"Initial window {i}: {count} packets")
    
    # Send initial data
    detection_queue_in.put(data.copy())
    input_storage_queue_in.put({
        'timestamp': time.time(),
        'packet_counts': data.copy(),
        'lookback': lookback,
        'window_size': window_size
    })

    # Continuous processing with sliding window
    while True:
        try:
            # Shift the sliding window
            data = data[1:]  # Remove oldest element
            count = 0
            start_time = time.time()
            
            # Count packets in current window
            while (time.time() - start_time) < window_size:
                try:
                    capture_queue_out.get_nowait()
                    count += 1
                except:
                    sleep(0.001)
            
            # Add new count to the end
            data.append(count)
            
            # Send processed data
            detection_queue_in.put(data.copy())
            input_storage_queue_in.put({
                'timestamp': time.time(),
                'packet_counts': data.copy(),
                'lookback': lookback,
                'window_size': window_size
            })
            
            previous_processed_time = time.time()
            logger.debug(f"Processed window: {count} packets, total feature: {data}")
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            sleep(0.1)

# --- Mitigation Component ---
def ddos_mitigation(mitigation_queue_out):
    """
    Safe DDoS mitigation component
    Analyzes detection results and implements countermeasures
    
    Args:
        mitigation_queue_out: Queue to receive detection results from
    """
    global network_attack, detection_halt
    
    logger.info("Starting DDoS mitigation process")
    
    status_history = []
    context_length = 40  # Number of recent predictions to consider
    last_mitigation_time = 0
    mitigation_cooldown = 30  # Minimum seconds between mitigation actions
    
    while True:
        try:
            # Get detection result
            is_attack = mitigation_queue_out.get()
            status_history.append(bool(is_attack))
            
            # Maintain sliding window of recent results
            if len(status_history) > context_length:
                status_history = status_history[-context_length:]
            
            # Calculate attack consensus ratio
            attack_ratio = sum(status_history) / len(status_history) if status_history else 0
            current_time = time.time()
            
            # Check if we should trigger mitigation
            if is_attack and attack_ratio > 0.7:  # 70% of recent predictions indicate attack
                
                # Check cooldown period to prevent spam
                if current_time - last_mitigation_time < mitigation_cooldown:
                    logger.debug(f"Mitigation in cooldown, {mitigation_cooldown - (current_time - last_mitigation_time):.1f}s remaining")
                    continue
                
                logger.warning(f"DDoS Attack Detected! Threat level: {attack_ratio:.2f}")
                
                try:
                    if NO_MITIGATE:
                        logger.info("NO_MITIGATE flag set - logging only, no system changes")
                        
                        # Create alert file for manual review
                        alert_data = {
                            'timestamp': current_time,
                            'threat_score': attack_ratio,
                            'detection_window': context_length,
                            'recommended_actions': [
                                'Review network logs',
                                'Check source IPs', 
                                'Consider rate limiting',
                                'Monitor system resources'
                            ]
                        }
                        
                        # Write alert to safe location
                        os.makedirs('logs', exist_ok=True)
                        alert_file = f'logs/ddos_alert_{int(current_time)}.json'
                        with open(alert_file, 'w') as f:
                            json.dump(alert_data, f, indent=2)
                        
                        logger.info(f"DDoS alert logged to {alert_file}")
                        
                    else:
                        # Production mitigation (when NO_MITIGATE=False)
                        logger.info("Implementing DDoS countermeasures...")
                        
                        # Safe mitigation using subprocess instead of os.system
                        mitigation_commands = [
                            # Clear connection tracking table
                            ['conntrack', '-F'],
                            # Flush ARP table
                            ['ip', 'neigh', 'flush', 'all']
                        ]
                        
                        for cmd in mitigation_commands:
                            try:
                                result = subprocess.run(
                                    cmd, 
                                    check=True, 
                                    timeout=10,
                                    capture_output=True, 
                                    text=True
                                )
                                logger.info(f"Executed: {' '.join(cmd)}")
                            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
                                logger.warning(f"Failed to execute {' '.join(cmd)}: {e}")
                        
                        # Set network attack flag for detection pause
                        network_attack = True
                        detection_halt = current_time
                        logger.info(f"Detection paused for {detection_halt_window} seconds")
                    
                    last_mitigation_time = current_time
                    
                except Exception as e:
                    logger.error(f"Error in mitigation: {e}")
            
            else:
                if is_attack:
                    logger.debug(f"Attack detected but below threshold: {attack_ratio:.2f}")
                    
        except Exception as e:
            logger.error(f"Error in ddos_mitigation: {e}")
            sleep(0.1)

# --- Storage Components ---
def create_storage_tables(lookback, window_size):
    """
    Create SQLite database tables for storing input and output data
    
    Args:
        lookback: Number of lookback windows (for table naming)
        window_size: Window size in seconds (for table naming)
        
    Returns:
        str: SQL insert query string for input table
    """
    logger.info(f"Creating storage tables for lookback={lookback}, window_size={window_size}")
    
    # Create database directory if it doesn't exist
    db_path = os.path.abspath('db')
    os.makedirs(db_path, exist_ok=True)
    
    # Connect to databases
    input_db_path = os.path.join(db_path, 'input.db')
    output_db_path = os.path.join(db_path, 'output.db')
    
    input_conn = sqlite3.connect(input_db_path)
    output_conn = sqlite3.connect(output_db_path)
    
    # Create table name based on parameters
    table_name = f"L{lookback}_WS{str(window_size).replace('.', 'p')}"
    
    # Input table - stores feature vectors
    columns = ", ".join([f'c{i} REAL' for i in range(lookback)])
    input_conn.execute(f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            {columns},
            label REAL DEFAULT 0
        )
    ''')
    input_conn.commit()
    input_conn.close()
    
    # Output table - stores predictions
    output_conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            status REAL,
            algorithm TEXT,
            confidence REAL DEFAULT 0.0,
            processing_time REAL DEFAULT 0.0
        )
    ''')
    output_conn.commit()
    output_conn.close()
    
    # Return insert query for input table
    column_names = ", ".join([f'c{i}' for i in range(lookback)])
    placeholders = ", ".join(['?'] * (lookback + 2))  # +2 for timestamp and label
    insert_query = f'INSERT INTO {table_name} (timestamp, {column_names}, label) VALUES ({placeholders})'
    
    logger.info(f"Storage tables created successfully")
    return insert_query

def store_input(insert_query, input_storage_queue_out):
    """
    Store input feature data to SQLite database
    
    Args:
        insert_query: SQL insert query string
        input_storage_queue_out: Queue to receive input data from
    """
    logger.info("Starting input data storage process")
    
    db_path = os.path.abspath(os.path.join('db', 'input.db'))
    
    while True:
        try:
            # Get data from queue
            data_dict = input_storage_queue_out.get()
            
            # Extract packet counts and create record
            packet_counts = data_dict.get('packet_counts', [])
            timestamp = data_dict.get('timestamp', time.time())
            
            # Create data tuple: [timestamp, c0, c1, ..., cn, label]
            data_tuple = [timestamp] + packet_counts + [flag]
            
            # Store to database
            conn = sqlite3.connect(db_path)
            conn.execute(insert_query, tuple(data_tuple))
            conn.commit()
            conn.close()
            
            logger.debug(f"Stored input data: {len(packet_counts)} features")
            
        except Exception as e:
            logger.error(f"Error in store_input: {e}")
            sleep(0.1)

def store_output(output_storage_queue_out):
    """
    Store prediction results to SQLite database
    
    Args:
        output_storage_queue_out: Queue to receive prediction results from
    """
    logger.info("Starting output data storage process")
    
    db_path = os.path.abspath(os.path.join('db', 'output.db'))
    
    while True:
        try:
            # Get prediction result
            result_dict = output_storage_queue_out.get()
            
            # Extract data
            timestamp = time.time()
            status = 1 if result_dict.get('is_attack', False) else 0
            algorithm = result_dict.get('algorithm', 'unknown')
            confidence = result_dict.get('confidence', 0.0)
            processing_time = result_dict.get('prediction_time', 0.0)
            
            # Store to database
            conn = sqlite3.connect(db_path)
            conn.execute('''
                INSERT INTO predictions 
                (timestamp, status, algorithm, confidence, processing_time) 
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, status, algorithm, confidence, processing_time))
            conn.commit()
            conn.close()
            
            logger.debug(f"Stored prediction: {algorithm} -> {'ATTACK' if status else 'NORMAL'}")
            
        except Exception as e:
            logger.error(f"Error in store_output: {e}")
            sleep(0.1)

# --- Config Server ---
def update_config(port=8000):
    """
    Simple HTTP server for dynamic configuration updates
    
    Args:
        port: Port number to listen on
    """
    logger.info(f"Starting configuration server on port {port}")
    
    class ConfigHandler(SimpleHTTPRequestHandler):
        def do_GET(self):
            global flag
            
            if self.path.startswith('/update-parameters'):
                try:
                    # Parse query parameters
                    params = parse_qs(urlparse(self.path).query)
                    
                    # Update flag if provided
                    if "flag" in params:
                        flag = int(params["flag"][0])
                        logger.info(f"Updated flag to: {flag}")
                    
                    # Send success response
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    
                    response = {
                        'status': 'success',
                        'current_flag': flag,
                        'timestamp': time.time()
                    }
                    
                    self.wfile.write(json.dumps(response, indent=2).encode())
                    
                except Exception as e:
                    logger.error(f"Error processing config update: {e}")
                    self.send_error(500, f"Configuration update failed: {e}")
                    
            elif self.path == '/status':
                # Return system status
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                status = {
                    'flag': flag,
                    'network_attack': network_attack,
                    'detection_halt': detection_halt,
                    'timestamp': time.time()
                }
                
                self.wfile.write(json.dumps(status, indent=2).encode())
                
            else:
                # Default response for other paths
                self.send_response(404)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(b'DDoS Detection System - Config Server\nAvailable endpoints: /update-parameters, /status')
    
    try:
        with TCPServer(("", port), ConfigHandler) as httpd:
            logger.info(f"Configuration server running on http://localhost:{port}")
            httpd.serve_forever()
    except Exception as e:
        logger.error(f"Error in config server: {e}")