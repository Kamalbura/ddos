#!/usr/bin/env python3
"""
Performance Configuration Manager
Handles loading and switching between different performance profiles
"""
import json
import os
from typing import Dict, Any, Optional

class PerformanceConfig:
    """Manages performance profiles and configuration loading"""
    
    def __init__(self, config_dir: Optional[str] = None):
        if config_dir is None:
            # Default to config directory relative to this file
            self.config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
        else:
            self.config_dir = config_dir
            
        self.config_file = os.path.join(self.config_dir, 'performance_profiles.json')
        self.profiles = {}
        self.current_profile = None
        self.load_profiles()
    
    def load_profiles(self):
        """Load performance profiles from JSON configuration"""
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
                
            self.profiles = config_data.get('profiles', {})
            default_profile = config_data.get('default_profile', 'medium')
            
            if default_profile in self.profiles:
                self.current_profile = default_profile
            else:
                # Fallback to first available profile
                self.current_profile = list(self.profiles.keys())[0] if self.profiles else None
                
        except FileNotFoundError:
            print(f"❌ Configuration file not found: {self.config_file}")
            self._create_default_config()
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing configuration file: {e}")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create a basic default configuration"""
        self.profiles = {
            'medium': {
                'name': 'Default Medium',
                'model_config': {'d_model': 64, 'n_heads': 8, 'n_layers': 2, 'd_ff': 256},
                'processing': {'augmentation_count': 1, 'ensemble_passes': 1}
            }
        }
        self.current_profile = 'medium'
    
    def get_available_profiles(self) -> Dict[str, str]:
        """Get list of available profiles with descriptions"""
        return {
            profile_id: profile_data.get('name', profile_id) 
            for profile_id, profile_data in self.profiles.items()
        }
    
    def set_profile(self, profile_id: str) -> bool:
        """Set the active performance profile"""
        if profile_id in self.profiles:
            self.current_profile = profile_id
            return True
        return False
    
    def get_current_profile_info(self) -> Dict[str, Any]:
        """Get complete information about current profile"""
        if self.current_profile and self.current_profile in self.profiles:
            return self.profiles[self.current_profile]
        return {}
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for current profile"""
        profile_info = self.get_current_profile_info()
        base_config = {
            'c_in': 1,
            'c_out': 2,
            'seq_len': 400,
            'max_seq_len': 512,
            'norm': 'BatchNorm',
            'attn_dropout': 0.1,
            'res_attention': True,
            'pre_norm': False,
            'pe': 'zeros',
            'learn_pe': True,
            'dropout': 0.1
        }
        
        # Override with profile-specific settings
        model_config = profile_info.get('model_config', {})
        base_config.update(model_config)
        
        return base_config
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration for current profile"""
        profile_info = self.get_current_profile_info()
        return profile_info.get('processing', {
            'augmentation_count': 1,
            'ensemble_passes': 1,
            'enable_statistics': False,
            'enable_heavy_features': False
        })
    
    def print_profile_info(self):
        """Print detailed information about current profile"""
        if not self.current_profile:
            print("❌ No profile selected")
            return
            
        profile = self.get_current_profile_info()
        print(f"🔧 Current Profile: {profile.get('name', self.current_profile)}")
        print(f"📝 Description: {profile.get('description', 'No description')}")
        print(f"🎯 CPU Target: {profile.get('cpu_target', 'Unknown')}")
        
        model_config = self.get_model_config()
        print(f"\n🧠 Model Configuration:")
        print(f"   - d_model: {model_config.get('d_model')}")
        print(f"   - n_heads: {model_config.get('n_heads')}")  
        print(f"   - n_layers: {model_config.get('n_layers')}")
        print(f"   - d_ff: {model_config.get('d_ff')}")
        
        processing = self.get_processing_config()
        print(f"\n⚡ Processing Configuration:")
        print(f"   - Augmentation: {processing.get('augmentation_count')}x")
        print(f"   - Ensemble passes: {processing.get('ensemble_passes')}x")
        print(f"   - Statistics: {'✅' if processing.get('enable_statistics') else '❌'}")
        print(f"   - Heavy features: {'✅' if processing.get('enable_heavy_features') else '❌'}")
        
        performance = profile.get('performance', {})
        if performance:
            print(f"\n📊 Expected Performance:")
            print(f"   - Inference: {performance.get('expected_inference_ms', 'Unknown')} ms")
            print(f"   - Memory: {performance.get('memory_mb', 'Unknown')} MB")
            print(f"   - Accuracy: {performance.get('accuracy', 'Unknown')}")

def main():
    """CLI interface for configuration management"""
    import argparse
    
    parser = argparse.ArgumentParser(description='TST Performance Configuration Manager')
    parser.add_argument('--list', action='store_true', help='List available profiles')
    parser.add_argument('--set', type=str, help='Set active profile')
    parser.add_argument('--info', action='store_true', help='Show current profile info')
    parser.add_argument('--current', action='store_true', help='Show current profile name')
    
    args = parser.parse_args()
    
    config = PerformanceConfig()
    
    if args.list:
        print("📋 Available Performance Profiles:")
        profiles = config.get_available_profiles()
        for profile_id, name in profiles.items():
            marker = " 🔥" if profile_id == config.current_profile else ""
            print(f"   - {profile_id}: {name}{marker}")
    
    elif args.set:
        if config.set_profile(args.set):
            print(f"✅ Profile set to: {args.set}")
            # Save the change (would need to implement profile persistence)
        else:
            print(f"❌ Profile '{args.set}' not found")
            print("Available profiles:", list(config.get_available_profiles().keys()))
    
    elif args.info:
        config.print_profile_info()
    
    elif args.current:
        print(f"Current profile: {config.current_profile}")
    
    else:
        print("Use --help for available commands")
        config.print_profile_info()

if __name__ == "__main__":
    main()