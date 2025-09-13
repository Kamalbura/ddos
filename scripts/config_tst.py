#!/usr/bin/env python3
"""
TST Performance Configuration CLI
Easy configuration management for TST performance profiles
"""
import os
import sys
import argparse

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from common.performance_config import PerformanceConfig

def main():
    parser = argparse.ArgumentParser(
        description='TST Performance Configuration Manager',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python config_tst.py --list                    # List all profiles
  python config_tst.py --set light               # Set to light profile (5% CPU)
  python config_tst.py --set medium              # Set to medium profile (30% CPU)  
  python config_tst.py --set heavy               # Set to heavy profile (60% CPU)
  python config_tst.py --set ultra               # Set to ultra profile (90% CPU)
  python config_tst.py --info                    # Show current profile details
  python config_tst.py --benchmark               # Run benchmark of all profiles

Performance Profiles:
  light     - ~5% CPU   - Production ready, fast inference
  medium    - ~30% CPU  - Balanced performance and accuracy  
  heavy     - ~60% CPU  - Research grade, high accuracy
  ultra     - ~90% CPU  - Maximum research quality
        """
    )
    
    parser.add_argument('--list', action='store_true', 
                       help='List available profiles')
    parser.add_argument('--set', type=str, metavar='PROFILE',
                       help='Set active profile (light/medium/heavy/ultra)')
    parser.add_argument('--info', action='store_true',
                       help='Show current profile details')
    parser.add_argument('--current', action='store_true',
                       help='Show current profile name only')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark of all profiles')
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    config = PerformanceConfig()
    
    if args.list:
        print("📋 Available TST Performance Profiles:")
        profiles = config.get_available_profiles()
        for profile_id, name in profiles.items():
            marker = " 🔥 (ACTIVE)" if profile_id == config.current_profile else ""
            profile_info = config.profiles[profile_id]
            cpu_target = profile_info.get('cpu_target', 'Unknown')
            description = profile_info.get('description', '')
            print(f"   {profile_id:8} - {name} ({cpu_target}){marker}")
            print(f"            {description}")
    
    elif args.set:
        profile = args.set.lower()
        if config.set_profile(profile):
            print(f"✅ Performance profile set to: {profile}")
            print(f"🎯 Expected CPU usage: {config.profiles[profile].get('cpu_target', 'Unknown')}")
            print(f"📝 Description: {config.profiles[profile].get('description', '')}")
            print(f"\n💡 To apply changes, restart the TST application:")
            print(f"   sudo ~/nenv/bin/python tst_app/main.py")
        else:
            available = list(config.get_available_profiles().keys())
            print(f"❌ Profile '{profile}' not found")
            print(f"Available profiles: {', '.join(available)}")
    
    elif args.info:
        config.print_profile_info()
        print(f"\n🚀 To run TST with current profile:")
        print(f"   sudo ~/nenv/bin/python tst_app/main.py")
        
        print(f"\n⚙️  To change profile:")
        print(f"   python scripts/config_tst.py --set <profile_name>")
    
    elif args.current:
        print(config.current_profile)
    
    elif args.benchmark:
        print("🔬 Running TST Performance Benchmark...")
        print("This will test all profiles and show expected performance")
        
        # Import benchmark functionality
        try:
            from benchmark_models import benchmark_model_configs
            benchmark_model_configs()
        except ImportError:
            print("❌ Benchmark module not available")
            print("Run: python scripts/benchmark_models.py")

if __name__ == "__main__":
    main()