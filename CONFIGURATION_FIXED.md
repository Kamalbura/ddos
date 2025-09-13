# TST DDoS Defender - Configuration Fixed! 🎉

## Issues Fixed:
✅ Configuration persistence - profile changes now save properly  
✅ Old research_mode references updated to new profile system  
✅ Thread monitoring compatibility fixed for Python 3.11+  
✅ Ultra profile now loads correctly for 90% CPU usage  

## How to Use on Raspberry Pi:

### 1. Set Performance Profile:
```bash
# For maximum research (90% CPU like your original setup)
python scripts/config_tst.py --set ultra

# Other options:
# python scripts/config_tst.py --set light   # 5% CPU
# python scripts/config_tst.py --set medium  # 30% CPU  
# python scripts/config_tst.py --set heavy   # 60% CPU
```

### 2. Verify Current Profile:
```bash
python scripts/config_tst.py --info
```

### 3. Run TST Application:
```bash
sudo ~/nenv/bin/python tst_app/main.py
```

### 4. Available Profiles:

| Profile | CPU Usage | Description | Model Config |
|---------|-----------|-------------|--------------|
| **light** | 5-15% | Production ready | d_model=32, n_heads=4, n_layers=1 |
| **medium** | 25-35% | Balanced | d_model=64, n_heads=8, n_layers=2 |
| **heavy** | 55-70% | Research grade | d_model=128, n_heads=16, n_layers=3 |
| **ultra** | 80-95% | Maximum research | d_model=256, n_heads=32, n_layers=6 |

## What was Fixed:

### Configuration Persistence:
- Profile changes are now saved to `config/current_profile.txt`
- The app automatically loads the saved profile on startup
- No more "still showing Balanced" issue

### Processing Configuration:
- **Ultra Profile Features:**
  - 5x data augmentation 
  - 3x ensemble passes per prediction
  - Statistical analysis enabled
  - Heavy computational features enabled
  - Expected: 1000-2000ms inference time
  - Expected: 90% CPU usage on Raspberry Pi

### Code Compatibility:
- Fixed Python 3.11 thread attribute changes
- Updated all old `research_mode` references
- Proper error handling for configuration loading

## Expected Results on Raspberry Pi:

When you run with `ultra` profile:
```
🔧 Current Profile: Ultra Research
📝 Description: Maximum research configuration (~90% CPU)
🎯 CPU Target: 80-95%
🧠 Model Configuration:
   - d_model: 256 (ultra heavy)
   - n_heads: 32 (maximum attention)
   - n_layers: 6 (deep architecture)
   - d_ff: 1024 (large feedforward)
⚡ Processing Configuration:
   - Augmentation: 5x
   - Ensemble passes: 3x
   - Statistics: ✅ 
   - Heavy features: ✅
```

This should now achieve your target 90% CPU usage like your original research setup! 🚀

## Quick Commands:
```bash
# Set to maximum research mode
python scripts/config_tst.py --set ultra

# Check current configuration  
python scripts/config_tst.py --info

# Run TST with elevated privileges
sudo ~/nenv/bin/python tst_app/main.py
```