# IntelliSight Setup Guide

## 📋 Complete Setup Instructions

### Prerequisites
- **Python**: 3.8 or higher
- **pip**: Latest version
- **Camera**: Optional (for live tracking)
- **Video files**: Optional (for analysis)

### Installation Steps

#### 1. Extract Project
```bash
# Extract the ZIP file to your desired location
unzip IntelliSight-AI-Customer-Analytics-Complete.zip
cd IntelliSight-AI-Customer-Analytics-Complete
```

#### 2. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# If you encounter PyTorch issues:
pip install torch==2.1.0 torchvision==0.16.0
pip install ultralytics --upgrade
```

#### 3. Verify Installation
```bash
# Run quick start to check system
python quick_start.py
# Choose option 5: System Check
```

### Quick Launch Options

#### Option A: Interactive Launcher (Recommended)
```bash
python quick_start.py
```
Then choose from the menu:
1. Customer Tracking
2. Business Dashboard
3. Demo Mode

#### Option B: Direct Launch

**Customer Tracking:**
```bash
# Use camera
python src/customer_tracking_engine.py --source 0

# Use video file
python src/customer_tracking_engine.py --source video.mp4

# Custom settings
python src/customer_tracking_engine.py --source video.mp4 --model yolov8n.pt --confidence 0.4
```

**Business Dashboard:**
```bash
streamlit run src/business_dashboard.py
# Opens at http://localhost:8501
```

### Configuration

#### Customize Zones
Edit `src/customer_tracking_engine.py`:
```python
self.zones = {
    'entrance': {'x1': 0, 'y1': 0, 'x2': 200, 'y2': 480, 
                'color': (255, 255, 0), 'name': '🚪 Entrance'},
    'browsing': {'x1': 200, 'y1': 0, 'x2': 440, 'y2': 300,
                'color': (255, 0, 255), 'name': '🛍️ Browsing'},
    # Add or modify zones here
}
```

#### Adjust AI Settings
```bash
# Faster processing (lower accuracy)
--model yolov8n.pt --confidence 0.3

# Balanced (recommended)
--model yolov8m.pt --confidence 0.35

# Higher accuracy (slower)
--model yolov8l.pt --confidence 0.4
```

### Troubleshooting

#### Model Loading Issues
```bash
# Fix PyTorch compatibility
pip uninstall torch torchvision -y
pip install torch==2.1.0 torchvision==0.16.0

# Reinstall YOLO
pip install ultralytics --upgrade
```

#### Camera Not Working
- Check camera permissions in system settings
- Try different camera index: `--source 1` or `--source 2`
- Test camera with: `python -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'Failed')"`

#### Dashboard Not Loading
```bash
# Upgrade Streamlit
pip install streamlit --upgrade

# Clear cache
streamlit cache clear

# Run with specific port
streamlit run src/business_dashboard.py --server.port 8502
```

#### Poor Detection Accuracy
- Adjust confidence threshold: `--confidence 0.25`
- Use better model: `--model yolov8m.pt` or `yolov8l.pt`
- Ensure good lighting in video/camera
- Position camera at optimal height (2-3 meters)

### Performance Optimization

#### For Real-time Processing
```python
# Adjust these parameters in the code:
process_every_n_frames = 2  # Skip frames for speed
max_detections = 50  # Limit detection count
```

#### For Accuracy
```python
process_every_n_frames = 1  # Process all frames
max_detections = 500  # Higher detection limit
```

### Data Management

#### Database Location
- **Path**: `data/analytics.db`
- **Format**: SQLite
- **Backup**: Copy the entire `data/` folder

#### Exports Location
- **Path**: `exports/`
- **Formats**: Excel (.xlsx), JSON (.json)
- **Auto-generated**: On quit ('q' key)

#### Screenshots
- **Path**: `output/`
- **Trigger**: Press 's' key during tracking
- **Format**: JPG with analytics overlay

### Advanced Configuration

#### Environment Variables
```bash
# Set custom paths
export INTELLISIGHT_DATA_PATH=/custom/path/data
export INTELLISIGHT_EXPORT_PATH=/custom/path/exports
```

#### Multiple Cameras
```python
# Edit quick_start.py to add multiple camera support
# Or run multiple instances with different sources
```

### Security & Privacy

#### Data Privacy
- No facial recognition or biometric storage
- Only coordinates and timestamps stored
- Anonymous customer IDs
- GDPR/CCPA compliant by design

#### Data Retention
```python
# Configure auto-cleanup (add to database setup)
DELETE FROM journeys WHERE created_at < datetime('now', '-30 days')
```

### Updates & Maintenance

#### Check for Updates
```bash
pip list --outdated
pip install --upgrade ultralytics opencv-python streamlit
```

#### Backup Your Data
```bash
# Regular backup
cp -r data/ backups/data_$(date +%Y%m%d)
cp -r exports/ backups/exports_$(date +%Y%m%d)
```

### Next Steps

1. **Test with sample video** - Verify all features work
2. **Customize zones** - Match your specific layout
3. **Adjust AI settings** - Optimize for your use case
4. **Run pilot test** - Collect initial data
5. **Analyze results** - Use dashboard for insights
6. **Scale deployment** - Expand to production

For additional help, see:
- `docs/USER_MANUAL.md` - Complete user guide
- `docs/API_DOCS.md` - API documentation
- `README.md` - Quick start guide
