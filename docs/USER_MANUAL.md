# IntelliSight User Manual

## 📖 Complete User Guide

### Getting Started

#### First Launch
1. Open terminal/command prompt
2. Navigate to project folder
3. Run: `python quick_start.py`
4. Follow on-screen instructions

### Customer Tracking Features

#### Individual Tracking
Each person detected gets:
- **Unique ID**: Customer_1, Customer_2, etc.
- **Entry timestamp**: When they entered
- **Zone visits**: Which areas they visited
- **Path tracking**: Complete movement history
- **Dwell times**: Time spent in each zone
- **Exit timestamp**: When they left

#### Zone System
Default zones (customizable):
- **🚪 Entrance**: Entry/exit area
- **🛍️ Browsing**: Product/service area
- **💳 Counter**: Transaction/service point
- **🍽️ Seating**: Dining/waiting area

#### Behavior Classification
Automatic customer type detection:
- **🛒 Purchaser**: Visited counter, long engagement
- **🔍 Browser**: Long browsing, no purchase
- **🍽️ Diner**: Extended seating time
- **👀 Visitor**: Moderate engagement
- **🚶 Quick Pass**: Brief visit (<30s)

#### Conversion Status
Purchase intent analysis:
- **💳 Purchased**: Spent >30s at counter
- **🤔 Considered**: Brief counter visit
- **👀 Browsed**: Viewed products only
- **🚶 Passed**: Quick walk-through

### Using the Tracking System

#### Start Tracking
```bash
# Camera
python src/customer_tracking_engine.py --source 0

# Video
python src/customer_tracking_engine.py --source myvideo.mp4
```

#### During Tracking
- **Live view**: Shows real-time customer tracking
- **Zone overlays**: Colored areas with labels
- **Customer circles**: Pulsing indicators for each person
- **Path trails**: Gradient lines showing movement
- **Analytics overlay**: Real-time metrics at top

#### Keyboard Controls
- **'q'**: Quit and save comprehensive report
- **'s'**: Save screenshot with analytics overlay
- **'r'**: Reset analytics (start fresh)
- **ESC**: Emergency quit

#### Understanding the Display

**Analytics Overlay:**
```
🎯 IntelliSight Customer Analytics
═════════════════════════════════════════
👥 Live: 5        Current occupancy
📈 Peak: 8        Maximum occupancy today
🎯 Total: 47      Total customers processed
✅ Done: 12       Completed journeys
⚡ FPS: 15.2      Processing speed
```

**Customer Info:**
```
Customer_23 | 4.2 min | browsing
└─ ID       └─ Duration └─ Current zone
```

### Using the Dashboard

#### Launch Dashboard
```bash
streamlit run src/business_dashboard.py
```
Opens at: http://localhost:8501

#### Dashboard Features

**KPI Cards (Top Row):**
- Total Customers
- Average Visit Duration
- Conversion Rate
- Peak Occupancy

**Customer Journey Analytics:**
- Behavior pie chart
- Duration histogram
- Pattern distribution

**Zone Performance:**
- Visit frequency bar chart
- Zone performance cards
- Utilization metrics

**Recent Activity:**
- Latest customer journeys
- Detailed journey table
- Export capabilities

**AI Insights:**
- Automated recommendations
- Business optimization tips
- Performance indicators

#### Dashboard Controls

**Sidebar:**
- 🔄 Refresh Data: Manual update
- Time Period: Filter data range
- Export Options: Download reports

**Interactive Charts:**
- Hover: See detailed info
- Click: Filter/drill down
- Zoom: Pinch or scroll
- Pan: Click and drag

### Analytics & Reports

#### Automatic Reports
Generated on quit ('q' key):
- **Excel**: `exports/analytics_YYYYMMDD_HHMMSS.xlsx`
- **JSON**: Full data with metadata

#### Report Contents
**Excel Columns:**
- Customer_ID
- Entry_Time
- Duration_Min
- Zones_Visited
- Behavior_Pattern
- Conversion_Status
- Total_Zones

#### Export from Dashboard
1. Navigate to sidebar
2. Click export button
3. Choose format (CSV/Excel)
4. Save to desired location

### Business Applications

#### Retail Stores
**Use Case**: Optimize product placement
1. Track customer paths
2. Identify popular zones
3. Measure dwell times
4. Adjust layout based on data

**Insights:**
- Which products attract attention
- Optimal product positioning
- Queue management needs
- Staff allocation recommendations

#### Restaurants & Cafes
**Use Case**: Improve table turnover
1. Monitor seating duration
2. Track entrance to seating time
3. Measure service efficiency
4. Optimize table arrangement

**Insights:**
- Average dining duration
- Peak meal times
- Service bottlenecks
- Capacity optimization

#### Office Spaces
**Use Case**: Space utilization
1. Monitor meeting room usage
2. Track collaboration areas
3. Measure space efficiency
4. Plan resource allocation

**Insights:**
- Room occupancy patterns
- Popular collaboration zones
- Space optimization opportunities
- Resource planning data

### Advanced Features

#### Custom Zone Configuration
1. Edit `src/customer_tracking_engine.py`
2. Modify zone coordinates
3. Add new zones as needed
4. Adjust colors and names

```python
'my_zone': {
    'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100,
    'color': (255, 0, 0), 'name': '📍 My Zone'
}
```

#### AI Model Selection
**yolov8n.pt**: Fast, good accuracy
- Best for: Real-time processing
- Speed: ~30 FPS
- Accuracy: 90%

**yolov8m.pt**: Balanced (default)
- Best for: Most use cases
- Speed: ~15 FPS
- Accuracy: 95%

**yolov8l.pt**: Slow, best accuracy
- Best for: Offline analysis
- Speed: ~8 FPS
- Accuracy: 97%

#### Confidence Tuning
**Low (0.2-0.3)**: High sensitivity
- Catches more customers
- More false positives
- Use in clear environments

**Medium (0.35-0.45)**: Balanced (default)
- Good accuracy
- Few false positives
- Recommended for most cases

**High (0.5-0.7)**: High precision
- Fewer false positives
- May miss some customers
- Use in crowded/complex scenes

### Tips & Best Practices

#### Camera Placement
- **Height**: 2-3 meters optimal
- **Angle**: 30-45 degrees downward
- **Coverage**: Overlap zones slightly
- **Lighting**: Ensure adequate lighting

#### Optimal Performance
- Process during off-peak hours for analysis
- Use SSD for better I/O performance
- Close unnecessary applications
- Use GPU if available

#### Data Quality
- Clean camera lens regularly
- Ensure stable mounting
- Avoid backlighting
- Maintain consistent lighting

#### Privacy Compliance
- Post visible notices
- Inform customers of tracking
- Follow local regulations
- Anonymize exported data

### Troubleshooting

#### Low FPS
- Use smaller YOLO model (yolov8n.pt)
- Reduce confidence threshold
- Process fewer frames (skip frames)
- Close background applications

#### Missed Detections
- Increase confidence threshold
- Use larger model (yolov8l.pt)
- Improve lighting
- Adjust camera angle

#### False Detections
- Increase confidence (0.5+)
- Use better model (yolov8m/l)
- Clean environment of clutter
- Adjust zone boundaries

#### Export Issues
- Check disk space
- Verify write permissions
- Ensure exports/ folder exists
- Try different export format

### FAQ

**Q: Can I use multiple cameras?**
A: Run multiple instances with different sources

**Q: Does it work at night?**
A: Yes, with adequate lighting (IR cameras recommended)

**Q: Can I track specific individuals?**
A: Tracks customers anonymously by session

**Q: How long is data stored?**
A: Indefinitely in database until manually cleared

**Q: Is facial recognition used?**
A: No, only coordinates and movement patterns

**Q: Can I integrate with POS?**
A: Yes, custom integration possible (see API docs)

**Q: What about privacy laws?**
A: Compliant by design (no biometric data stored)

**Q: Can I run on Raspberry Pi?**
A: Yes, use yolov8n model for better performance

### Support

For additional help:
- Check `docs/SETUP_GUIDE.md` for installation issues
- Review `docs/API_DOCS.md` for integration
- See `README.md` for quick reference

### Updates

Check for updates regularly:
```bash
pip install --upgrade ultralytics opencv-python streamlit
```
