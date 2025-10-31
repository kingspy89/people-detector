# IntelliSight API Documentation

## 🔌 API Reference

### Database Schema

#### Customer Journeys Table
```sql
CREATE TABLE journeys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT,           -- Unique customer ID
    entry_time REAL,            -- Unix timestamp
    exit_time REAL,             -- Unix timestamp
    duration REAL,              -- Seconds
    zones TEXT,                 -- JSON array
    behavior TEXT,              -- Behavior pattern
    conversion TEXT,            -- Conversion status
    created_at TIMESTAMP        -- Record creation time
)
```

### Python API

#### CustomerTracker Class

**Initialization:**
```python
from customer_tracking_engine import CustomerTracker

tracker = CustomerTracker(
    model_name='yolov8m.pt',  # YOLO model
    confidence=0.35            # Detection threshold
)
```

**Process Video:**
```python
tracker.process_video(source='video.mp4')
```

**Access Data:**
```python
# Active customers
active = tracker.active_customers

# Completed journeys
completed = tracker.completed_journeys

# Statistics
total = tracker.total_customers
peak = tracker.peak_occupancy
```

#### CustomerJourney Dataclass

**Structure:**
```python
@dataclass
class CustomerJourney:
    customer_id: str
    session_id: str
    entry_time: float
    exit_time: Optional[float]
    total_duration: Optional[float]
    zones_visited: List[str]
    path_coordinates: List[Tuple]
    dwell_times: Dict[str, float]
    behavior_pattern: str
    conversion_status: str
```

**Usage:**
```python
for journey in tracker.completed_journeys:
    print(f"Customer: {journey.customer_id}")
    print(f"Duration: {journey.total_duration/60:.1f} min")
    print(f"Zones: {journey.zones_visited}")
    print(f"Behavior: {journey.behavior_pattern}")
```

### Custom Integration

#### Real-time Data Access
```python
import sqlite3
import json

def get_recent_customers(limit=10):
    conn = sqlite3.connect('data/analytics.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT customer_id, duration, zones, behavior, conversion
        FROM journeys
        ORDER BY entry_time DESC
        LIMIT ?
    """, (limit,))

    results = cursor.fetchall()
    conn.close()

    return [{
        'id': r[0],
        'duration': r[1]/60,
        'zones': json.loads(r[2]),
        'behavior': r[3],
        'conversion': r[4]
    } for r in results]
```

#### Export Functions
```python
import pandas as pd

def export_analytics(output_file='report.xlsx'):
    conn = sqlite3.connect('data/analytics.db')
    df = pd.read_sql_query("SELECT * FROM journeys", conn)
    conn.close()

    df.to_excel(output_file, index=False)
    print(f"Exported: {output_file}")
```

#### Custom Analysis
```python
def calculate_conversion_rate():
    conn = sqlite3.connect('data/analytics.db')
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM journeys")
    total = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM journeys 
        WHERE conversion LIKE '%Purchased%'
    """)
    conversions = cursor.fetchone()[0]

    conn.close()

    return (conversions / total * 100) if total > 0 else 0
```

### Dashboard API

#### Custom Metrics
```python
import streamlit as st

def display_custom_metric(df):
    # Your custom calculation
    avg_duration = df['duration'].mean() / 60

    st.metric(
        label="Avg Visit Duration",
        value=f"{avg_duration:.1f} min",
        delta="+10% vs last week"
    )
```

#### Custom Charts
```python
import plotly.express as px

def create_custom_chart(df):
    fig = px.scatter(
        df, 
        x='duration', 
        y='zones_count',
        color='behavior',
        title="Duration vs Zone Visits"
    )
    return fig
```

### REST API (Optional)

#### Setup FastAPI
```python
from fastapi import FastAPI
import sqlite3

app = FastAPI()

@app.get("/api/customers")
def get_customers():
    conn = sqlite3.connect('data/analytics.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM journeys ORDER BY entry_time DESC LIMIT 100")
    data = cursor.fetchall()
    conn.close()
    return {"customers": data}

@app.get("/api/stats")
def get_stats():
    conn = sqlite3.connect('data/analytics.db')
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as total FROM journeys")
    total = cursor.fetchone()[0]
    conn.close()
    return {"total_customers": total}
```

#### Run API
```bash
pip install fastapi uvicorn
uvicorn api:app --reload
```

### Webhooks (Optional)

#### Customer Entry Event
```python
import requests

def on_customer_entry(customer_id):
    webhook_url = "https://your-server.com/webhook"
    data = {
        'event': 'customer_entry',
        'customer_id': customer_id,
        'timestamp': time.time()
    }
    requests.post(webhook_url, json=data)
```

#### Journey Complete Event
```python
def on_journey_complete(journey):
    webhook_url = "https://your-server.com/webhook"
    data = {
        'event': 'journey_complete',
        'customer_id': journey.customer_id,
        'duration': journey.total_duration,
        'behavior': journey.behavior_pattern,
        'conversion': journey.conversion_status
    }
    requests.post(webhook_url, json=data)
```

### Integration Examples

#### POS Integration
```python
class POSIntegration:
    def match_customer_to_transaction(self, transaction_time):
        # Find customer active at transaction time
        conn = sqlite3.connect('data/analytics.db')
        cursor = conn.cursor()
        cursor.execute("""
            SELECT customer_id FROM journeys
            WHERE entry_time <= ? AND exit_time >= ?
        """, (transaction_time, transaction_time))
        result = cursor.fetchone()
        conn.close()

        return result[0] if result else None
```

#### CRM Integration
```python
class CRMIntegration:
    def sync_customer_behavior(self):
        # Export behavior data to CRM
        customers = get_recent_customers()

        for customer in customers:
            # Sync to your CRM
            crm_api.update_customer_behavior(
                behavior=customer['behavior'],
                visit_duration=customer['duration'],
                zones_visited=customer['zones']
            )
```

### Advanced Customization

#### Custom Behavior Rules
```python
def custom_behavior_classification(journey):
    zones = journey.zones_visited
    duration = journey.total_duration
    dwell = journey.dwell_times

    # Your custom logic
    if 'vip_area' in zones and duration > 600:
        return "🌟 VIP Customer"
    elif 'clearance' in zones:
        return "💰 Bargain Hunter"
    else:
        return "👤 Regular Customer"
```

#### Custom Zone Types
```python
custom_zones = {
    'premium': {
        'x1': 0, 'y1': 0, 'x2': 200, 'y2': 300,
        'color': (255, 215, 0),  # Gold
        'name': '⭐ Premium Section',
        'type': 'high_value'
    },
    'clearance': {
        'x1': 400, 'y1': 0, 'x2': 640, 'y2': 300,
        'color': (255, 0, 0),  # Red
        'name': '🔥 Clearance',
        'type': 'discount'
    }
}
```

### Performance Optimization

#### Batch Processing
```python
def batch_process_videos(video_list):
    for video in video_list:
        tracker = CustomerTracker()
        tracker.process_video(video)
        tracker.save_report()
```

#### Parallel Processing
```python
from multiprocessing import Pool

def process_single_video(video_path):
    tracker = CustomerTracker()
    tracker.process_video(video_path)
    return tracker.total_customers

videos = ['video1.mp4', 'video2.mp4', 'video3.mp4']
with Pool(3) as p:
    results = p.map(process_single_video, videos)
```

### Error Handling

#### Graceful Degradation
```python
try:
    tracker = CustomerTracker()
    tracker.process_video(source)
except Exception as e:
    print(f"Error: {e}")
    # Fallback behavior
    print("Switching to fallback mode...")
```

### Logging & Monitoring

#### Custom Logging
```python
import logging

logging.basicConfig(
    filename='logs/tracking.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info("Customer tracking started")
```

For more examples and documentation, visit the GitHub repository or contact support.
