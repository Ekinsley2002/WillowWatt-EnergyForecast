import requests
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import onnxruntime as ort
import numpy as np
from pathlib import Path
import argparse
import os
os.environ['QT_API'] = 'pyside6'
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvas
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PySide6.QtCore import QTimer, Qt
import time
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


'''
HOW TO SETUP FOR PRESENTATION:
Go to swagger
Run python to get latest datapoints (for historical)
Plug into swagger for latest (now for range) and datetime range (plug in what we got from python for range)
Copy and paste both bearer key and url for both current_url and hourly_urlauthorization bearer

Should be able to then get the API's, graph timeseries.
'''

def test_api():
    current_scalar_values = None
    
    # Ensure we have a token (fetch if not already stored)
    if not Config.LIVE_DATA_TOKEN:
        Config.get_live_data_token()
    
    try:
        current_url = f"{Config.BASE_URL}/time-series/{Config.ID}/latest?includeDataQuality=false"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {Config.LIVE_DATA_TOKEN}"
        }

        current_response = requests.get(current_url, headers=headers)
        response_data = current_response.json()
        
        if isinstance(response_data, list):
            current_values = response_data
        else:
            current_values = response_data.get("data", [])
        
        current_scalar_values = [v["scalarValue"] for v in current_values] if current_values and isinstance(current_values[0], dict) else current_values
        print(current_scalar_values)

    except Exception as e:
        if "Expecting value: line 1 column 1 (char 0)" in str(e):
            print("Please recheck your bearer token and url for the current values")
        else:
            print(f"Error: {e}")


    try:
        # Ensure we have a token (fetch if not already stored)
        if not Config.HISTORICAL_DATA_TOKEN:
            Config.get_historical_data_token()
        
        # Use dynamic dates for historical data
        start_date, end_date = Config.get_start_and_end_date_mst(hours_back=1)
        hourly_url = f"{Config.BASE_URL}/time-series/{Config.ID}?start={start_date}&end={end_date}&pageSize=4&includeDataQuality=false"

        hourly_headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {Config.HISTORICAL_DATA_TOKEN}"
        }
        hourly_response = requests.get(hourly_url, headers=hourly_headers)

        response_data = hourly_response.json()
        
        if isinstance(response_data, list):
            values = response_data
        else:
            values = response_data.get("data", [])
        
        hourly_scalar_values = [v["scalarValue"] for v in values] if values and isinstance(values[0], dict) else values
        print(hourly_scalar_values)

    except Exception as e:
        if "Expecting value: line 1 column 1 (char 0)" in str(e):
            print("Please recheck your bearer token and url for the historic values")
        else:
            print(f"Error: {e}")
    
    return current_scalar_values

def get_last_hour_values():
    # Build dynamic time range
    now = datetime.now(timezone.utc)
    one_hour_ago = now - timedelta(hours=1)
    
    start = one_hour_ago.strftime("%Y-%m-%dT%H:%M:%SZ")
    end = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    print(start, end)
    
    #response = requests.get(URL, headers=headers, params=params)
    #return response.json().get("data", [])

def use_model(current_scalar_values):
    model_path = Path(__file__).parent.parent / "Models/willow_energy_15min.onnx"
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    
    current_value = current_scalar_values[-1] if current_scalar_values else 0.0
    history_kw = [v / 1000.0 for v in current_scalar_values[-6:]]
    while len(history_kw) < 6:
        history_kw.insert(0, history_kw[0] if history_kw else current_value / 1000.0)
    
    now = datetime.now(timezone.utc)
    hour_normalized = now.hour / 23.0
    day_normalized = now.weekday() / 6.0
    
    features = np.array([[
        history_kw[-1],
        history_kw[-2] if len(history_kw) >= 2 else history_kw[-1],
        history_kw[-3] if len(history_kw) >= 3 else history_kw[-1],
        history_kw[-4] if len(history_kw) >= 4 else history_kw[-1],
        history_kw[-5] if len(history_kw) >= 5 else history_kw[-1],
        history_kw[-6] if len(history_kw) >= 6 else history_kw[-1],
        hour_normalized,
        day_normalized
    ]], dtype=np.float32)
    
    outputs = session.run(None, {input_name: features})
    predicted_kw = float(outputs[0].squeeze())
    predicted_watts = predicted_kw * 1000.0
    
    print(f"Model prediction for next 15 minutes: {predicted_watts:.2f} Watts")


def run_real_time_graph():
    # Ensure we have tokens
    if not Config.LIVE_DATA_TOKEN:
        Config.get_live_data_token()
    if not Config.HISTORICAL_DATA_TOKEN:
        Config.get_historical_data_token()
    
    def get_headers():
        """Get headers with current token (refresh if needed)"""
        if not Config.LIVE_DATA_TOKEN:
            Config.get_live_data_token()
        return {
            "accept": "application/json",
            "Authorization": f"Bearer {Config.LIVE_DATA_TOKEN}"
        }
    
    headers = get_headers()
    
    model_path = Path(__file__).parent.parent / "Models/willow_energy_15min.onnx"
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    
    now = datetime.now(timezone.utc)
    
    # First get current point, then get 3 historical points going back from it
    # This ensures we have 4 distinct points
    from urllib.parse import quote
    
    # Get current point first
    current_url = f"{Config.BASE_URL}/time-series/{Config.ID}/latest?includeDataQuality=false"
    try:
        current_response = requests.get(current_url, headers=headers)
        if current_response.status_code != 200:
            print(f"Error: Current API returned status code {current_response.status_code}")
            return
        current_data = current_response.json()
        if isinstance(current_data, list):
            current_values = current_data
        else:
            current_values = current_data.get("data", [])
        
        if not current_values:
            print("Error: No current data received")
            return
        
        if isinstance(current_values[0] if current_values else {}, dict):
            current_timestamp = datetime.fromisoformat(current_values[-1]["sourceTimestamp"].replace("Z", "+00:00"))
            current_value = current_values[-1]["scalarValue"]
        else:
            current_timestamp = now
            current_value = current_values[-1]
    except Exception as e:
        print(f"Error fetching current timestamp: {e}")
        current_timestamp = now
        current_value = None
    
    # Now get 3 historical points going back 15, 30, 45 minutes from current point
    end_time = current_timestamp
    start_time = current_timestamp - timedelta(minutes=45)  # Go back 45 minutes to get 3 points at 15-min intervals
    
    start_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    historical_url = f"{Config.BASE_URL}/time-series/{Config.ID}?start={quote(start_str)}&end={quote(end_str)}&pageSize=10&includeDataQuality=false"
    
    bottom_times = []
    bottom_values = []
    top_actual_times = []  # For initial 4 points on top graph (3 historical + 1 current)
    top_actual_values = []
    top_pred_times = []
    top_pred_values = []
    
    # Get 3 historical points
    try:
        historical_response = requests.get(historical_url, headers=headers)
        if historical_response.status_code != 200:
            print(f"Error: Historical API returned status code {historical_response.status_code}")
            return
        historical_data = historical_response.json()
        if isinstance(historical_data, list):
            historical_values = historical_data
        else:
            historical_values = historical_data.get("data", [])
        
        # Get historical points, excluding any that match the current timestamp
        # We want 3 distinct historical points before the current one
        sorted_historical = sorted(historical_values, key=lambda x: x.get("sourceTimestamp", "") if isinstance(x, dict) else "")
        
        # Filter out points that match or are after current_timestamp, then take the last 3
        historical_points = []
        for v in sorted_historical:
            if isinstance(v, dict):
                timestamp = datetime.fromisoformat(v["sourceTimestamp"].replace("Z", "+00:00"))
                # Only include if this point is BEFORE current_timestamp (not equal)
                if timestamp < current_timestamp:
                    historical_points.append(v)
            else:
                historical_points.append(v)
        
        # Get the last 3 historical points (most recent 3 before current)
        historical_points = historical_points[-3:] if len(historical_points) >= 3 else historical_points
        
        # Add 3 historical points (oldest first)
        for v in historical_points:
            if isinstance(v, dict):
                timestamp = datetime.fromisoformat(v["sourceTimestamp"].replace("Z", "+00:00"))
                value = v["scalarValue"]
            else:
                timestamp = (current_timestamp - timedelta(minutes=45 - 15 * len(bottom_times))).replace(tzinfo=timezone.utc)
                value = v
            
            bottom_times.append(timestamp)
            bottom_values.append(value)
            top_actual_times.append(timestamp)
            top_actual_values.append(value)
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return
    
    # Add the current point (4th yellow point) - we already fetched it above
    if current_value is not None:
        # Add current point (4th yellow point) - it should be distinct from historical points
        bottom_times.append(current_timestamp)
        bottom_values.append(current_value)
        top_actual_times.append(current_timestamp)
        top_actual_values.append(current_value)
        
        # Make first prediction based on current point
        history_kw = [v / 1000.0 for v in top_actual_values[-6:]]
        while len(history_kw) < 6:
            history_kw.insert(0, history_kw[0] if history_kw else current_value / 1000.0)
        
        next_pred_time = current_timestamp + timedelta(minutes=15)
        pred_hour = next_pred_time.hour / 23.0
        pred_day = next_pred_time.weekday() / 6.0
        
        features = np.array([[
            history_kw[-1],
            history_kw[-2] if len(history_kw) >= 2 else history_kw[-1],
            history_kw[-3] if len(history_kw) >= 3 else history_kw[-1],
            history_kw[-4] if len(history_kw) >= 4 else history_kw[-1],
            history_kw[-5] if len(history_kw) >= 5 else history_kw[-1],
            history_kw[-6] if len(history_kw) >= 6 else history_kw[-1],
            pred_hour,
            pred_day
        ]], dtype=np.float32)
        
        outputs = session.run(None, {input_name: features})
        predicted_kw = float(outputs[0].squeeze())
        predicted_watts = predicted_kw * 1000.0
        
        top_pred_times.append(next_pred_time)
        top_pred_values.append(predicted_watts)
    
    # Debug: Print what we have
    print(f"DEBUG: top_actual_times count: {len(top_actual_times)}, top_pred_times count: {len(top_pred_times)}")
    if top_actual_times:
        print(f"DEBUG: top_actual_times: {[t.strftime('%H:%M:%S') for t in top_actual_times]}")
    if top_pred_times:
        print(f"DEBUG: top_pred_times: {[t.strftime('%H:%M:%S') for t in top_pred_times]}")
    
    # No need for confirmed_predictions tracking - the most recent prediction is always dotted
    
    class RealTimeWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle('Real-Time Energy Forecast')
            self.setGeometry(100, 100, 1400, 900)
            
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            central_widget.setStyleSheet("background-color: #002454;")
            layout = QVBoxLayout(central_widget)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(13, 8))
            self.fig.patch.set_facecolor('#002454')
            self.fig.suptitle('Real-Time Energy Forecast', fontsize=16, fontweight='bold', color='white')
            
            self.mst = ZoneInfo('America/Phoenix')
            
            self.canvas = FigureCanvas(self.fig)
            layout.addWidget(self.canvas)
            
            self.status_label = QLabel()
            self.status_label.setStyleSheet("color: white; font-size: 10pt; padding: 8px; background-color: #002454; border-top: 1px solid rgba(255,255,255,0.2);")
            self.status_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.status_label)
            
            self.start_time = datetime.now(timezone.utc)
            self.last_update_time = self.start_time
            
            self.top_actual_times = top_actual_times  # Initial 4 historical points
            self.top_actual_values = top_actual_values
            self.top_pred_times = top_pred_times
            self.top_pred_values = top_pred_values
            self.bottom_times = bottom_times
            self.bottom_values = bottom_values
            
            # Store the ONNX session for predictions
            self.session = session
            self.input_name = input_name
            
            self.update_timer = QTimer()
            self.update_timer.timeout.connect(self.update_plots)
            self.update_timer.start(1000)
            
            # Timer for bottom graph: update every minute to show real-time data approaching prediction
            self.bottom_graph_timer = QTimer()
            self.bottom_graph_timer.timeout.connect(self.update_bottom_graph_only)
            self.bottom_graph_timer.start(60000)  # 1 minute = 60 * 1000 ms
            
            # Timer for top graph: check every 30 seconds if we're at a prediction time
            self.prediction_timer = QTimer()
            self.prediction_timer.timeout.connect(self.check_and_update_if_needed)
            self.prediction_timer.start(30000)  # 30 seconds = 30 * 1000 ms for precise timing
            
            self.update_plots()
        
        def to_mst(self, utc_time):
            if utc_time.tzinfo is None:
                utc_time = utc_time.replace(tzinfo=timezone.utc)
            return utc_time.astimezone(self.mst)
        
        def format_time_12h(self, mst_time):
            return mst_time.strftime("%I:%M %p").lstrip('0')
        
        def update_plots(self):
            self.ax1.clear()
            self.ax2.clear()
            
            current_time = datetime.now(timezone.utc)
            # 2-hour window: show 2 hours back from current time, plus 15 minutes ahead for predictions
            window_start = current_time - timedelta(hours=2)
            window_end = current_time + timedelta(minutes=15)  # Include future predictions
            
            # Top graph: Show initial 4 yellow points + white prediction points
            # Plot initial 4 yellow points (3 historical + 1 current) - show ALL, don't filter by window for initial points
            top_act_t_mst = []
            top_act_v = []
            if self.top_actual_times:
                # Don't filter initial points - show all 4 regardless of window
                top_act_t_mst = [self.to_mst(t) for t in self.top_actual_times]
                top_act_v = list(self.top_actual_values)
                if top_act_t_mst:
                    self.ax1.plot(top_act_t_mst, top_act_v, '-', color='#FAC01A', linewidth=2.5, zorder=2, label='Actual')
                    self.ax1.plot(top_act_t_mst, top_act_v, 'o', color='#FAC01A', markersize=10, zorder=3, markeredgecolor='white', markeredgewidth=1.5)
                    for t, v in zip(top_act_t_mst, top_act_v):
                        self.ax1.text(t, v, self.format_time_12h(t), fontsize=9, ha='center', va='bottom', color='white', weight='bold')
            
            # Keep reference for connecting to predictions
            top_actual_filtered = list(zip(self.top_actual_times, self.top_actual_values)) if self.top_actual_times else []
            
            # Plot predictions (white points)
            # The most recent prediction is always dotted, previous ones are solid
            if self.top_pred_times:
                top_pred_filtered = [(t, v) for t, v in zip(self.top_pred_times, self.top_pred_values) if window_start <= t <= window_end]
                
                if top_pred_filtered:
                    # Sort by time
                    top_pred_filtered = sorted(top_pred_filtered, key=lambda x: x[0])
                    
                    # Separate: all but the last are solid (confirmed), last is dotted (current prediction)
                    if len(top_pred_filtered) > 1:
                        confirmed_preds = top_pred_filtered[:-1]  # All but the last
                        unconfirmed_pred = top_pred_filtered[-1]  # Most recent
                    else:
                        confirmed_preds = []
                        unconfirmed_pred = top_pred_filtered[0]
                    
                    # Plot confirmed predictions as solid white lines
                    if confirmed_preds:
                        conf_t, conf_v = zip(*confirmed_preds)
                        conf_t_mst = [self.to_mst(t) for t in conf_t]
                        self.ax1.plot(conf_t_mst, conf_v, '-', color='white', linewidth=2.5, zorder=2)
                        self.ax1.plot(conf_t_mst, conf_v, 'o', color='white', markersize=10, zorder=3, markeredgecolor='#FAC01A', markeredgewidth=2, label='Predicted (Confirmed)')
                        for t, v in zip(conf_t_mst, conf_v):
                            self.ax1.text(t, v, self.format_time_12h(self.to_mst(t)), fontsize=9, ha='center', va='bottom', color='white', weight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#002454', alpha=0.7))
                    
                    # Plot current (most recent) prediction as dotted white line
                    unconf_t_mst = self.to_mst(unconfirmed_pred[0])
                    unconf_v = unconfirmed_pred[1]
                    self.ax1.plot([unconf_t_mst], [unconf_v], 'o', color='white', markersize=10, zorder=3, markeredgecolor='#FAC01A', markeredgewidth=2, label='Predicted (Forecast)')
                    self.ax1.text(unconf_t_mst, unconf_v, self.format_time_12h(unconf_t_mst), fontsize=9, ha='center', va='bottom', color='white', weight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#002454', alpha=0.7))
                    
                    # Connect last yellow point to first white prediction with dotted line
                    if self.top_actual_times and top_pred_filtered:
                        last_actual_t_mst = top_act_t_mst[-1] if top_act_t_mst else None
                        last_actual_v = top_act_v[-1] if top_act_v else None
                        if last_actual_t_mst is not None:
                            first_pred_t_mst = self.to_mst(top_pred_filtered[0][0])
                            first_pred_v = top_pred_filtered[0][1]
                            self.ax1.plot([last_actual_t_mst, first_pred_t_mst], [last_actual_v, first_pred_v], '--', color='white', linewidth=2.5, zorder=2)
                    
                    # Connect white prediction points: solid between confirmed, dotted to most recent
                    if len(top_pred_filtered) > 1:
                        pred_times_mst = [self.to_mst(t) for t, _ in top_pred_filtered]
                        pred_values = [v for _, v in top_pred_filtered]
                        # Solid line between all points except the last segment
                        if len(pred_times_mst) >= 2:
                            # Solid lines between confirmed points
                            if len(pred_times_mst) > 2:
                                self.ax1.plot(pred_times_mst[:-1], pred_values[:-1], '-', color='white', linewidth=2.5, zorder=2)
                            # Dotted line from second-to-last to last (current prediction)
                            self.ax1.plot(pred_times_mst[-2:], pred_values[-2:], '--', color='white', linewidth=2.5, zorder=2)
            
            bottom_filtered = [(t, v) for t, v in zip(self.bottom_times, self.bottom_values) if window_start <= t <= window_end]
            if bottom_filtered:
                bottom_t, bottom_v = zip(*bottom_filtered)
                bottom_t_mst = [self.to_mst(t) for t in bottom_t]
                self.ax2.plot(bottom_t_mst, bottom_v, '-', color='#FAC01A', linewidth=2.5, zorder=2)
                self.ax2.plot(bottom_t_mst, bottom_v, 'o', color='#FAC01A', markersize=10, label='Actual', zorder=3, markeredgecolor='white', markeredgewidth=1.5)
                for t, v in zip(bottom_t_mst, bottom_v):
                    self.ax2.text(t, v, self.format_time_12h(t), fontsize=9, ha='center', va='bottom', color='white', weight='bold')
            
            # Set x-axis limits to show 1-hour window, starting from the earliest data point
            if self.bottom_times:
                earliest_time = min(self.bottom_times)
                window_start_mst = self.to_mst(max(earliest_time, window_start))
            else:
                window_start_mst = self.to_mst(window_start)
            window_end_mst = self.to_mst(window_end)
            self.ax1.set_xlim(window_start_mst, window_end_mst)
            self.ax2.set_xlim(window_start_mst, window_end_mst)
            
            # Y-axis scaling for top graph based on both initial actual points and predictions
            if self.top_actual_values or self.top_pred_values:
                all_top_values = list(self.top_actual_values) + list(self.top_pred_values)
                if all_top_values:
                    y_min = min(all_top_values) * 0.98
                    y_max = max(all_top_values) * 1.02
                    self.ax1.set_ylim(y_min, y_max)
            
            if self.bottom_values:
                y_min = min(self.bottom_values) * 0.98
                y_max = max(self.bottom_values) * 1.02
                self.ax2.set_ylim(y_min, y_max)
            
            self.ax1.set_facecolor('#002454')
            self.ax2.set_facecolor('#002454')
            
            self.ax1.set_ylabel('Watts', fontsize=11, weight='bold', color='white')
            self.ax1.set_title('Model Predictions', fontsize=13, weight='bold', color='white', pad=10)
            legend1 = self.ax1.legend(loc='upper right', fontsize=10, framealpha=0.95, facecolor='white', edgecolor='#002454')
            for text in legend1.get_texts():
                text.set_color('#002454')
            self.ax1.grid(True, alpha=0.15, linestyle='--', linewidth=0.5, color='white')
            self.ax1.tick_params(axis='x', rotation=45, labelsize=9, colors='white')
            self.ax1.tick_params(axis='y', labelsize=9, colors='white')
            self.ax1.spines['bottom'].set_color('white')
            self.ax1.spines['top'].set_color('white')
            self.ax1.spines['right'].set_color('white')
            self.ax1.spines['left'].set_color('white')
            
            self.ax2.set_ylabel('Watts', fontsize=11, weight='bold', color='white')
            self.ax2.set_xlabel('Time', fontsize=11, weight='bold', color='white')
            self.ax2.set_title('Actual Data', fontsize=13, weight='bold', color='white', pad=10)
            legend2 = self.ax2.legend(loc='upper right', fontsize=10, framealpha=0.95, facecolor='white', edgecolor='#002454')
            for text in legend2.get_texts():
                text.set_color('#002454')
            self.ax2.grid(True, alpha=0.15, linestyle='--', linewidth=0.5, color='white')
            self.ax2.tick_params(axis='x', rotation=45, labelsize=9, colors='white')
            self.ax2.tick_params(axis='y', labelsize=9, colors='white')
            self.ax2.spines['bottom'].set_color('white')
            self.ax2.spines['top'].set_color('white')
            self.ax2.spines['right'].set_color('white')
            self.ax2.spines['left'].set_color('white')
            
            current_mst = self.to_mst(current_time)
            last_update_mst = self.to_mst(self.last_update_time) if hasattr(self, 'last_update_time') else current_mst
            date_str = last_update_mst.strftime("%B %d, %Y")
            time_str = self.format_time_12h(last_update_mst)
            current_time_str = self.format_time_12h(current_mst)
            status_text = f"Data Source: Last updated {date_str} at {time_str} MST | Current time: {current_time_str} MST"
            self.status_label.setText(status_text)
            
            self.fig.tight_layout(pad=3.0)
            self.canvas.draw()
        
        def update_bottom_graph_only(self):
            """Update bottom graph with latest actual data (runs every minute)"""
            try:
                # Refresh token if needed and get headers
                if not Config.LIVE_DATA_TOKEN:
                    Config.get_live_data_token()
                current_headers = {
                    "accept": "application/json",
                    "Authorization": f"Bearer {Config.LIVE_DATA_TOKEN}"
                }
                current_url = f"{Config.BASE_URL}/time-series/{Config.ID}/latest?includeDataQuality=false"
                current_response = requests.get(current_url, headers=current_headers)
                
                if current_response.status_code != 200:
                    return
                
                current_data = current_response.json()
                
                if isinstance(current_data, list):
                    current_values = current_data
                else:
                    current_values = current_data.get("data", [])
                
                if current_values:
                    if isinstance(current_values[0], dict):
                        new_timestamp = datetime.fromisoformat(current_values[-1]["sourceTimestamp"].replace("Z", "+00:00"))
                        new_value = current_values[-1]["scalarValue"]
                    else:
                        new_timestamp = datetime.now(timezone.utc)
                        new_value = current_values[-1]
                    
                    # Check if this is a new actual data point (for bottom graph)
                    should_add_actual_data = True
                    if self.bottom_times:
                        time_diff = abs((new_timestamp - self.bottom_times[-1]).total_seconds())
                        if time_diff < 30:  # Less than 30 seconds, duplicate - don't add
                            should_add_actual_data = False
                    
                    # Add new actual data to bottom graph if it's new
                    if should_add_actual_data:
                        print(f"DEBUG: Bottom graph update - new actual data at {new_timestamp}, value: {new_value:.2f} Watts")
                        self.bottom_times.append(new_timestamp)
                        self.bottom_values.append(new_value)
                        self.last_update_time = new_timestamp
            except Exception as e:
                print(f"Error updating bottom graph: {e}")
        
        def update_data(self):
            """Update both graphs: add actual data and generate prediction if at prediction time"""
            try:
                current_time = datetime.now(timezone.utc)
                
                # Check if we should generate a new prediction based on CURRENT time (not API timestamp)
                # This ensures we generate predictions exactly every 15 minutes ON THE DOT
                should_generate_prediction = False
                if self.top_pred_times:
                    last_pred_time = self.top_pred_times[-1]
                    time_until_pred = (last_pred_time - current_time).total_seconds()
                    # If we're at or past the prediction time (ON THE DOT - within 30 seconds for precision)
                    if time_until_pred <= 30:  # At or past prediction time (within 30 seconds for precision)
                        should_generate_prediction = True
                        print(f"DEBUG: At/past prediction time (diff: {time_until_pred:.1f}s), will generate new prediction ON THE DOT")
                    else:
                        print(f"DEBUG: Not yet at prediction time - {time_until_pred/60:.1f} minutes until prediction")
                else:
                    # No predictions yet, generate first one
                    should_generate_prediction = True
                
                # Refresh token if needed and get headers
                if not Config.LIVE_DATA_TOKEN:
                    Config.get_live_data_token()
                current_headers = {
                    "accept": "application/json",
                    "Authorization": f"Bearer {Config.LIVE_DATA_TOKEN}"
                }
                current_url = f"{Config.BASE_URL}/time-series/{Config.ID}/latest?includeDataQuality=false"
                current_response = requests.get(current_url, headers=current_headers)
                
                if current_response.status_code != 200:
                    return
                
                current_data = current_response.json()
                
                if isinstance(current_data, list):
                    current_values = current_data
                else:
                    current_values = current_data.get("data", [])
                
                if current_values:
                    if isinstance(current_values[0], dict):
                        new_timestamp = datetime.fromisoformat(current_values[-1]["sourceTimestamp"].replace("Z", "+00:00"))
                        new_value = current_values[-1]["scalarValue"]
                    else:
                        new_timestamp = datetime.now(timezone.utc)
                        new_value = current_values[-1]
                    
                    # Add actual data to bottom graph (if new)
                    should_add_actual_data = True
                    if self.bottom_times:
                        time_diff = abs((new_timestamp - self.bottom_times[-1]).total_seconds())
                        if time_diff < 30:  # Less than 30 seconds, duplicate - don't add
                            should_add_actual_data = False
                    
                    if should_add_actual_data:
                        print(f"DEBUG: Prediction update - new actual data at {new_timestamp}, value: {new_value:.2f} Watts")
                        self.bottom_times.append(new_timestamp)
                        self.bottom_values.append(new_value)
                        self.last_update_time = new_timestamp
                    
                    # Generate new prediction only if we're at/past the prediction time
                    if should_generate_prediction:
                        # Use the last 6 actual values from bottom graph (API data) for history
                        history_kw = [v / 1000.0 for v in self.bottom_values[-6:]]
                        while len(history_kw) < 6:
                            history_kw.insert(0, history_kw[0] if history_kw else new_value / 1000.0)
                        
                        # Calculate next prediction time: EXACTLY 15 minutes from CURRENT API call timestamp
                        # Use new_timestamp (the CURRENT API call) not a previous timestamp
                        next_pred_time = new_timestamp + timedelta(minutes=15)
                        
                        # Check if this prediction time already exists (with tolerance)
                        prediction_exists = False
                        if self.top_pred_times:
                            for existing_pred_time in self.top_pred_times:
                                time_diff = abs((next_pred_time - existing_pred_time).total_seconds())
                                if time_diff < 300:  # Within 5 minutes, consider it a duplicate
                                    prediction_exists = True
                                    print(f"DEBUG: Prediction for {next_pred_time} already exists (diff: {time_diff:.1f}s), skipping")
                                    break
                        
                        if not prediction_exists:
                            pred_hour = next_pred_time.hour / 23.0
                            pred_day = next_pred_time.weekday() / 6.0
                            
                            features = np.array([[
                                history_kw[-1],
                                history_kw[-2] if len(history_kw) >= 2 else history_kw[-1],
                                history_kw[-3] if len(history_kw) >= 3 else history_kw[-1],
                                history_kw[-4] if len(history_kw) >= 4 else history_kw[-1],
                                history_kw[-5] if len(history_kw) >= 5 else history_kw[-1],
                                history_kw[-6] if len(history_kw) >= 6 else history_kw[-1],
                                pred_hour,
                                pred_day
                            ]], dtype=np.float32)
                            
                            outputs = self.session.run(None, {self.input_name: features})
                            predicted_kw = float(outputs[0].squeeze())
                            predicted_watts = predicted_kw * 1000.0
                            
                            # Add new prediction to top graph (most recent is always dotted)
                            self.top_pred_times.append(next_pred_time)
                            self.top_pred_values.append(predicted_watts)
                            print(f"DEBUG: Added new prediction for {next_pred_time}, value: {predicted_watts:.2f} Watts")
            except Exception as e:
                print(f"Error updating data: {e}")
                import traceback
                traceback.print_exc()
        
        def check_and_update_if_needed(self):
            """Check if we're at or past a prediction time, and update if needed - runs every 30 seconds for precise timing"""
            try:
                current_time = datetime.now(timezone.utc)
                
                # Check if we're at or past the next prediction time
                should_update = False
                if self.top_pred_times:
                    next_pred_time = self.top_pred_times[-1]
                    time_until_pred = (next_pred_time - current_time).total_seconds()
                    
                    # If we're at or past the prediction time (ON THE DOT - within 30 seconds tolerance)
                    # This ensures we trigger exactly when the prediction time arrives
                    if time_until_pred <= 30:  # At or past prediction time (within 30 seconds for precision)
                        print(f"DEBUG: Prediction time reached (diff: {time_until_pred:.1f}s), triggering update ON THE DOT...")
                        should_update = True
                
                # Safety check: if more than 15.5 minutes since last prediction, force update
                if not should_update and hasattr(self, 'last_update_time') and self.last_update_time:
                    time_since_update = (current_time - self.last_update_time).total_seconds()
                    if time_since_update > 930:  # 15.5 minutes (930 seconds) - safety net
                        print(f"DEBUG: Safety check triggered - {time_since_update/60:.1f} minutes since last update, forcing update...")
                        should_update = True
                
                if should_update:
                    self.update_data()
            except Exception as e:
                print(f"Error in safety check: {e}")
                import traceback
                traceback.print_exc()
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = RealTimeWindow()
    window.show()
    sys.exit(app.exec())


def main():
    parser = argparse.ArgumentParser(description='Your script description')
    parser.add_argument('mode', choices=['run', 'time', 'test', 'graph'], help='Mode to run')

    args = parser.parse_args()
    if args.mode == 'run':
        current_scalar_values = test_api()
        if current_scalar_values:
            use_model(current_scalar_values)
        else:
            print("No current values found")
    elif args.mode == 'time':
        get_last_hour_values()
    elif args.mode == 'test':
        # Fetch tokens (will be stored as class variables)
        print("Fetching live data token...")
        Config.get_live_data_token()
        print("Fetching historical data token...")
        Config.get_historical_data_token()
        print("\nTesting API calls...")
        test_api()
    elif args.mode == 'graph':
        run_real_time_graph()
    else:
        print("Invalid mode")

if __name__ == '__main__':
    main()
