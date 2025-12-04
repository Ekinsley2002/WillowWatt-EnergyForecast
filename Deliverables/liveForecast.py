import requests
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from urllib.parse import quote
import onnxruntime as ort
import numpy as np
from pathlib import Path
import argparse
import os
import sys

os.environ['QT_API'] = 'pyside6'
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvas
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PySide6.QtCore import QTimer, Qt

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


def test_api():
    current_scalar_values = None
    
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
        if not Config.HISTORICAL_DATA_TOKEN:
            Config.get_historical_data_token()
        
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
    now = datetime.now(timezone.utc)
    one_hour_ago = now - timedelta(hours=1)
    
    start = one_hour_ago.strftime("%Y-%m-%dT%H:%M:%SZ")
    end = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    print(start, end)


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
    if not Config.LIVE_DATA_TOKEN:
        Config.get_live_data_token()
    if not Config.HISTORICAL_DATA_TOKEN:
        Config.get_historical_data_token()
    
    def get_headers():
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
    
    end_time = current_timestamp
    start_time = current_timestamp - timedelta(minutes=45)
    
    start_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    historical_url = f"{Config.BASE_URL}/time-series/{Config.ID}?start={quote(start_str)}&end={quote(end_str)}&pageSize=10&includeDataQuality=false"
    
    bottom_times = []
    bottom_values = []
    top_actual_times = []
    top_actual_values = []
    top_pred_times = []
    top_pred_values = []
    
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
        
        sorted_historical = sorted(historical_values, key=lambda x: x.get("sourceTimestamp", "") if isinstance(x, dict) else "")
        
        historical_points = []
        for v in sorted_historical:
            if isinstance(v, dict):
                timestamp = datetime.fromisoformat(v["sourceTimestamp"].replace("Z", "+00:00"))
                if timestamp < current_timestamp:
                    historical_points.append(v)
            else:
                historical_points.append(v)
        
        historical_points = historical_points[-3:] if len(historical_points) >= 3 else historical_points
        
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
    
    if current_value is not None:
        bottom_times.append(current_timestamp)
        bottom_values.append(current_value)
        top_actual_times.append(current_timestamp)
        top_actual_values.append(current_value)
        
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
            
            self.top_actual_times = top_actual_times
            self.top_actual_values = top_actual_values
            self.top_pred_times = top_pred_times
            self.top_pred_values = top_pred_values
            self.bottom_times = bottom_times
            self.bottom_values = bottom_values
            
            self.session = session
            self.input_name = input_name
            
            self.update_timer = QTimer()
            self.update_timer.timeout.connect(self.update_plots)
            self.update_timer.start(1000)
            
            self.bottom_graph_timer = QTimer()
            self.bottom_graph_timer.timeout.connect(self.update_bottom_graph_only)
            self.bottom_graph_timer.start(60000)
            
            self.prediction_timer = QTimer()
            self.prediction_timer.timeout.connect(self.check_and_update_if_needed)
            self.prediction_timer.start(30000)
            
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
            window_start = current_time - timedelta(hours=2)
            window_end = current_time + timedelta(minutes=15)
            
            top_act_t_mst = []
            top_act_v = []
            if self.top_actual_times:
                top_act_t_mst = [self.to_mst(t) for t in self.top_actual_times]
                top_act_v = list(self.top_actual_values)
                if top_act_t_mst:
                    self.ax1.plot(top_act_t_mst, top_act_v, '-', color='#FAC01A', linewidth=2.5, zorder=2, label='Actual')
                    self.ax1.plot(top_act_t_mst, top_act_v, 'o', color='#FAC01A', markersize=10, zorder=3, markeredgecolor='white', markeredgewidth=1.5)
                    for t, v in zip(top_act_t_mst, top_act_v):
                        self.ax1.text(t, v, self.format_time_12h(t), fontsize=9, ha='center', va='bottom', color='white', weight='bold')
            
            if self.top_pred_times:
                top_pred_filtered = [(t, v) for t, v in zip(self.top_pred_times, self.top_pred_values) if window_start <= t <= window_end]
                
                if top_pred_filtered:
                    top_pred_filtered = sorted(top_pred_filtered, key=lambda x: x[0])
                    
                    if len(top_pred_filtered) > 1:
                        confirmed_preds = top_pred_filtered[:-1]
                        unconfirmed_pred = top_pred_filtered[-1]
                    else:
                        confirmed_preds = []
                        unconfirmed_pred = top_pred_filtered[0]
                    
                    if confirmed_preds:
                        conf_t, conf_v = zip(*confirmed_preds)
                        conf_t_mst = [self.to_mst(t) for t in conf_t]
                        self.ax1.plot(conf_t_mst, conf_v, '-', color='white', linewidth=2.5, zorder=2)
                        self.ax1.plot(conf_t_mst, conf_v, 'o', color='white', markersize=10, zorder=3, markeredgecolor='#FAC01A', markeredgewidth=2, label='Predicted (Confirmed)')
                        for t, v in zip(conf_t_mst, conf_v):
                            self.ax1.text(t, v, self.format_time_12h(self.to_mst(t)), fontsize=9, ha='center', va='bottom', color='white', weight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#002454', alpha=0.7))
                    
                    unconf_t_mst = self.to_mst(unconfirmed_pred[0])
                    unconf_v = unconfirmed_pred[1]
                    self.ax1.plot([unconf_t_mst], [unconf_v], 'o', color='white', markersize=10, zorder=3, markeredgecolor='#FAC01A', markeredgewidth=2, label='Predicted (Forecast)')
                    self.ax1.text(unconf_t_mst, unconf_v, self.format_time_12h(unconf_t_mst), fontsize=9, ha='center', va='bottom', color='white', weight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#002454', alpha=0.7))
                    
                    if self.top_actual_times and top_pred_filtered:
                        last_actual_t_mst = top_act_t_mst[-1] if top_act_t_mst else None
                        last_actual_v = top_act_v[-1] if top_act_v else None
                        if last_actual_t_mst is not None:
                            first_pred_t_mst = self.to_mst(top_pred_filtered[0][0])
                            first_pred_v = top_pred_filtered[0][1]
                            self.ax1.plot([last_actual_t_mst, first_pred_t_mst], [last_actual_v, first_pred_v], '--', color='white', linewidth=2.5, zorder=2)
                    
                    if len(top_pred_filtered) > 1:
                        pred_times_mst = [self.to_mst(t) for t, _ in top_pred_filtered]
                        pred_values = [v for _, v in top_pred_filtered]
                        if len(pred_times_mst) >= 2:
                            if len(pred_times_mst) > 2:
                                self.ax1.plot(pred_times_mst[:-1], pred_values[:-1], '-', color='white', linewidth=2.5, zorder=2)
                            self.ax1.plot(pred_times_mst[-2:], pred_values[-2:], '--', color='white', linewidth=2.5, zorder=2)
            
            bottom_filtered = [(t, v) for t, v in zip(self.bottom_times, self.bottom_values) if window_start <= t <= window_end]
            if bottom_filtered:
                bottom_t, bottom_v = zip(*bottom_filtered)
                bottom_t_mst = [self.to_mst(t) for t in bottom_t]
                self.ax2.plot(bottom_t_mst, bottom_v, '-', color='#FAC01A', linewidth=2.5, zorder=2)
                self.ax2.plot(bottom_t_mst, bottom_v, 'o', color='#FAC01A', markersize=10, label='Actual', zorder=3, markeredgecolor='white', markeredgewidth=1.5)
                for t, v in zip(bottom_t_mst, bottom_v):
                    self.ax2.text(t, v, self.format_time_12h(t), fontsize=9, ha='center', va='bottom', color='white', weight='bold')
            
            if self.bottom_times:
                earliest_time = min(self.bottom_times)
                window_start_mst = self.to_mst(max(earliest_time, window_start))
            else:
                window_start_mst = self.to_mst(window_start)
            window_end_mst = self.to_mst(window_end)
            self.ax1.set_xlim(window_start_mst, window_end_mst)
            self.ax2.set_xlim(window_start_mst, window_end_mst)
            
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
            try:
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
                    
                    should_add_actual_data = True
                    if self.bottom_times:
                        time_diff = abs((new_timestamp - self.bottom_times[-1]).total_seconds())
                        if time_diff < 30:
                            should_add_actual_data = False
                    
                    if should_add_actual_data:
                        self.bottom_times.append(new_timestamp)
                        self.bottom_values.append(new_value)
                        self.last_update_time = new_timestamp
            except Exception as e:
                print(f"Error updating bottom graph: {e}")
        
        def update_data(self):
            try:
                current_time = datetime.now(timezone.utc)
                
                should_generate_prediction = False
                if self.top_pred_times:
                    last_pred_time = self.top_pred_times[-1]
                    time_until_pred = (last_pred_time - current_time).total_seconds()
                    if time_until_pred <= 30:
                        should_generate_prediction = True
                else:
                    should_generate_prediction = True
                
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
                    
                    should_add_actual_data = True
                    if self.bottom_times:
                        time_diff = abs((new_timestamp - self.bottom_times[-1]).total_seconds())
                        if time_diff < 30:
                            should_add_actual_data = False
                    
                    if should_add_actual_data:
                        self.bottom_times.append(new_timestamp)
                        self.bottom_values.append(new_value)
                        self.last_update_time = new_timestamp
                    
                    if should_generate_prediction:
                        history_kw = [v / 1000.0 for v in self.bottom_values[-6:]]
                        while len(history_kw) < 6:
                            history_kw.insert(0, history_kw[0] if history_kw else new_value / 1000.0)
                        
                        next_pred_time = new_timestamp + timedelta(minutes=15)
                        
                        prediction_exists = False
                        if self.top_pred_times:
                            for existing_pred_time in self.top_pred_times:
                                time_diff = abs((next_pred_time - existing_pred_time).total_seconds())
                                if time_diff < 300:
                                    prediction_exists = True
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
                            
                            self.top_pred_times.append(next_pred_time)
                            self.top_pred_values.append(predicted_watts)
            except Exception as e:
                print(f"Error updating data: {e}")
                import traceback
                traceback.print_exc()
        
        def check_and_update_if_needed(self):
            try:
                current_time = datetime.now(timezone.utc)
                
                should_update = False
                if self.top_pred_times:
                    next_pred_time = self.top_pred_times[-1]
                    time_until_pred = (next_pred_time - current_time).total_seconds()
                    if time_until_pred <= 30:
                        should_update = True
                
                if not should_update and hasattr(self, 'last_update_time') and self.last_update_time:
                    time_since_update = (current_time - self.last_update_time).total_seconds()
                    if time_since_update > 930:
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
    parser = argparse.ArgumentParser(description='Live Energy Forecast Application')
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