"""
TrafficAI Pro - Flask Backend
COMPLETE VERSION WITH ALL ROUTES
"""

from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps
import os
from datetime import datetime
import json
import threading
import cv2
import torch
import numpy as np
import random
from collections import deque
from ultralytics import YOLO
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import boto3
from botocore.exceptions import NoCredentialsError


# Flask Configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///traffic_monitor.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['STATIC_FOLDER'] = 'static'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

db = SQLAlchemy(app)

# Ensure folders exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'], 
               os.path.join(app.config['STATIC_FOLDER'], 'charts'),
               os.path.join(app.config['STATIC_FOLDER'], 'videos')]:
    os.makedirs(folder, exist_ok=True)

# Global dictionary to track analysis progress
analysis_progress = {}

# ============================================================================
# YOLO + DDQN TRAFFIC MONITORING CODE (PRESERVED FROM ORIGINAL)
# ============================================================================

class DQN(nn.Module):
    """Deep Q-Network for traffic signal optimization"""
    def __init__(self, input_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, input_size)
        )

    def forward(self, x):
        return self.fc(x)


class DDQNAgent:
    """Double Deep Q-Network Agent for adaptive traffic control"""
    def __init__(self, n_lanes):
        self.n_lanes = n_lanes
        self.q_eval = DQN(n_lanes)
        self.q_target = DQN(n_lanes)
        self.q_target.load_state_dict(self.q_eval.state_dict())
        self.memory = deque(maxlen=10000)
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.optimizer = optim.Adam(self.q_eval.parameters(), lr=0.001)

    def choose_action(self, state):
        """Select action using epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.n_lanes - 1)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        return torch.argmax(self.q_eval(state_tensor)).item()

    def store_transition(self, s, a, r, s_):
        """Store experience in replay memory"""
        self.memory.append((s, a, r, s_))

    def learn(self):
        """Update Q-network using experience replay"""
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s_ = zip(*batch)
        s = torch.FloatTensor(s)
        a = torch.LongTensor(a).unsqueeze(1)
        r = torch.FloatTensor(r).unsqueeze(1)
        s_ = torch.FloatTensor(s_)

        q_values = self.q_eval(s).gather(1, a)
        next_actions = self.q_eval(s_).argmax(1).unsqueeze(1)
        q_targets = self.q_target(s_).gather(1, next_actions)
        target = r + self.gamma * q_targets

        loss = nn.MSELoss()(q_values, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self):
        """Sync target network with evaluation network"""
        self.q_target.load_state_dict(self.q_eval.state_dict())

    def save_model(self, path):
        """Save trained model"""
        torch.save(self.q_eval.state_dict(), path)

    def load_model(self, path):
        """Load pre-trained model"""
        try:
            self.q_eval.load_state_dict(torch.load(path))
            self.q_target.load_state_dict(self.q_eval.state_dict())
            return True
        except FileNotFoundError:
            return False


# Vehicle classes recognized by YOLO
VEHICLE_CLASSES = ['car', 'motorcycle', 'bus', 'truck', 'bicycle']


def get_vehicle_count(model, frame):
    """Count vehicles in a frame using YOLO"""
    result = model(frame)[0]
    return sum(1 for box in result.boxes if model.names[int(box.cls[0])] in VEHICLE_CLASSES)


def read_lane_frames(cap_list):
    """Read frames from multiple video captures"""
    return [cap.read()[1] if cap.isOpened() else None for cap in cap_list]


def run_traffic_controller(n_lanes, video_paths, output_path, model_path="traffic_agent.pth", 
                          progress_callback=None, analysis_id=None):
    """
    Main traffic controller with YOLO detection and DDQN optimization
    """
    start_time = time.time()
    
    def update_progress(stage, percent, message, current_frame=0, total_frames=0, vehicle_count=0):
        if analysis_id and analysis_id in analysis_progress:
            elapsed = time.time() - start_time
            analysis_progress[analysis_id] = {
                'stage': stage,
                'percent': percent,
                'message': message,
                'elapsed_time': elapsed,
                'current_frame': current_frame,
                'total_frames': total_frames,
                'vehicle_count': vehicle_count
            }
    
    update_progress('initialization', 5, 'Validating video files...')
    print(f" Starting traffic analysis for {n_lanes} lanes...")
    print(f" Video paths: {video_paths}")
    
    # Validate all video files exist
    for i, video_path in enumerate(video_paths):
        if not os.path.exists(video_path):
            error_msg = f"Video file not found: {video_path}"
            print(f" {error_msg}")
            raise FileNotFoundError(error_msg)
        print(f" Video {i+1} validated: {video_path}")
    
    update_progress('initialization', 8, 'Loading YOLO model...')
    print(" Loading YOLO model...")
    
    # Load YOLO v11 model for vehicle detection
    model = YOLO('yolo11n.pt')
    print("YOLO v11 model loaded successfully")
    agent = DDQNAgent(n_lanes)
    
    update_progress('initialization', 10, 'Loading DDQN agent...')
    
    # Try to load pre-trained model
    if agent.load_model(model_path):
        print(f" Loaded previous knowledge from {model_path}")
    else:
        print(" No previous model found – starting fresh.")

    update_progress('initialization', 15, 'Opening video files...')
    print(f"📹 Opening {len(video_paths)} video file(s)...")
    
    # Load all video sources (lanes)
    caps = []
    for i, path in enumerate(video_paths):
        print(f"   Opening video {i+1}/{len(video_paths)}: {path}")
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            print(f"   ✅ Video {i+1} opened successfully")
            caps.append(cap)
        else:
            error_msg = f"Failed to open video file: {path}"
            print(f"    {error_msg}")
            raise ValueError(error_msg)
    
    print(f" All {len(caps)} video files opened successfully")

    # Get video dimensions and properties
    print(" Reading video properties...")
    width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(caps[0].get(cv2.CAP_PROP_FPS))
    target_height = min(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for cap in caps)
    total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps)
    
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    print(f"   Duration: {total_frames/fps:.2f} seconds")

    update_progress('initialization', 20, 'Setting up output video...')
    print(" Setting up output video writer...")
    
    # Setup output video with H264 codec for better browser support
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H264 codec
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width * n_lanes, target_height)
    )
    
    if not out.isOpened():
        print("  H264 codec failed, trying mp4v...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width * n_lanes, target_height))
    
    print(f" Video writer created with codec: {'H264' if out.isOpened() else 'mp4v'}")

    frame_count = 0
    rewards, car_counts = [], []
    lane_metrics = {i: {'total_vehicles': 0, 'green_time': 0} for i in range(n_lanes)}
    total_vehicle_count = 0  # Initialize total vehicle counter
    
    # Track per-lane vehicle counts over time for charting
    per_lane_counts = {i: [] for i in range(n_lanes)}

    cooldown_frames = 30
    current_action = 0
    frame_since_action = cooldown_frames
    prev_counts = [0] * n_lanes

    update_progress('ml_analysis', 25, 'Starting ML-based traffic analysis...', 0, total_frames, 0)
    print("🚦 Running adaptive traffic controller with ML analysis...")
    print(f"   📊 Processing {total_frames} frames")
    print(f"   ⏱️  Estimated time: {(total_frames/fps/60):.1f} minutes")
    print("   🎬 Starting frame-by-frame processing NOW...")

    while True:
        # Log every 10 frames in early stages
        if frame_count < 100 and frame_count % 10 == 0:
            print(f"   📹 Reading frame {frame_count}...")
        
        frames = read_lane_frames(caps)
        if any(f is None for f in frames):
            print(f"   ⚠️  End of video reached at frame {frame_count}")
            break
        
        if frame_count == 0:
            print("   First frame read successfully - processing starting!")

        frames = [cv2.resize(frame, (width, target_height)) for frame in frames]
        counts = [get_vehicle_count(model, f) for f in frames]

        if frame_since_action >= cooldown_frames:
            action = agent.choose_action(counts)
            current_action = action
            frame_since_action = 0
        else:
            action = current_action
            frame_since_action += 1

        total_prev = sum(prev_counts)
        total_curr = sum(counts)
        reward = (total_prev - total_curr) - 0.1 * np.std(counts)
        prev_counts = counts[:]

        next_counts = counts[:]
        next_counts[action] = max(0, next_counts[action] - random.randint(2, 5))

        agent.store_transition(counts, action, reward, next_counts)
        agent.learn()
        
        if frame_count % 20 == 0:
            agent.update_target()

        lane_metrics[action]['green_time'] += 1
        for i, count in enumerate(counts):
            lane_metrics[i]['total_vehicles'] += count
            per_lane_counts[i].append(count)  # Store per-lane count for this frame
        
        # Update total vehicle count
        total_vehicle_count = sum(lane_metrics[i]['total_vehicles'] for i in range(n_lanes))

        signal_colors = [(0, 0, 255)] * n_lanes
        signal_colors[action] = (0, 255, 0)

        annotated_frames = []
        for i, (frame, count, color) in enumerate(zip(frames, counts, signal_colors)):
            frame_copy = frame.copy()
            
            cv2.rectangle(frame_copy, (10, 10), (100, 100), color, -1)
            cv2.putText(frame_copy, f"Lane {i+1}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.rectangle(frame_copy, (10, 120), (200, 180), (0, 0, 0), -1)
            cv2.putText(frame_copy, f"Vehicles: {count}", (20, 155),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            annotated_frames.append(frame_copy)

        combined = np.hstack(annotated_frames)
        
        cv2.rectangle(combined, (0, 0), (combined.shape[1], 40), (50, 50, 50), -1)
        cv2.putText(combined, f"Frame: {frame_count}/{total_frames} | Total Reward: {sum(rewards):.2f}",
                   (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(combined)
        
        rewards.append(reward)
        car_counts.append(sum(counts))
        frame_count += 1

        if frame_count % 30 == 0:
            progress_percent = 25 + int((frame_count / total_frames) * 65)
            elapsed = time.time() - start_time
            eta = (elapsed / frame_count) * (total_frames - frame_count) if frame_count > 0 else 0
            update_progress('ml_analysis', progress_percent, 
                          f'ML Analysis: Frame {frame_count}/{total_frames} (ETA: {eta:.0f}s)',
                          frame_count, total_frames, total_vehicle_count)

    update_progress('finalization', 95, f'Finalizing analysis... Processed {total_vehicle_count} vehicles', 
                   frame_count, total_frames, total_vehicle_count)
    
    for cap in caps:
        cap.release()
    out.release()
    
    agent.save_model(model_path)
    
    # Calculate metrics
    # Total cumulative vehicle detections across all frames
    cumulative_detections = sum(lane_metrics[i]['total_vehicles'] for i in range(n_lanes))
    
    # Estimate unique vehicles (assuming average vehicle stays in frame for ~3 seconds at 30fps = ~90 frames)
    avg_frames_per_vehicle = 90  # Adjust based on typical vehicle dwell time
    estimated_unique_vehicles = max(1, cumulative_detections // avg_frames_per_vehicle)
    
    # Video duration in seconds
    video_duration = frame_count / fps if frame_count > 0 and fps > 0 else 1
    
    # Throughput: estimated unique vehicles per second
    throughput = estimated_unique_vehicles / video_duration if video_duration > 0 else 0
    
    # Average wait time: based on red light duration
    # Calculate average red time per lane
    avg_red_time_frames = sum(
        (frame_count - lane_metrics[i]['green_time']) for i in range(n_lanes)
    ) / n_lanes
    avg_wait_time = avg_red_time_frames / fps if fps > 0 else 0
    
    # Calculate efficiency (capped at 100%)
    lane_utilization = [lane_metrics[i]['green_time'] / max(frame_count, 1) for i in range(n_lanes)]
    fairness = 1 - np.std(lane_utilization) if n_lanes > 1 else 1.0  # 0 to 1
    
    # Normalize throughput to 0-1 scale (assume max reasonable throughput is 5 vehicles/sec)
    normalized_throughput = min(throughput / 5.0, 1.0)
    
    # Efficiency: weighted average of throughput and fairness (0-100%)
    efficiency = min(int((normalized_throughput * 50 + fairness * 50)), 100)
    
    update_progress('completed', 100, f'Analysis completed! {estimated_unique_vehicles} vehicles detected', 
                   frame_count, total_frames, total_vehicle_count)
    
    elapsed_time = time.time() - start_time
    print(f"✅ Analysis completed in {elapsed_time:.1f} seconds")
    
    return {
        'rewards': rewards,
        'car_counts': car_counts,
        'lane_metrics': lane_metrics,
        'per_lane_counts': per_lane_counts,  # Add per-lane tracking data
        'total_vehicles': estimated_unique_vehicles,  # Use estimated unique vehicles
        'avg_wait_time': round(avg_wait_time, 2),
        'throughput': round(throughput, 2),
        'efficiency': efficiency,
        'total_frames': frame_count,
        'processing_time': round(elapsed_time, 1)
    }


def generate_charts(rewards, car_counts, analysis_id, lane_metrics=None, n_lanes=4, per_lane_counts=None):
    """Generate matplotlib charts matching the style of sample images"""
    charts = {}
    
    # Convert rewards to error rate (negative rewards represent system inefficiency)
    error_rates = [-r for r in rewards]  # Flip sign to make it "error rate"
    
    # Chart 1: Error Rate Over Time (was Negative Reward)
    plt.figure(figsize=(12, 6))
    plt.plot(error_rates, color='#3b82f6', linewidth=1.5, alpha=0.8)
    plt.title('Error Rate Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Frame', fontsize=12)
    plt.ylabel('Error Rate', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    reward_path = f'/static/charts/rewards_{analysis_id}.png'
    plt.savefig(os.path.join(app.config['STATIC_FOLDER'], 'charts', f'rewards_{analysis_id}.png'), dpi=100)
    plt.close()
    charts['rewards'] = reward_path
    
    # Chart 2: Car Count Over Time per Lane - USE REAL PER-LANE DATA
    if per_lane_counts and n_lanes:
        plt.figure(figsize=(12, 6))
        
        # Generate colors dynamically based on number of lanes
        cmap = plt.cm.get_cmap('tab10')  # Supports up to 10 distinct colors
        colors = [cmap(i / max(n_lanes - 1, 1)) for i in range(n_lanes)]
        
        # Plot actual per-lane counts tracked during processing
        for i in range(n_lanes):
            if i in per_lane_counts and len(per_lane_counts[i]) > 0:
                frames = list(range(len(per_lane_counts[i])))
                plt.plot(frames, per_lane_counts[i], color=colors[i], linewidth=1.8, 
                        label=f'Lane {i+1}', alpha=0.85)
        
        plt.title('Car Count Over Time per Lane', fontsize=14, fontweight='bold')
        plt.xlabel('Frame', fontsize=12)
        plt.ylabel('Car Count', fontsize=12)
        plt.legend(loc='upper right', framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        count_path = f'/static/charts/counts_{analysis_id}.png'
        plt.savefig(os.path.join(app.config['STATIC_FOLDER'], 'charts', f'counts_{analysis_id}.png'), dpi=100)
        plt.close()
        charts['counts'] = count_path
    else:
        # Fallback: single line for total count
        plt.figure(figsize=(12, 6))
        plt.plot(car_counts, color='#764ba2', linewidth=2)
        plt.title('Total Vehicle Count Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Frame', fontsize=12)
        plt.ylabel('Vehicles', fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        count_path = f'/static/charts/counts_{analysis_id}.png'
        plt.savefig(os.path.join(app.config['STATIC_FOLDER'], 'charts', f'counts_{analysis_id}.png'), dpi=100)
        plt.close()
        charts['counts'] = count_path
    
    return charts


# ============================================================================
# DATABASE MODELS
# ============================================================================

class User(db.Model):
    """User model"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    locations = db.relationship('Location', backref='user', lazy=True, cascade='all, delete-orphan')


class Location(db.Model):
    """Traffic location/intersection"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    address = db.Column(db.String(200))
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    num_lanes = db.Column(db.Integer, default=4)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    analyses = db.relationship('Analysis', backref='location', lazy=True, cascade='all, delete-orphan')
    signals = db.relationship('Signal', backref='location', lazy=True, cascade='all, delete-orphan')


class Analysis(db.Model):
    """Traffic analysis record"""
    id = db.Column(db.Integer, primary_key=True)
    location_id = db.Column(db.Integer, db.ForeignKey('location.id'), nullable=False)
    video_path = db.Column(db.Text)
    output_video = db.Column(db.String(200))
    status = db.Column(db.String(20), default='pending')
    total_vehicles = db.Column(db.Integer)
    avg_wait_time = db.Column(db.Float)
    throughput = db.Column(db.Float)
    efficiency = db.Column(db.Integer)
    duration = db.Column(db.String(50))
    peak_hour = db.Column(db.String(50))
    results_json = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)


class Signal(db.Model):
    """Traffic signal configuration"""
    id = db.Column(db.Integer, primary_key=True)
    location_id = db.Column(db.Integer, db.ForeignKey('location.id'), nullable=False)
    name = db.Column(db.String(100))
    status = db.Column(db.String(20), default='active')
    phases = db.Column(db.Integer, default=4)
    cycle_time = db.Column(db.Integer, default=120)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)


# ============================================================================
# AUTHENTICATION
# ============================================================================

def login_required(f):
    """Login required decorator"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


# ============================================================================
# PAGE ROUTES (FIXED: ADDED ALL MISSING ROUTES)
# ============================================================================

@app.route('/')
@login_required
def index():
    """Home page"""
    return render_template('home.html')


@app.route('/login')
def login():
    """Login page"""
    return render_template('login.html')


@app.route('/register')
def register():
    """Register page"""
    return render_template('register.html')


@app.route('/signal-inventory')
@login_required
def signal_inventory():
    """Signal inventory page"""
    return render_template('signal-inventory.html')


@app.route('/signal-training')
@login_required
def signal_training():
    """Signal training page"""
    return render_template('signal-training.html')


@app.route('/analysis/<int:analysis_id>')
@login_required
def analysis_result(analysis_id):
    """Analysis result page"""
    return render_template('analysis-result.html')


# ============================================================================
# AUTHENTICATION API
# ============================================================================

@app.route('/api/auth/register', methods=['POST'])
def api_register():
    """Register new user"""
    data = request.get_json()
    
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    if not username or not email or not password:
        return jsonify({'error': 'Missing required fields'}), 400
    
    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'Username already exists'}), 400
    
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already exists'}), 400
    
    user = User(
        username=username,
        email=email,
        password=generate_password_hash(password)
    )
    
    db.session.add(user)
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Registration successful'}), 201


@app.route('/api/auth/login', methods=['POST'])
def api_login():
    """Login user"""
    data = request.get_json()
    
    email = data.get('email')
    password = data.get('password')
    
    user = User.query.filter_by(email=email).first()
    
    if not user or not check_password_hash(user.password, password):
        return jsonify({'error': 'Invalid email or password'}), 401
    
    session['user_id'] = user.id
    session['username'] = user.username
    
    return jsonify({
        'success': True,
        'user': {
            'id': user.id,
            'username': user.username,
            'email': user.email
        },
        'redirect': '/'
    }), 200


@app.route('/api/auth/logout', methods=['POST'])
def api_logout():
    """Logout user"""
    session.clear()
    return jsonify({'success': True}), 200


# ============================================================================
# DASHBOARD API
# ============================================================================

@app.route('/api/dashboard/stats', methods=['GET'])
@login_required
def api_dashboard_stats():
    """Get dashboard statistics"""
    user_id = session['user_id']
    
    locations = Location.query.filter_by(user_id=user_id).count()
    signals = Signal.query.join(Location).filter(Location.user_id == user_id).count()
    analyses = Analysis.query.join(Location).filter(Location.user_id == user_id).count()
    
    completed_analyses = Analysis.query.join(Location).filter(
        Location.user_id == user_id,
        Analysis.status == 'completed',
        Analysis.efficiency.isnot(None)
    ).all()
    
    avg_efficiency = None
    if completed_analyses:
        avg_efficiency = round(sum(a.efficiency for a in completed_analyses) / len(completed_analyses), 1)
    
    return jsonify({
        'locations': locations,
        'signals': signals,
        'analyses': analyses,
        'avg_efficiency': avg_efficiency
    })


@app.route('/api/dashboard/activity', methods=['GET'])
@login_required
def api_dashboard_activity():
    """Get recent activity"""
    user_id = session['user_id']
    limit = request.args.get('limit', 10, type=int)
    
    analyses = Analysis.query.join(Location).filter(
        Location.user_id == user_id
    ).order_by(Analysis.created_at.desc()).limit(limit).all()
    
    return jsonify([{
        'id': a.id,
        'location_name': a.location.name,
        'status': a.status,
        'created_at': a.created_at.isoformat()
    } for a in analyses])


@app.route('/api/dashboard/charts', methods=['GET'])
@login_required
def api_dashboard_charts():
    """Get chart data for dashboard"""
    user_id = session['user_id']
    
    analyses = Analysis.query.join(Location).filter(
        Location.user_id == user_id,
        Analysis.status == 'completed'
    ).order_by(Analysis.created_at.desc()).limit(10).all()
    
    traffic_flow = []
    efficiency = []
    
    for analysis in analyses:
        traffic_flow.append({
            'vehicles': analysis.total_vehicles or 0,
            'location': analysis.location.name
        })
        efficiency.append({
            'efficiency': analysis.efficiency or 0,
            'location': analysis.location.name
        })
    
    return jsonify({
        'traffic_flow': traffic_flow,
        'efficiency': efficiency
    })


# ============================================================================
# LOCATION API
# ============================================================================

@app.route('/api/locations', methods=['GET'])
@login_required
def api_get_locations():
    """Get all locations"""
    user_id = session['user_id']
    locations = Location.query.filter_by(user_id=user_id).all()
    
    return jsonify([{
        'id': loc.id,
        'name': loc.name,
        'address': loc.address,
        'num_lanes': loc.num_lanes,
        'created_at': loc.created_at.isoformat()
    } for loc in locations])


@app.route('/api/locations', methods=['POST'])
@login_required
def api_create_location():
    """Create new location"""
    user_id = session['user_id']
    data = request.get_json()
    
    location = Location(
        user_id=user_id,
        name=data.get('name'),
        address=data.get('address'),
        latitude=data.get('latitude'),
        longitude=data.get('longitude'),
        num_lanes=data.get('num_lanes', 4)
    )
    
    db.session.add(location)
    db.session.commit()
    
    # Create default signals
    for i in range(location.num_lanes):
        signal = Signal(
            location_id=location.id,
            name=f"Signal {i+1}",
            status='active'
        )
        db.session.add(signal)
    
    db.session.commit()
    
    return jsonify({'success': True, 'id': location.id}), 201


@app.route('/api/locations/<int:location_id>', methods=['DELETE'])
@login_required
def api_delete_location(location_id):
    """Delete location"""
    user_id = session['user_id']
    location = Location.query.filter_by(id=location_id, user_id=user_id).first()
    
    if not location:
        return jsonify({'error': 'Location not found'}), 404
    
    db.session.delete(location)
    db.session.commit()

    return jsonify({'success': True}), 200


# ============================================================================
# ANALYSIS API
# ============================================================================

@app.route('/api/analyses', methods=['GET'])
@login_required
def api_get_analyses():
    """Get all analyses"""
    user_id = session['user_id']
    limit = request.args.get('limit', 10, type=int)
    
    analyses = Analysis.query.join(Location).filter(
        Location.user_id == user_id
    ).order_by(Analysis.created_at.desc()).limit(limit).all()

    return jsonify([{
        'id': a.id,
        'location_name': a.location.name,
        'status': a.status,
        'created_at': a.created_at.isoformat()
    } for a in analyses])


@app.route('/api/analyses/<int:analysis_id>', methods=['GET'])
@login_required
def api_get_analysis(analysis_id):
    """Get specific analysis"""
    user_id = session['user_id']
    analysis = Analysis.query.join(Location).filter(
        Analysis.id == analysis_id,
        Location.user_id == user_id
    ).first()

    if not analysis:
        return jsonify({'error': 'Analysis not found'}), 404

    result = {
        'id': analysis.id,
        'location_name': analysis.location.name,
        'status': analysis.status,
        'total_vehicles': analysis.total_vehicles,
        'avg_wait_time': analysis.avg_wait_time,
        'throughput': analysis.throughput,
        'efficiency': analysis.efficiency,
        'output_video': analysis.output_video,
        'duration': analysis.duration,
        'peak_hour': analysis.peak_hour,
        'created_at': analysis.created_at.isoformat()
    }

    if analysis.results_json:
        try:
            detailed_results = json.loads(analysis.results_json)
            result.update(detailed_results)
        except:
            pass

    return jsonify(result)


@app.route('/api/analyses/<int:analysis_id>/progress', methods=['GET'])
def api_get_analysis_progress(analysis_id):
    """Get analysis progress"""
    if analysis_id in analysis_progress:
        progress = analysis_progress[analysis_id]
        return jsonify({
            'stage': progress.get('stage', 'initializing'),
            'percent': progress.get('percent', 0),
            'message': progress.get('message', 'Processing...'),
            'elapsed_time': round(progress.get('elapsed_time', 0), 1),
            'current_frame': progress.get('current_frame', 0),
            'total_frames': progress.get('total_frames', 0),
            'vehicle_count': progress.get('vehicle_count', 0),
            'is_ml_processing': progress.get('stage') == 'ml_analysis'
        })
    
    analysis = Analysis.query.get(analysis_id)
    if analysis:
        if analysis.status == 'completed':
            return jsonify({
                'stage': 'completed',
                'percent': 100,
                'message': 'Analysis completed',
                'elapsed_time': 0,
                'current_frame': 0,
                'total_frames': 0,
                'vehicle_count': 0,
                'is_ml_processing': False
            })
        elif analysis.status == 'failed':
            return jsonify({
                'stage': 'failed',
                'percent': 0,
                'message': 'Analysis failed',
                'elapsed_time': 0,
                'current_frame': 0,
                'total_frames': 0,
                'vehicle_count': 0,
                'is_ml_processing': False
            })
        elif analysis.status == 'processing':
            # Analysis started but not in progress dict yet
            return jsonify({
                'stage': 'initialization',
                'percent': 5,
                'message': 'Starting analysis...',
                'elapsed_time': 0,
                'current_frame': 0,
                'total_frames': 0,
                'vehicle_count': 0,
                'is_ml_processing': False
            })
    
    return jsonify({
        'stage': 'pending',
        'percent': 0,
        'message': 'Waiting to start...',
        'elapsed_time': 0,
        'current_frame': 0,
        'total_frames': 0,
        'vehicle_count': 0,
        'is_ml_processing': False
    })


@app.route('/api/analyses', methods=['POST'])
@login_required
def api_create_analysis():
    """Start new traffic analysis"""
    user_id = session['user_id']
    location_id = request.form.get('location_id')
    
    location = Location.query.filter_by(id=location_id, user_id=user_id).first()
    if not location:
        return jsonify({'error': 'Location not found'}), 404

    video_paths = []
    num_lanes = location.num_lanes
    
    # Multiple video upload support
    for i in range(num_lanes):
        video_key = f'video_lane_{i+1}'
        if video_key in request.files:
            video = request.files[video_key]
            if video.filename:
                filename = secure_filename(video.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                location_folder = os.path.join(app.config['UPLOAD_FOLDER'], 
                                              f"location_{location_id}")
                os.makedirs(location_folder, exist_ok=True)
                
                save_path = os.path.join(location_folder, 
                                       f'lane_{i+1}_{timestamp}_{filename}')
                video.save(save_path)
                video_paths.append(save_path)
    
    # Fallback: single video
    if len(video_paths) == 1:
        video_paths = video_paths * num_lanes
    elif len(video_paths) == 0:
        if 'video' in request.files:
            video = request.files['video']
            if video.filename:
                filename = secure_filename(video.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                location_folder = os.path.join(app.config['UPLOAD_FOLDER'], 
                                              f"location_{location_id}")
                os.makedirs(location_folder, exist_ok=True)
                
                save_path = os.path.join(location_folder, 
                                       f'lane_all_{timestamp}_{filename}')
                video.save(save_path)
                video_paths = [save_path] * num_lanes

    if not video_paths or len(video_paths) < num_lanes:
        return jsonify({
            'error': f'Please provide {num_lanes} video files (one per lane)'
        }), 400
    
    # Validate all video files exist
    print(f"📹 Validating {len(video_paths)} video files...")
    for i, path in enumerate(video_paths):
        if not os.path.exists(path):
            print(f"❌ Video file not found: {path}")
            return jsonify({
                'error': f'Video file for lane {i+1} not found: {path}'
            }), 400
        file_size = os.path.getsize(path) / (1024 * 1024)  # MB
        print(f"✅ Lane {i+1}: {path} ({file_size:.2f} MB)")

    analysis = Analysis(
        location_id=location_id,
        video_path=json.dumps(video_paths),
        status='processing'
    )
    
    db.session.add(analysis)
    db.session.commit()

    analysis_progress[analysis.id] = {
        'stage': 'initializing',
        'percent': 0,
        'message': 'Starting analysis...',
        'elapsed_time': 0,
        'current_frame': 0,
        'total_frames': 0,
        'vehicle_count': 0
    }
    
    print(f"🚀 Starting analysis {analysis.id} with {location.num_lanes} lanes")
    print(f"   Creating thread with args:")
    print(f"     - analysis_id: {analysis.id}")
    print(f"     - n_lanes: {location.num_lanes}")
    print(f"     - location_name: {location.name}")
    print(f"     - video_paths: {video_paths}")

    thread = threading.Thread(
        target=process_analysis_async,
        args=(analysis.id, location.num_lanes, video_paths, location.name)
    )
    thread.daemon = True
    
    print(f"   Thread object created: {thread}")
    print(f"   Thread daemon: {thread.daemon}")
    print(f"   Calling thread.start()...", flush=True)
    
    thread.start()
    
    print(f"✅ thread.start() called successfully", flush=True)
    print(f"   Thread is alive: {thread.is_alive()}", flush=True)
    print(f"   Thread ID: {thread.ident}", flush=True)

    return jsonify({
        'success': True,
        'id': analysis.id,
        'message': 'Analysis started'
    }), 201

def upload_to_s3(local_file, bucket_name, s3_path):
    """Uploads a file to S3 and returns its public URL."""
    s3 = boto3.client('s3')
    try:
        s3.upload_file(local_file, bucket_name, s3_path)
        # Make the object publicly accessible
        
        public_url = f"https://{bucket_name}.s3.ap-south-1.amazonaws.com/{s3_path}"
        print(f" Uploaded: {public_url}")
        return public_url
    except FileNotFoundError:
        print(f" File not found: {local_file}")
        return None
    except NoCredentialsError:
        print(" AWS credentials not found or invalid.")
        return None
    except Exception as e:
        print(f" S3 upload failed: {e}")
        return None


def process_analysis_async(analysis_id, n_lanes, video_paths, location_name):
    """Process traffic analysis asynchronously"""
    import sys
    sys.stdout.flush()  # Force output to show immediately
    
    print(f"\n{'='*60}", flush=True)
    print(f"📊 THREAD STARTED - Analysis ID {analysis_id}", flush=True)
    print(f"   Lanes: {n_lanes}", flush=True)
    print(f"   Videos: {video_paths}", flush=True)
    print(f"{'='*60}\n", flush=True)
    try:
        with app.app_context():
            print(f"✅ App context acquired for analysis {analysis_id}", flush=True)
            analysis = Analysis.query.get(analysis_id)
            
            if not analysis:
                print(f"❌ Analysis {analysis_id} not found in database!", flush=True)
                return
            
            print(f"✅ Analysis {analysis_id} found in database", flush=True)
            
            try:
                location_output_folder = os.path.join(
                    app.config['STATIC_FOLDER'], 
                    'videos',
                    f'location_{analysis.location_id}'
                )
                os.makedirs(location_output_folder, exist_ok=True)
                print(f"✅ Output folder created: {location_output_folder}", flush=True)
                
                output_filename = f'analysis_{analysis_id}_{location_name.replace(" ", "_")}.mp4'
                output_path = os.path.join(location_output_folder, output_filename)
                
                print(f"\n🎬 CALLING run_traffic_controller...", flush=True)
                print(f"   Analysis ID: {analysis_id}", flush=True)
                print(f"   Lanes: {n_lanes}", flush=True)
                print(f"   Output: {output_path}", flush=True)
                
                results = run_traffic_controller(
                    n_lanes=n_lanes,
                    video_paths=video_paths,
                    output_path=output_path,
                    analysis_id=analysis_id
                )
                
                print(f"\n✅ run_traffic_controller completed!", flush=True)
                print(f"📈 Generating charts for analysis {analysis_id}", flush=True)
                
                charts = generate_charts(
                    results['rewards'],
                    results['car_counts'],
                    analysis_id,
                    lane_metrics=results.get('lane_metrics'),
                    n_lanes=n_lanes,
                    per_lane_counts=results.get('per_lane_counts')
                )
                
                # === Upload video and charts to S3 ===
                BUCKET_NAME = "traffic369"
                print("📤 Uploading output video and charts to S3...")
                video_s3_url = None
                if os.path.exists(output_path):
                    video_s3_url = upload_to_s3(
                        output_path,
                        BUCKET_NAME,
                        f"videos/{os.path.basename(output_path)}"
                    )
                
                charts_s3 = {}
                for key, path in charts.items():
                    full_path = os.path.join(app.config['STATIC_FOLDER'], path.lstrip('/'))
                    if os.path.exists(full_path):
                        s3_key = f"charts/{os.path.basename(path)}"
                        s3_url = upload_to_s3(full_path, BUCKET_NAME, s3_key)
                        if s3_url:
                            charts_s3[key] = s3_url
                
                if video_s3_url:
                    analysis.output_video = video_s3_url
                else:
                    analysis.output_video = f"/static/videos/location_{analysis.location_id}/{os.path.basename(output_path)}"
                
                if charts_s3:
                    charts = charts_s3
                    print("✅ Uploaded successfully to S3.")

                print(f"✅ Charts generated successfully", flush=True)
                print(f"   Rewards chart: {charts.get('rewards', 'MISSING')}", flush=True)
                print(f"   Counts chart: {charts.get('counts', 'MISSING')}", flush=True)
                
                analysis.status = 'completed'
                analysis.total_vehicles = results['total_vehicles']
                analysis.avg_wait_time = results['avg_wait_time']
                analysis.throughput = results['throughput']
                analysis.efficiency = results['efficiency']
                analysis.duration = f"{results['processing_time']}s"
                analysis.completed_at = datetime.utcnow()
                detailed_results = {
                    'charts': charts,
                    'lane_metrics': results['lane_metrics'],
                    'processing_time': results['processing_time']
                }
                analysis.results_json = json.dumps(detailed_results)
                db.session.commit()
                
                if analysis_id in analysis_progress:
                    del analysis_progress[analysis_id]
                
                print(f"✅ Analysis {analysis_id} completed successfully", flush=True)
            except Exception as e:
                import traceback
                print(f"❌ Analysis {analysis_id} failed: {str(e)}", flush=True)
                print(f"   Traceback: {traceback.format_exc()}", flush=True)
                analysis.status = 'failed'
                db.session.commit()
                if analysis_id in analysis_progress:
                    analysis_progress[analysis_id] = {
                        'stage': 'failed',
                        'percent': 0,
                        'message': f'Analysis failed: {str(e)}',
                        'elapsed_time': 0
                    }
    except Exception as outer_e:
        import traceback
        print(f"\n{'='*60}", flush=True)
        print(f"💥 CRITICAL ERROR in process_analysis_async", flush=True)
        print(f"   Analysis ID: {analysis_id}", flush=True)
        print(f"   Error: {str(outer_e)}", flush=True)
        print(f"   Traceback:", flush=True)
        print(traceback.format_exc(), flush=True)
        print(f"{'='*60}\n", flush=True)
        
        # Try to update database even if outer error
        try:
            with app.app_context():
                analysis = Analysis.query.get(analysis_id)
                if analysis:
                    analysis.status = 'failed'
                    db.session.commit()
        except:
            print(f"❌ Could not update database for failed analysis {analysis_id}", flush=True)


@app.route('/api/analyses/<int:analysis_id>', methods=['DELETE'])
@login_required
def api_delete_analysis(analysis_id):
    """Delete an analysis"""
    user_id = session['user_id']
    
    # Find analysis and verify ownership through location
    analysis = Analysis.query.join(Location).filter(
        Analysis.id == analysis_id,
        Location.user_id == user_id
    ).first()
    
    if not analysis:
        return jsonify({'error': 'Analysis not found'}), 404
    
    try:
        # Delete associated files
        if analysis.output_video and os.path.exists(analysis.output_video.lstrip('/')):
            try:
                os.remove(analysis.output_video.lstrip('/'))
            except:
                pass
        
        # Delete charts if they exist in results_json
        if analysis.results_json:
            try:
                results = json.loads(analysis.results_json)
                if 'charts' in results:
                    for chart_url in results['charts'].values():
                        chart_path = chart_url.lstrip('/')
                        if os.path.exists(chart_path):
                            try:
                                os.remove(chart_path)
                            except:
                                pass
            except:
                pass
        
        # Remove from progress tracking if still there
        if analysis_id in analysis_progress:
            del analysis_progress[analysis_id]
        
        # Delete from database
        db.session.delete(analysis)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Analysis deleted'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


# ============================================================================
# SIGNAL API
# ============================================================================

@app.route('/api/signals', methods=['GET'])
@login_required
def api_get_signals():
    """Get all signals"""
    user_id = session['user_id']
    
    signals = Signal.query.join(Location).filter(
        Location.user_id == user_id
    ).all()

    return jsonify([{
        'id': sig.id,
        'name': sig.name or f'Signal {sig.id}',
        'location_name': sig.location.name,
        'status': sig.status,
        'phases': sig.phases,
        'cycle_time': sig.cycle_time
    } for sig in signals])


@app.route('/api/signals/<int:signal_id>', methods=['PATCH'])
@login_required
def api_update_signal(signal_id):
    """Update signal"""
    user_id = session['user_id']
    data = request.get_json()
    
    signal = Signal.query.join(Location).filter(
        Signal.id == signal_id,
        Location.user_id == user_id
    ).first()

    if not signal:
        return jsonify({'error': 'Signal not found'}), 404

    if 'status' in data:
        signal.status = data['status']
    
    signal.updated_at = datetime.utcnow()
    db.session.commit()

    return jsonify({'success': True}), 200


# ============================================================================
# INITIALIZATION
# ============================================================================

def init_db():
    """Initialize database"""
    with app.app_context():
        db.create_all()
        print("✅ Database initialized")


if __name__ == '__main__':
    init_db()
    print("🚀 TrafficAI Pro Backend Started")
    print("📡 Server running on http://localhost:5000")
    print("⚠️  Debug mode disabled for better threading support")
    print("   If you need to restart, press Ctrl+C and run again")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
