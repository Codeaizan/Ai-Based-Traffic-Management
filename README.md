AI Traffic Management System
An intelligent traffic control system that uses deep learning to analyze vehicle density across four lanes and dynamically allocates green signals to prevent congestion and deadlocks. The system is designed for simulation purposes and can be extended for real-world applications.

Table of Contents
Features

Installation

Usage

Configuration

Code Structure

Key Components

How It Works

Limitations

Future Improvements

License

Features
🚗 Vehicle Detection: Uses Faster R-CNN with ResNet-50 backbone pre-trained on the COCO dataset.

🚦 Anti-Deadlock Logic: Prevents lane starvation with dynamic priority weighting.

🚨 Emergency Threshold: Prioritizes overcrowded lanes (15+ vehicles).

⚡ GPU Acceleration: Optimized for NVIDIA GPUs using CUDA and cuDNN.

📊 Real-Time Simulation: Console-based visualization of traffic status.

⏱ Time-Decay Priority: Gradually increases priority of waiting lanes.

Installation
Requirements
Python 3.8+

NVIDIA GPU with CUDA 11.7+ (recommended)

4GB VRAM (for GPU acceleration)

8GB RAM (minimum)

Steps
Clone the repository:

bash
Copy
git clone https://github.com/yourusername/ai-traffic-management.git
cd ai-traffic-management
Install dependencies:

bash
Copy
pip install torch torchvision pillow numpy
Prepare your dataset:

Create a folder data/images/.

Add four traffic images named lane1.jpg, lane2.jpg, lane3.jpg, and lane4.jpg.

Usage
Run the system:

bash
Copy
python traffic_system.py
Expected Output:

Copy
========================================
Current Cycle: 14:35:22
Lane 1 (lane1.jpg):
  Vehicles: 12
  Status: GREEN
  Priority Weight: 0.72
Lane 2 (lane2.jpg):
  Vehicles: 8
  Status: RED
  Priority Weight: 1.21
Lane 3 (lane3.jpg):
  Vehicles: 9
  Status: RED
  Priority Weight: 1.15
Lane 4 (lane4.jpg):
  Vehicles: 14
  Status: RED
  Priority Weight: 1.18

Decision Logic:
- Consecutive greens: 0
- Emergency override: No
The system will continuously analyze the images and decide which lane should get the green signal.

Configuration
Modify the following parameters in the TrafficScheduler class for custom behavior:

python
Copy
self.params = {
    'max_consecutive': 2,       # Max consecutive green cycles
    'emergency_threshold': 15,  # Vehicles needed for emergency priority
    'time_decay': 0.9           # Priority decay per minute
}
Code Structure
Copy
ai-traffic-management/
├── data/
│   └── images/          # Input images (lane1.jpg, lane2.jpg, etc.)
├── traffic_system.py    # Main application
├── README.md            # Project documentation
└── requirements.txt     # Dependency list
Key Components
1. TrafficScheduler Class
Dynamic Priority Calculation: Adjusts lane priorities based on vehicle count and waiting time.

Emergency Override: Forces a green signal for lanes with more than 15 vehicles.

Weight Adjustment: Updates priority weights to prevent lane starvation.

Cycle Management: Ensures fair distribution of green signals.

2. Vehicle Detection
Faster R-CNN Model: Pre-trained on the COCO dataset for accurate vehicle detection.

Non-Maximum Suppression (NMS): Removes duplicate detections.

GPU Acceleration: Uses CUDA for faster inference.

3. Console Interface
Displays real-time lane statuses.

Shows decision-making factors (e.g., priority weights, emergency override).

Color-coded output for better visualization.

How It Works
Input: Four images representing traffic lanes.

Vehicle Detection: Faster R-CNN detects and counts vehicles in each lane.

Priority Calculation:

Each lane's priority is calculated using:

Copy
Priority = (Vehicle Count) × (Priority Weight) × (Time Decay) + (Emergency Bonus)
Time decay reduces priority for recently green lanes.

Decision Making:

The lane with the highest priority gets the green signal.

Emergency override forces a green signal for overcrowded lanes.

Output: Console display of lane statuses and decision logic.

Limitations
Requires clear visibility of vehicles in images.

Accuracy depends on camera angle and image quality.

Simulation-only (no hardware integration).

Limited to four lanes in the current implementation.

Future Improvements
Real-Time Camera Integration: Process live video feeds instead of static images.

Vehicle Speed Estimation: Detect vehicle speed to optimize signal timing.

Pedestrian Detection: Include pedestrian crossings in decision-making.

Historical Data Analysis: Use past traffic patterns to predict future congestion.

Multi-Intersection Support: Extend the system to manage multiple intersections.

License
MIT License

Contributors: Faizanur Rahman
              Rohan Srivastava
              Faizan Khan
              Mohammad Kaab
Repository: https://github.com/Codeaizan/Ai-Based-Traffic-Management
