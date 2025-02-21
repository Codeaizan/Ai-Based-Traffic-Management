import torch
import os
import time
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from PIL import Image
import torchvision.ops as ops

# GPU configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

# Load model with optimized parameters
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(
    weights=weights,
    box_score_thresh=0.85,
    box_nms_thresh=0.2,
    min_size=640
).to(device)
model.eval()

# Vehicle classes (COCO dataset)
VEHICLE_CLASSES = {3: 'car', 6: 'bus', 8: 'truck'}

class TrafficScheduler:
    def __init__(self, lanes):
        self.lanes = lanes
        self.priority_weights = [1.0] * len(lanes)
        self.last_green = [0] * len(lanes)
        self.current_green = None
        self.consecutive_cycles = 0
        self.params = {
            'max_consecutive': 2,
            'emergency_threshold': 15,
            'time_decay': 0.9  # Per minute decay
        }

    def calculate_priority(self, counts):
        priorities = []
        current_time = time.time()
        
        for i, count in enumerate(counts):
            time_since_green = (current_time - self.last_green[i]) / 60
            decay = self.params['time_decay'] ** time_since_green
            emergency = 10 if count > self.params['emergency_threshold'] else 0
            priorities.append(count * self.priority_weights[i] * decay + emergency)
            
        return priorities

    def update_weights(self, selected):
        for i in range(len(self.priority_weights)):
            if i == selected:
                self.priority_weights[i] = max(0.7, self.priority_weights[i] * 0.8)
            else:
                self.priority_weights[i] = min(1.3, self.priority_weights[i] * 1.1)

    def get_green_lane(self, counts):
        priorities = self.calculate_priority(counts)
        
        # Emergency override
        emergency = np.argwhere(np.array(counts) > self.params['emergency_threshold']).flatten()
        if len(emergency) > 0 and self.current_green not in emergency:
            return emergency[0]
        
        # Force rotation after max consecutive
        if self.consecutive_cycles >= self.params['max_consecutive']:
            return (self.current_green + 1) % len(self.lanes)
            
        return np.argmax(priorities)

    def decide_signal(self, counts):
        new_green = self.get_green_lane(counts)
        
        if new_green != self.current_green:
            self.update_weights(new_green)
            self.last_green[new_green] = time.time()
            self.consecutive_cycles = 0
            self.current_green = new_green
        else:
            self.consecutive_cycles += 1
            
        return new_green

def count_vehicles(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    with torch.cuda.amp.autocast(), torch.no_grad():
        img = Image.open(image_path).convert("RGB")
        transform = weights.transforms()
        img_tensor = transform(img).unsqueeze(0).to(device)
        predictions = model(img_tensor)[0]
        
        keep = ops.nms(predictions['boxes'], predictions['scores'], 0.2)
        predictions = {k: v[keep] for k, v in predictions.items()}

        return sum(1 for label, score in zip(predictions["labels"], predictions["scores"])
                if label.item() in VEHICLE_CLASSES and score.item() > 0.85)

if __name__ == "__main__":
    # Warm-up GPU
    if torch.cuda.is_available():
        dummy = torch.randn(1, 3, 640, 640, device=device)
        _ = model(dummy)
        torch.cuda.empty_cache()

    lanes = [
        r"D:\Traffic\images\image1.jpg",
        r"D:\Traffic\images\image2.jpg",
        r"D:\Traffic\images\image3.jpg",
        r"D:\Traffic\images\image4.jpg"
    ]
    
    scheduler = TrafficScheduler(lanes)
    
    try:
        while True:
            counts = [count_vehicles(path) for path in lanes]
            green_lane = scheduler.decide_signal(counts)
            
            print("\n" + "="*40)
            print(f"Current Cycle: {time.strftime('%H:%M:%S')}")
            for i, (path, count) in enumerate(zip(lanes, counts)):
                status = "GREEN" if i == green_lane else "RED"
                print(f"Lane {i+1} ({os.path.basename(path)}):")
                print(f"  Vehicles: {count}")
                print(f"  Status: {status}")
                print(f"  Priority Weight: {scheduler.priority_weights[i]:.2f}")
            
            print("\nDecision Logic:")
            print(f"- Consecutive greens: {scheduler.consecutive_cycles}")
            print(f"- Emergency override: {'Yes' if any(c > 15 for c in counts) else 'No'}")
            
            time.sleep(5)  # Simulate 5-second cycle
    
    except KeyboardInterrupt:
        print("\nTraffic control system stopped.")