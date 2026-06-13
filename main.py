"""
main.py
-------
Smart Waste Bin — Jetson Nano serial interface to ESP32
Sensors  : Ultrasonic (HC-SR04) connected to ESP32
           MQ-135 Gas Sensor connected to ESP32 (Analog)
Actuator : Tilt servo connected to ESP32

FLOW:
  1. ESP32 monitors ultrasonic distance.
  2. ESP32 constantly sends: "DIST:<cm>,GAS:<val>" over Serial USB.
  3. main.py on Jetson Nano reads Serial data.
  4. If a person is detected (distance < 20cm), Jetson Nano captures camera frame.
  5. Classifier runs on Jetson GPU to predict "wet" or "dry" waste.
  6. Jetson sends "TILT:WET" or "TILT:DRY" command over Serial.
  7. ESP32 tilts the plate, waits 3 seconds, and returns to neutral.
"""

import time
import argparse
import cv2
import numpy as np
import serial
from waste_classifier import WasteClassifier

# ─── Argument parsing ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Smart Waste Bin Controller — ESP32 Serial Interface")
parser.add_argument("--model_path", default="models/waste_classifier.pt", help="Path to PyTorch model")
parser.add_argument("--camera",     type=int,   default=0,                  help="Camera index (or sensor ID for CSI camera)")
parser.add_argument("--csi",        action="store_true",                    help="Use Jetson CSI camera via GStreamer")
parser.add_argument("--threshold",  type=float, default=0.6,                help="Min confidence threshold to act")
parser.add_argument("--port",       default="/dev/ttyUSB0",                 help="Serial port of the ESP32 (e.g. /dev/ttyUSB0 or COM3)")
parser.add_argument("--baud",       type=int,   default=115200,             help="Serial baud rate")
args = parser.parse_args()

PERSON_THRESHOLD_CM = 20    # Trigger range
DEBOUNCE_WAIT_SEC   = 5     # Cool-down wait after a classification cycle

# ─── Camera Helpers ───────────────────────────────────────────────────────────
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=224,
    display_height=224,
    framerate=30,
    flip_method=0,
):
    """GStreamer pipeline for Jetson Nano CSI camera."""
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def capture_frame():
    """Capture a single frame from the camera."""
    if args.csi:
        pipeline = gstreamer_pipeline(sensor_id=args.camera)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(args.camera)
        
    time.sleep(0.3)   # Let camera auto-exposure warm up
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Camera capture failed")
    return frame

# ─── Serial Helpers ───────────────────────────────────────────────────────────
def read_serial_data(ser):
    """
    Reads a line from serial and parses distance and gas level.
    Expected format: "DIST:<float>,GAS:<int>"
    Returns: (distance_cm, gas_level)
    """
    if ser is None or not ser.is_open:
        return None, None
        
    try:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            parts = line.split(',')
            dist_val = None
            gas_val = None
            for part in parts:
                if part.startswith("DIST:"):
                    dist_val = float(part.split(':')[1])
                elif part.startswith("GAS:"):
                    gas_val = int(part.split(':')[1])
            if dist_val is not None and gas_val is not None:
                return dist_val, gas_val
    except Exception:
        pass  # Quietly catch corrupt or incomplete lines
    return None, None

def send_tilt_command(ser, label):
    """Sends command to ESP32 to tilt the bin plate."""
    if ser is None or not ser.is_open:
        print(f"[Demo] Sending tilt command for {label.upper()}")
        return
        
    cmd = f"TILT:{'WET' if 'wet' in label.lower() else 'DRY'}\n"
    try:
        ser.write(cmd.encode('utf-8'))
        ser.flush()
        print(f"[Serial] Sent command to ESP32: {cmd.strip()}")
    except Exception as e:
        print(f"[Error] Failed to send command to ESP32: {e}")

# ─── Main Run Loop ────────────────────────────────────────────────────────────
print("[System] Loading ML model ...")
classifier = WasteClassifier(model_path=args.model_path)
print("[System] Model ready.\n")

print(f"[System] Connecting to ESP32 on port {args.port} at {args.baud} baud...")
try:
    ser = serial.Serial(args.port, args.baud, timeout=1.0)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    print("[System] Serial connection established.")
except Exception as e:
    print(f"\n[Warning] Failed to open serial port: {e}")
    print("[Warning] Running in demo mode without physical serial device.")
    ser = None

print("\n[System] Smart Waste Bin active. Press Ctrl+C to stop.\n")

distance_cm = 999.0
gas_level = 0
last_cycle_time = 0

try:
    while True:
        # Read status updates from ESP32
        dist_in, gas_in = read_serial_data(ser)
        if dist_in is not None:
            distance_cm = dist_in
        if gas_in is not None:
            gas_level = gas_in

        # If a person is close, trigger the classification cycle
        current_time = time.time()
        if distance_cm < PERSON_THRESHOLD_CM and (current_time - last_cycle_time) > DEBOUNCE_WAIT_SEC:
            print(f"\n[Ultrasonic] Person detected at {distance_cm:.1f} cm (Gas sensor level: {gas_level})")
            print("[Camera] Capturing waste image ...")
            
            try:
                # Capture and classify
                frame = capture_frame()
                result = classifier.classify_image(frame)
                
                label = result["label"]
                confidence = result["confidence"]
                
                print(f"[ML] Classification: {label.upper()}")
                print(f"     Confidence : {confidence*100:.1f}%")
                print(f"     Dry prob   : {result['dry_prob']*100:.1f}%")
                print(f"     Wet prob   : {result['wet_prob']*100:.1f}%")
                print(f"     Inference  : {result['inference_ms']} ms")
                
                # Actuate servo via ESP32 Serial
                if confidence >= args.threshold:
                    send_tilt_command(ser, label)
                else:
                    print(f"[ML] Low confidence ({confidence*100:.1f}%) — servo not moved")
                    
                # Set cool-down timer
                last_cycle_time = time.time()
                    
            except Exception as e:
                print(f"[Error] Cycle execution failed: {e}")
                
            print("[Done] Cycle complete. Waiting for debounce...\n")

        time.sleep(0.05)     # High-frequency polling

except KeyboardInterrupt:
    print("\n[System] Shutting down...")
    if ser and ser.is_open:
        # Tell ESP32 to return to neutral if it was tilting
        try:
            ser.write(b"TILT:NEUTRAL\n")
            ser.close()
        except Exception:
            pass
    print("[System] Goodbye.")