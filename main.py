"""
main.py
-------
Smart Waste Bin — Updated Hardware Configuration
Sensors : Ultrasonic (HC-SR04)
Actuator: Single tilt servo (wet=right/135°, dry=left/45°, neutral=90°)

FLOW:
  1. Ultrasonic detects person within range
  2. Camera captures image
  3. MobileNetV2 classifies → wet or dry
  4. Servo tilts to correct side
  5. After 3 seconds → servo returns to neutral

Run on Pi:  python main.py
Simulate:   python main.py --simulate
"""

import time
import argparse

# ─── Argument parsing ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--simulate",   action="store_true", help="Run without real GPIO hardware")
parser.add_argument("--model_path", default="models/waste_classifier.pt")
parser.add_argument("--camera",     type=int,   default=0)
parser.add_argument("--threshold",  type=float, default=0.6, help="Min confidence to act")
args = parser.parse_args()

# ─── GPIO / Hardware setup ────────────────────────────────────────────────────
if not args.simulate:
    try:
        import RPi.GPIO as GPIO
        import spidev

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        # ── Pin definitions ───────────────────────────────────────────────────
        TRIG_PIN       = 23     # Ultrasonic TRIG
        ECHO_PIN       = 24     # Ultrasonic ECHO
        SERVO_PIN      = 17     # Tilt servo signal pin

        # ── GPIO setup ────────────────────────────────────────────────────────
        GPIO.setup(TRIG_PIN,  GPIO.OUT)
        GPIO.setup(ECHO_PIN,  GPIO.IN)
        GPIO.setup(SERVO_PIN, GPIO.OUT)

        # ── Servo PWM ─────────────────────────────────────────────────────────
        tilt_servo = GPIO.PWM(SERVO_PIN, 50)   # 50Hz PWM
        tilt_servo.start(0)

        # ── SPI for MQ sensor (via MCP3008 ADC) ───────────────────────────────
        spi = spidev.SpiDev()
        spi.open(0, 0)
        spi.max_speed_hz = 1350000

        def read_adc(channel):
            """Read analog value from MCP3008 ADC (0–1023)."""
            r = spi.xfer2([1, (8 + channel) << 4, 0])
            return ((r[1] & 3) << 8) + r[2]

        def read_ultrasonic():
            """Returns distance in cm."""
            GPIO.output(TRIG_PIN, True)
            time.sleep(0.00001)
            GPIO.output(TRIG_PIN, False)
            start = stop = time.time()
            while GPIO.input(ECHO_PIN) == 0:
                start = time.time()
            while GPIO.input(ECHO_PIN) == 1:
                stop = time.time()
            return (stop - start) * 34300 / 2

        def set_servo_angle(angle):
            """Move tilt servo to given angle (0–180°)."""
            duty = 2 + (angle / 18)
            tilt_servo.ChangeDutyCycle(duty)
            time.sleep(0.6)
            tilt_servo.ChangeDutyCycle(0)   # Stop jitter

    except ImportError:
        print("[Error] RPi.GPIO not found. Run with --simulate on non-Pi machines.")
        exit(1)

else:
    # ── Simulation stubs ──────────────────────────────────────────────────────
    import random

    def read_ultrasonic():
        dist = random.uniform(5, 35)
        print(f"    [SIM] Ultrasonic → {dist:.1f} cm")
        return dist

    def read_adc(channel):
        val = random.randint(200, 900)
        return val

    def set_servo_angle(angle):
        direction = "NEUTRAL"
        if angle < 90:
            direction = "LEFT  (DRY side)"
        elif angle > 90:
            direction = "RIGHT (WET side)"
        print(f"    [SIM] Tilt servo → {angle}° — {direction}")

# ─── Servo angle constants ────────────────────────────────────────────────────
SERVO_NEUTRAL = 90    # Flat / waiting position
SERVO_DRY     = 45    # Tilt left  → dry compartment
SERVO_WET     = 135   # Tilt right → wet compartment

# ─── Thresholds ───────────────────────────────────────────────────────────────
PERSON_THRESHOLD_CM = 20    # Trigger if person within 20 cm
HOLD_SECONDS        = 3     # How long servo stays tilted before returning

# ─── Load classifier ──────────────────────────────────────────────────────────
from waste_classifier import WasteClassifier
import cv2
import numpy as np

print("[System] Loading ML model ...")
classifier = WasteClassifier(model_path=args.model_path)
print("[System] Model ready.\n")

# ─── Helper functions ─────────────────────────────────────────────────────────
def capture_frame():
    """Capture a single frame from the camera."""
    cap = cv2.VideoCapture(args.camera)
    time.sleep(0.3)   # Let camera warm up
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Camera capture failed")
    return frame

def tilt_for_waste(label):
    """Tilt servo based on classification result, then return to neutral."""
    if "wet" in label.lower():
        print("  → Tilting RIGHT for WET waste (135°)")
        set_servo_angle(SERVO_WET)
    else:
        print("  → Tilting LEFT for DRY waste (45°)")
        set_servo_angle(SERVO_DRY)

    print(f"  → Holding for {HOLD_SECONDS} seconds ...")
    time.sleep(HOLD_SECONDS)

    print("  → Returning to NEUTRAL (90°)")
    set_servo_angle(SERVO_NEUTRAL)

def run_interactive_simulation():
    """Run interactive simulation with live webcam feed and visual indicators."""
    import math
    import cv2
    import numpy as np
    import random

    print("\n" + "="*55)
    print("  SMART WASTE BIN INTERACTIVE SIMULATOR (WEBCAM)")
    print("="*55)
    print("Instructions:")
    print("  - Hold up waste item to the webcam.")
    print("  - Press [SPACE] to capture & classify (Manual mode).")
    print("  - Press [D] to capture & save frame as DRY waste training image.")
    print("  - Press [W] to capture & save frame as WET waste training image.")
    print("  - Press [C] to toggle Continuous classification.")
    print("  - Press [Q] or [ESC] to Quit.")
    print("="*55 + "\n")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[Error] Could not open webcam at index {args.camera}.")
        print("Please check your camera connection or change index using --camera <index>.")
        return

    # Warm up camera
    time.sleep(0.5)

    servo_angle = SERVO_NEUTRAL
    servo_state = "NEUTRAL"
    tilt_timer = None
    continuous_mode = False
    
    last_label = None
    last_confidence = 0.0
    last_dry_prob = 0.0
    last_wet_prob = 0.0
    last_inference_ms = 0.0
    low_confidence_triggered = False
    
    dataset_saved_text = None
    dataset_saved_color = (0, 255, 0)
    dataset_saved_timer = 0.0

    classification_cooldown = 0  # To prevent classifying too rapidly in continuous mode

    # Create window and bring it to front
    cv2.namedWindow("Smart Waste Bin Simulation", cv2.WINDOW_AUTOSIZE)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Failed to read frame from webcam.")
            time.sleep(0.1)
            continue

        # Flip horizontally for natural mirror behavior
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        display_frame = frame.copy()

        # ─── 1. Handle Servo Tilt Timing ─────────────────────────────────────
        if tilt_timer is not None:
            elapsed = time.time() - tilt_timer
            if elapsed >= HOLD_SECONDS:
                # Return to neutral
                servo_angle = SERVO_NEUTRAL
                servo_state = "NEUTRAL"
                tilt_timer = None
                low_confidence_triggered = False

        # ─── 2. Continuous Classification Mode ──────────────────────────────
        if continuous_mode and tilt_timer is None:
            classification_cooldown += 1
            # Classify every 15 frames (~0.5s) to ensure smooth UI
            if classification_cooldown >= 15:
                classification_cooldown = 0
                try:
                    result = classifier.classify_image(frame)
                    last_label = result["label"]
                    last_confidence = result["confidence"]
                    last_dry_prob = result["dry_prob"]
                    last_wet_prob = result["wet_prob"]
                    last_inference_ms = result["inference_ms"]

                    if last_confidence >= args.threshold:
                        tilt_timer = time.time()
                        if "wet" in last_label.lower():
                            servo_angle = SERVO_WET
                            servo_state = "WET"
                        else:
                            servo_angle = SERVO_DRY
                            servo_state = "DRY"
                    else:
                        low_confidence_triggered = False
                except Exception as e:
                    print(f"[Simulation Error] Classification failed: {e}")

        # ─── 4. Drawing Overlays (Premium Aesthetic) ─────────────────────────
        
        # A. Top Header Opaque Bar
        cv2.rectangle(display_frame, (0, 0), (w, 60), (30, 30, 30), -1)
        cv2.putText(display_frame, "SMART WASTE BIN SIMULATOR", (15, 38), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

        # B. Reticle (Target Area in Center)
        # We draw nice corner brackets in green
        rw, rh = 220, 220
        rx = (w - rw) // 2
        ry = (h - rh) // 2 + 10
        color_reticle = (0, 255, 0) if tilt_timer is None else (100, 100, 100)
        # Draw corners
        d = 20
        # Top-Left
        cv2.line(display_frame, (rx, ry), (rx + d, ry), color_reticle, 2)
        cv2.line(display_frame, (rx, ry), (rx, ry + d), color_reticle, 2)
        # Top-Right
        cv2.line(display_frame, (rx + rw, ry), (rx + rw - d, ry), color_reticle, 2)
        cv2.line(display_frame, (rx + rw, ry), (rx + rw, ry + d), color_reticle, 2)
        # Bottom-Left
        cv2.line(display_frame, (rx, ry + rh), (rx + d, ry + rh), color_reticle, 2)
        cv2.line(display_frame, (rx, ry + rh), (rx, ry + rh - d), color_reticle, 2)
        # Bottom-Right
        cv2.line(display_frame, (rx + rw, ry + rh), (rx + rw - d, ry + rh), color_reticle, 2)
        cv2.line(display_frame, (rx + rw, ry + rh), (rx + rw, ry + rh - d), color_reticle, 2)

        # Reticle helper text
        if tilt_timer is None:
            cv2.putText(display_frame, "PLACE WASTE HERE", (rx + 25, ry + rh // 2 + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # C. Servo Dial Gauge (Bottom Right)
        dial_center = (w - 75, h - 70)
        dial_r = 45
        cv2.ellipse(display_frame, dial_center, (dial_r, dial_r), 0, 180, 360, (100, 100, 100), 2)
        cv2.putText(display_frame, "WET", (dial_center[0] - dial_r - 10, dial_center[1] + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(display_frame, "DRY", (dial_center[0] + dial_r - 10, dial_center[1] + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(display_frame, "SERVO DIAL", (dial_center[0] - 40, dial_center[1] + 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Draw needle
        rad = math.radians(servo_angle)
        needle_len = dial_r - 5
        ndx = int(math.cos(rad) * needle_len)
        ndy = int(-math.sin(rad) * needle_len)
        
        # Color of needle based on state
        needle_color = (255, 255, 0)
        if servo_state == "DRY":
            needle_color = (0, 255, 0)
        elif servo_state == "WET":
            needle_color = (0, 0, 255)
            
        cv2.line(display_frame, dial_center, (dial_center[0] + ndx, dial_center[1] + ndy), needle_color, 3)
        cv2.circle(display_frame, dial_center, 5, (255, 255, 255), -1)

        # D. Current State Text panel (Left Side Info)
        mode_str = "MODE: CONTINUOUS [C]" if continuous_mode else "MODE: MANUAL [SPACE]"
        mode_color = (0, 255, 255) if continuous_mode else (255, 200, 0)
        cv2.putText(display_frame, mode_str, (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)

        # Servo state text overlay
        servo_color = (255, 255, 255)
        if servo_state == "DRY":
            servo_color = (0, 255, 0)
        elif servo_state == "WET":
            servo_color = (0, 140, 255)
        elif servo_state == "LOW CONFIDENCE":
            servo_color = (0, 0, 255)
            
        cv2.putText(display_frame, f"SERVO: {servo_state} ({servo_angle} deg)", (15, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, servo_color, 1)

        # F. Classification Result Display (Bottom Center Overlay)
        if last_label is not None:
            res_box_w, res_box_h = 360, 95
            res_x = (w - res_box_w) // 2
            res_y = h - res_box_h - 15

            # Semi-transparent box background
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (res_x, res_y), (res_x + res_box_w, res_y + res_box_h), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)

            # Determine colors and text based on results
            is_wet = "wet" in last_label.lower()
            tag_color = (0, 0, 255) if is_wet else (0, 255, 0)
            tag_text = "WET WASTE" if is_wet else "DRY WASTE"
            
            cv2.rectangle(display_frame, (res_x, res_y), (res_x + res_box_w, res_y + res_box_h), tag_color, 2)
            
            cv2.putText(display_frame, f"CLASSIFIED: {tag_text}", (res_x + 15, res_y + 25), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, tag_color, 2)
            cv2.putText(display_frame, f"CONFIDENCE: {last_confidence*100:.1f}%", (res_x + 15, res_y + 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Dry prob: {last_dry_prob*100:.1f}% | Wet prob: {last_wet_prob*100:.1f}%", 
                        (res_x + 15, res_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
            cv2.putText(display_frame, f"{last_inference_ms}ms", (res_x + res_box_w - 65, res_y + 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
            
            if low_confidence_triggered:
                cv2.putText(display_frame, "LOW CONFIDENCE (IGNORED)", (res_x + 15, res_y + 85), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        # G. Dataset Capture Feedback Banner
        if dataset_saved_text is not None:
            if time.time() - dataset_saved_timer < 1.2:
                box_w, box_h = 320, 35
                bx = (w - box_w) // 2
                by = 75
                cv2.rectangle(display_frame, (bx, by), (bx + box_w, by + box_h), (25, 25, 25), -1)
                cv2.rectangle(display_frame, (bx, by), (bx + box_w, by + box_h), dataset_saved_color, 1)
                cv2.putText(display_frame, dataset_saved_text, (bx + 15, by + 23), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, dataset_saved_color, 1)
            else:
                dataset_saved_text = None

        # Check if window was closed via [X]
        try:
            if cv2.getWindowProperty("Smart Waste Bin Simulation", cv2.WND_PROP_VISIBLE) < 1:
                break
        except Exception:
            pass

        # ─── 5. Display Window ────────────────────────────────────────────────
        cv2.imshow("Smart Waste Bin Simulation", display_frame)

        # ─── 6. Keyboard input handler ────────────────────────────────────────
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('c') or key == ord('C'):
            continuous_mode = not continuous_mode
            last_label = None
            print(f"[Simulation] Continuous mode: {'ON' if continuous_mode else 'OFF'}")
        elif (key == ord('d') or key == ord('D') or key == ord('w') or key == ord('W')):
            import os
            is_wet = (key == ord('w') or key == ord('W'))
            label_folder = "wet waste" if is_wet else "dry waste"
            
            # Decide whether to save to train or val (80/20 split)
            is_val = (random.random() < 0.2)
            subset = "val" if is_val else "train"
            
            # Target path
            folder_path = os.path.join(subset, label_folder)
            os.makedirs(folder_path, exist_ok=True)
            
            # Filename based on timestamp
            img_name = f"webcam_{subset}_{int(time.time() * 1000)}.jpg"
            img_path = os.path.join(folder_path, img_name)
            
            # Save original camera frame (raw frame without overlays)
            cv2.imwrite(img_path, frame)
            
            print(f"[Dataset] Captured and saved to {img_path}")
            
            # Show visual confirmation on screen
            dataset_saved_text = f"SAVED TO {subset.upper()} ({label_folder.upper()})"
            dataset_saved_color = (0, 0, 255) if is_wet else (0, 255, 0)
            dataset_saved_timer = time.time()
        elif key == ord(' ') and tilt_timer is None and not continuous_mode:
            print("\n[Simulation] Spacebar pressed. Capturing and classifying...")
            try:
                result = classifier.classify_image(frame)
                last_label = result["label"]
                last_confidence = result["confidence"]
                last_dry_prob = result["dry_prob"]
                last_wet_prob = result["wet_prob"]
                last_inference_ms = result["inference_ms"]

                print(f"  Classification: {last_label.upper()}")
                print(f"  Confidence    : {last_confidence*100:.1f}%")
                print(f"  Inference Time: {last_inference_ms} ms")

                tilt_timer = time.time()
                if last_confidence >= args.threshold:
                    if "wet" in last_label.lower():
                        servo_angle = SERVO_WET
                        servo_state = "WET"
                        print("  → Tilting RIGHT for WET waste (135°)")
                    else:
                        servo_angle = SERVO_DRY
                        servo_state = "DRY"
                        print("  → Tilting LEFT for DRY waste (45°)")
                else:
                    print(f"  → Low confidence ({last_confidence*100:.1f}% < {args.threshold*100:.1f}%). Servo does not move.")
                    low_confidence_triggered = True
                    servo_angle = SERVO_NEUTRAL
                    servo_state = "NEUTRAL"
            except Exception as e:
                print(f"[Simulation Error] Classification failed: {e}")

    cap.release()
    cv2.destroyAllWindows()
    print("[Simulation] Ended. Window closed.")

# ─── Servo starts at neutral ──────────────────────────────────────────────────
if not args.simulate:
    print("[System] Setting servo to neutral position ...")
    set_servo_angle(SERVO_NEUTRAL)

# ─── Main loop ────────────────────────────────────────────────────────────────
if args.simulate:
    try:
        run_interactive_simulation()
    except KeyboardInterrupt:
        print("\n[System] Shutting down simulator ...")
        print("[System] Goodbye.")
else:
    print("[System] Smart Waste Bin running. Press Ctrl+C to stop.\n")
    try:
        while True:
            # ── 2. Read ultrasonic ────────────────────────────────────────────────
            distance_cm = read_ultrasonic()

            if distance_cm < PERSON_THRESHOLD_CM:
                print(f"\n[Ultrasonic] Person detected at {distance_cm:.1f} cm")
                print("[Camera] Capturing image ...")

                # ── 3. Capture and classify ───────────────────────────────────────
                try:
                    frame = capture_frame()
                    result = classifier.classify_image(frame)

                    print(f"[ML] Classification: {result['label'].upper()}")
                    print(f"     Confidence : {result['confidence']*100:.1f}%")
                    print(f"     Dry prob   : {result['dry_prob']*100:.1f}%")
                    print(f"     Wet prob   : {result['wet_prob']*100:.1f}%")
                    print(f"     Inference  : {result['inference_ms']} ms")

                except Exception as e:
                    print(f"[Error] Classification failed: {e}")
                    time.sleep(1)
                    continue

                # ── 4. Actuate servo ──────────────────────────────────────────────
                if result["confidence"] >= args.threshold:
                    tilt_for_waste(result["label"])
                else:
                    print(f"[ML] Low confidence ({result['confidence']*100:.1f}%) — servo not moved")

                print("[Done] Cycle complete.\n")
                time.sleep(2)   # Debounce before next detection

            time.sleep(0.1)     # Polling interval

    except KeyboardInterrupt:
        print("\n[System] Shutting down ...")
        set_servo_angle(SERVO_NEUTRAL)
        tilt_servo.stop()
        spi.close()
        GPIO.cleanup()
        print("[System] Goodbye.")