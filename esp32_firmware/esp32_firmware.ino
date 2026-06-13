/*
 * esp32_firmware.ino
 * -------------------
 * ESP32 Smart Waste Bin controller.
 * Handles:
 *   - HC-SR04 Ultrasonic Sensor (Trigger & Echo)
 *   - MQ135 Gas Sensor (Analog Pin)
 *   - 5V High-Torque Servo Motor (PWM Pin)
 *   - Serial communication with Jetson Nano (115200 Baud)
 * 
 * Dependencies:
 *   - ESP32Servo Library (Install via Arduino Library Manager)
 */

#include <ESP32Servo.h>

// ─── Pin Configurations ──────────────────────────────────────────────────────
#define TRIG_PIN     5    // Ultrasonic TRIG pin
#define ECHO_PIN    18    // Ultrasonic ECHO pin
#define MQ135_PIN   34    // MQ135 Analog Input pin (ADC1 Channel 6)
#define SERVO_PIN   13    // Servo PWM Output pin

// ─── Constants ───────────────────────────────────────────────────────────────
const int ANGLE_NEUTRAL = 90;
const int ANGLE_DRY     = 45;
const int ANGLE_WET     = 135;
const unsigned long HOLD_TIME_MS = 3000; // Hold tilt position for 3 seconds

// ─── State Variables ─────────────────────────────────────────────────────────
Servo tiltServo;
unsigned long lastSensorReadTime = 0;
const unsigned long SENSOR_INTERVAL_MS = 150; // Read/send sensor data every 150ms

enum BinState {
  STATE_NEUTRAL,
  STATE_TILTED
};
BinState currentState = STATE_NEUTRAL;
unsigned long tiltStartTime = 0;

// ─── Function Declarations ───────────────────────────────────────────────────
float readDistance();
int readGasSensor();
void handleSerialCommands();
void checkTiltTimer();

void setup() {
  Serial.begin(115200);
  
  // Configure pins
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(MQ135_PIN, INPUT);
  
  // Set up ESP32 PWM for the Servo
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);
  
  tiltServo.setPeriodHertz(50); // Standard 50Hz servo
  tiltServo.attach(SERVO_PIN, 500, 2400); // Attach servo with min/max pulse widths
  
  // Set to initial neutral flat position
  tiltServo.write(ANGLE_NEUTRAL);
  delay(500); // Allow servo to reach target
  
  // Detach/disable PWM to prevent initial jitter/humming
  tiltServo.write(0); 
}

void loop() {
  // Read and send sensor data periodically
  unsigned long currentMillis = millis();
  if (currentMillis - lastSensorReadTime >= SENSOR_INTERVAL_MS) {
    lastSensorReadTime = currentMillis;
    
    float distance = readDistance();
    int gasVal = readGasSensor();
    
    // Send telemetry to Jetson Nano: "DIST:<float>,GAS:<int>"
    Serial.print("DIST:");
    Serial.print(distance, 1);
    Serial.print(",GAS:");
    Serial.println(gasVal);
  }
  
  // Check for serial commands from Jetson Nano
  handleSerialCommands();
  
  // Handle non-blocking return-to-neutral timer
  checkTiltTimer();
  
  delay(10); // Small loop pacing
}

// ─── Sensor Readings ─────────────────────────────────────────────────────────
float readDistance() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  
  // Read echo pulse duration (timeout after 20ms = ~3.4 meters max range)
  long duration = pulseIn(ECHO_PIN, HIGH, 20000);
  if (duration == 0) {
    return 999.0; // Out of range or sensor timeout
  }
  
  // Calculate distance in cm
  float distanceCm = (duration * 0.0343) / 2.0;
  return distanceCm;
}

int readGasSensor() {
  // Read ADC (values 0 - 4095 on ESP32 12-bit ADC)
  return analogRead(MQ135_PIN);
}

// ─── Actuator Controls ───────────────────────────────────────────────────────
void handleSerialCommands() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    
    if (cmd.startsWith("TILT:")) {
      String commandType = cmd.substring(5);
      
      if (commandType == "WET") {
        tiltServo.write(ANGLE_WET);
        currentState = STATE_TILTED;
        tiltStartTime = millis();
      } 
      else if (commandType == "DRY") {
        tiltServo.write(ANGLE_DRY);
        currentState = STATE_TILTED;
        tiltStartTime = millis();
      } 
      else if (commandType == "NEUTRAL") {
        tiltServo.write(ANGLE_NEUTRAL);
        currentState = STATE_NEUTRAL;
        delay(400); // Allow time to return
        tiltServo.write(0); // Release to stop jitter
      }
    }
  }
}

void checkTiltTimer() {
  if (currentState == STATE_TILTED) {
    if (millis() - tiltStartTime >= HOLD_TIME_MS) {
      tiltServo.write(ANGLE_NEUTRAL);
      currentState = STATE_NEUTRAL;
      delay(600); // Wait for servo to return to center before turning off PWM
      tiltServo.write(0); // Detach to release holding torque & eliminate jitter
    }
  }
}
