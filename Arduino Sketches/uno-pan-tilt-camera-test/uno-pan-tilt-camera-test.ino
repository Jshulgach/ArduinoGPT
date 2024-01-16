#include <Servo.h>

// Servo objects for pan/tilt
Servo panServo;
Servo tiltServo;

// Define pins for servos
#define panServoPin 9
#define tiltServoPin 10

// Variables to track previous servo positions (do not edit, unless you want the 
// starting position to be somewhere other than faceforward)
int previousPanPos = 90;
int previousTiltPos = 180;

// CONFIGURABLE PARAMETERS
bool USE_DELTA_ANGLES = true; // Set to true to use delta angles, makes python script simpler

void setup() {
  Serial.begin(9600); // Serial communication at baudrate 9600

  // Attach pins 9, 10 to pan, tilt servos
  panServo.attach(panServoPin);
  tiltServo.attach(tiltServoPin);

  // Set servos to starting positions, facing forward
  panServo.write(previousPanPos);
  tiltServo.write(previousTiltPos);
  
}

void loop() {
  if (Serial.available() > 0) {
    // Read incoming string, either "<PAN,ANGLE>" or "<TILT,ANGLE>"
    String command = Serial.readStringUntil('\n');

    // Remove leading and trailing whitespace
    command.trim();

    // Check if command is enclosed in <>
    if (command.startsWith("<") && command.endsWith(">")) { 
      // Remove the <>
      command = command.substring(1, command.length() - 1);\

      // Parse command
      int commaIndex = command.indexOf(',');
      String servoCommand = command.substring(0, commaIndex);
      int angle = command.substring(commaIndex + 1).toInt();
      
      // Constrain angle just in case (but wont, because negative deltas!)
      //angle = constrain(angle, 0, 180);

      // Move the requested servo the requested angle
      if (servoCommand == "PAN") { 
        if (USE_DELTA_ANGLES) { 
          angle = previousPanPos + angle; // Use current angle as origin
          previousPanPos = angle;
        }
        //Serial.print("Setting pan angle: ");
        //Serial.println(angle);
        panServo.write(angle);
      } else if (servoCommand == "TILT") {
        if (USE_DELTA_ANGLES) { 
          angle = previousTiltPos + angle; // Use current angle as origin
          previousTiltPos = angle;
        } 
        //Serial.print("Setting tilt angle: ");
        //Serial.println(angle);
        tiltServo.write(angle);
      }
      
    }    
  }
}
