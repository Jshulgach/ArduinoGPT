// Define the LED pin
const int ledPin = 13;

void setup() {
  // Initialize serial communication at 9600 baud
  Serial.begin(9600);
  
  // Set the LED pin as an OUTPUT
  pinMode(ledPin, OUTPUT);
  
  // Initially turn off the LED
  digitalWrite(ledPin, LOW);
}

void loop() {
  if (Serial.available() > 0) {
    // Read the incoming command
    String command = Serial.readStringUntil('\n');
    
    // Remove leading and trailing whitespace
    command.trim();
    
    // Check if the command is enclosed in <>
    if (command.startsWith("<") && command.endsWith(">")) {
      // Remove the <>
      command = command.substring(1, command.length() - 1);
      
      // Check if the command contains a comma
      int commaIndex = command.indexOf(",");
      
      if (commaIndex != -1) {
        // Extract the command and argument
        String cmd = command.substring(0, commaIndex);
        String arg = command.substring(commaIndex + 1);
        
        // Check the received command and perform actions accordingly
        if (cmd == "on") {
          digitalWrite(ledPin, HIGH); // Turn the LED on
          Serial.println("LED is ON");
        } else if (cmd == "off") {
          digitalWrite(ledPin, LOW); // Turn the LED off
          Serial.println("LED is OFF");
        } else if (cmd == "wait") {
          int duration = arg.toInt();
          if (duration > 0) {
            delay(duration); // Wait for the specified duration (in milliseconds)
            Serial.println("Waited for " + String(duration) + " milliseconds");
          } else {
            Serial.println("Invalid wait duration");
          }
        } else {
          Serial.println("Invalid command"); // Unknown command
        }
        
        // Print the argument
        Serial.println("Argument: " + arg);
      } else {
        // No comma found, treat the entire command as-is
        // Check the received command and perform actions accordingly
        if (command == "on") {
          digitalWrite(ledPin, HIGH); // Turn the LED on
          Serial.println("LED is ON");
        } else if (command == "off") {
          digitalWrite(ledPin, LOW); // Turn the LED off
          Serial.println("LED is OFF");
        } else {
          Serial.println("Invalid command"); // Unknown command
        }
      }
    } else {
      Serial.println("Invalid command format"); // Command not enclosed in <>
    }
  }
}
