# ArduinoGPT
Repository for controlling Arduino microcontrollers with ChatGPT

In the `tests` folder, I have test scripts playing around with various features like the OpenAI API, voice commands, serial communication, etc. It's unorganized for now

## Installation
ALl the required libraries can be installed by navigating to this folder directory, and running the following command in a terminal:
```
pip install -r requirements.txt
```

## Test Scripts
`uno-led-test.py` - This program is a speech recognition and text-to-speech program that utilizes the OpenAI API to send commands to a microcontroller that controls it's built-in LED.

<figure>
  <img src="https://github.com/Jshulgach/ArduinoGPT/blob/main/media/uno-chatgpt-voice-demo.gif" alt="uno-chatgpt" width="800" height="400"><br>
</figure>

To run this program, you must have an OpenAI API key.
See the instructions from [OpenAI](https://platform.openai.com/docs/quickstart?context=python) for how to set up your API key.

You should also have the [Arduino IDE](https://www.arduino.cc/en/software) editor installed which should take care of AVR board driver installation. 

Check the port assigned to your Arduino board plugged into the computer. For Windows OS, the port will look something like 'COM#' and an assigned number (e.g. 'COM6'). For Linux systems, USB devices will be displayed with '/dev/ttyACM#' (e.g. /dev/ttyACM0).

Navigate to the `tests` folder and run the example below. Remember to replace the port with your board's port:
```
cd tests
python uno-led-test.py --port YOURCOMPORT
```



