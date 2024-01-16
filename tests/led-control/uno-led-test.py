####  Copyright 2024 Jonathan Shulgach jonathan@shulgach.com


#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.

import serial, time, re, sys
import argparse
import speech_recognition as sr
from typing import Any
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import StringPromptTemplate

template = """
You are an LLM that is acting as a translator between human input and an Arduino Uno microcontroller.
You will be given a list of commands and arguments that each command can take. 
You will decide based on the human input which command and arguments to pass to the output and format them appropriately.
You will operate the LED on the UNO board.

Here is a list of the commands you can send:
[
"wait" Waits for the following argument to this command that can be passed as a float number between 0 and 1000.
"on" Turns on the LED
"off" Turns off the LED
]

If you do not find a suitable command, then respond with "unable"

To format the command string, place a "<" symbol before the command, then the command, then a comma "," and then any arguments as a comma separated list.  Finally include the ">" symbol.  

Here are some examples of formatted command strings:
To turn the led on: <on>
To turn the led off: <off>

Be sure to include only formatted command strings in your response
If you need to send more than one command then each command must be fully formatted in its own set of < and >.
For example to turn on the lights, wait 2 seconds, and then turn them off:  <on><wait,2><off>


Begin!

Human Input:
{input}

"""


class ArduinoGPTPromptTemplate(StringPromptTemplate):
    template: str
    device: Any

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)


class ArduinoGPT:
    """This class is a wrapper around the OpenAI GPT-3.5 model that is used to translate human input into Arduino
    commands """

    def __init__(self, device, use_voice=False):

        self.use_voice = use_voice
        self.human_input = None
        self.ai_output = None
        self.exit_flag = False
        self.device = device
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0
        )

        prompt = ArduinoGPTPromptTemplate(
            input_variables=["input"],
            template=template,
            device=self.device,
        )

        self.llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

        return

    def get_user_input(self):
        print("\n*******************************\n\nHuman:  ")
        if self.use_voice:
            self.human_input = self.get_voice_input()
        else:
            self.human_input = input("")
        if self.human_input == "exit":
            self.exit_flag = True
        return

    def get_voice_input(self):
        """
        Listen for audio input and convert it to text
        """
        # Create speech recognizer object
        r = sr.Recognizer()
        # Use default microphone as input source
        with sr.Microphone() as source:
            # Clear any background noise
            r.adjust_for_ambient_noise(source)
            # Listen for user input
            print("Listening...")
            audio = r.listen(source)
            try:
                # Convert audio to text
                text = r.recognize_google(audio)
                print("You said: " + text)
                return text
            except sr.UnknownValueError:
                print("Could not understand audio")
                return None

    def print_response(self):

        if self.ai_output is not None:
            print("\n*******************************\n\nAI:  ")
            if 'unable' not in self.ai_output:
                #print(self.ai_output)
                self.ai_output = self.ai_output.split('><')
                # The self.ai_output property will have something for example like "<on><wait,10><off>".

                print(self.ai_output)
                for i in range(len(self.ai_output)):
                    self.ai_output[i] = '<' + self.ai_output[i].replace('>','').replace('<','') + '>'
                    # Then send each command to the Arduino

                    # This is where the actual Arduino commands are sent. No responses are expected yet
                    self.device.write(bytes(self.ai_output[i] + "\n", 'utf-8'))

                    # Search for the pattern in the input string
                    match = re.search(r'<wait,(\d+)>', self.ai_output[i])
                    if match:
                        # If the pattern is found, then wait for the specified amount of time
                        time.sleep(float(match.group(1)))

        return

    def get_response(self):
        if self.human_input is not None:
            self.ai_output = self.llm_chain.run(self.human_input)
        return

    def run(self):
        """This is the main loop that runs the program"""
        self.get_user_input()
        while not self.exit_flag:
            self.get_response()
            self.print_response()
            self.get_user_input()
        return


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=str, required=True, help='Port to establish serial communication ')
    args = parser.parse_args()

    # Open serial port to Arduino
    arduino = serial.Serial(port=args.port, baudrate=9600, timeout=0.1)

    ard = ArduinoGPT(arduino, use_voice=True)
    ard.run()


if __name__ == "__main__":
    __main__()
