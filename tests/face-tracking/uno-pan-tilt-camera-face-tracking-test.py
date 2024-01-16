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

import cv2
import mediapipe as mp
import serial
import sys
import asyncio
import argparse
from simple_pid import PID
import speech_recognition as sr
from typing import Any
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import StringPromptTemplate

template = """
You are an LLM that is acting as a translator between human input and a tracking application where detected objects have a category associated with them. 
You will decide based on the human input which object to track and then output the corresponding label or integer to the tracking application. 
The output will be one word or number but it should also be contained in the human input.

Here are some examples of human inputs and interpretations:
2) If the human input is "track the cat", then you will output "cat"
2) If the human input is t"track person", you will output "person"
3) If the input is track person 1, you will output "1"
4) If you get the human input "reset position", you will output "reset"

If you are unsure how how to respond, then respond with "unable"

Begin!

Human Input:
{input}

"""


class ArduinoGPTPromptTemplate(StringPromptTemplate):
    template: str
    device: Any

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)


class ChatGPT(object):
    """This class is a wrapper around the OpenAI GPT-3.5 model that is used to translate human input into Arduino
    commands """

    def __init__(self, device, use_voice=False, verbose=False):
        self.use_voice = use_voice
        self.verbose = verbose
        self.human_input = None
        self.ai_output = None
        self.exit_flag = False
        self.device = device
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        prompt = ArduinoGPTPromptTemplate(input_variables=["input"], template=template, device=self.device)
        self.llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
        return

    def get_response(self, human_input=None):
        if human_input is not None:
            self.ai_output = self.llm_chain.run(human_input)
        return self.ai_output

    def handle_response(self, response):
        if response is not None:
            if self.verbose:
                print(f"AI Response: {response}")
            if response == "exit":
                self.exit_flag = True
            else:
                self.ai_output = response
        else:
            print("No response from AI")
        return self.ai_output


class HumanInput(object):
    """ A class that handles human inputs from voice or keyboard

        Parameters:
        -----------
        use_voice                   : (bool) If True, use voice commands to choose objects to detect
        use_openai                  : (bool) If True, use OpenAI GPT-3.5 to translate human input into commands
        verbose                     : (bool) If True, print debug messages to console

    """

    def __init__(self, use_voice=False, use_openai=False, verbose=False):
        self.use_voice = use_voice
        self.use_openai = use_openai
        self.verbose = verbose
        self.human_input = None
        self.ai_output = None

        if self.use_voice:
            self.r = sr.Recognizer()
            self.m = sr.Microphone()
            with self.m as source:
                self.r.adjust_for_ambient_noise(source)  # we only need to calibrate once, before we start listening
            self.stop_listening = self.r.listen_in_background(self.m, self.voice_input_cb)

        if self.use_openai:
            self.ai = ChatGPT(device=self, use_voice=self.use_voice, verbose=self.verbose)

    def voice_input_cb(self, recognizer, audio):
        try:
            self.human_input = recognizer.recognize_google(audio)
            print(f"You said: {self.human_input}")
            self.ai_output = self.ai.get_response(self.human_input)
            self.handle_response(self.ai_output)
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            self.human_input = None
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
            self.human_input = None
        return self.human_input

    def handle_response(self, response):
        if response is not None:
            print(f"AI Response: {response}")
            if response == "exit":
                self.exit_flag = True
            else:
                self.ai_output = response
        else:
            print("No response from AI")
        return self.ai_output


class Tracking(object):
    """ This is the tracking class that will be used to track objects using the pan/tilt mechanism

        Parameters:
        -----------
        name                        : (str) Name of the object being tracked
        use_voice                   : (bool) If True, use voice commands to choose objects to detect
        use_openai                  : (bool) If True, use OpenAI GPT-3.5 to translate human input into Arduino commands
        use_serial                  : (bool) If True, use serial communication to send commands to Arduino
        verbose                     : (bool) If True, print debug messages to console

    """

    def __init__(self, name=None, port='COM7', use_voice=False, use_openai=False, use_serial=False, use_face_detection=False, verbose=False):
        self.name = name
        self.use_voice = use_voice
        self.use_openai = use_openai
        self.use_serial = use_serial
        self.use_face_detection = use_face_detection
        self.verbose = verbose

        self.exit_flag = False
        self.reset_position = False
        self.pan = 0
        self.tilt = 0
        self.tracking_object = 0
        self.tracking_object_center = None
        self.image_center = None

        # Initialize the PID controller for the pan/tilt mechanism
        self.pan_pid = PID(0.02, 0.0005, 0.00001, setpoint=0)
        self.tilt_pid = PID(0.02, 0.0005, 0.00001, setpoint=0)

        # Initialize the camera attribute instance
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

        # Create the Object Detection object from mediapipe
        use_face = True
        if self.use_face_detection:
            model_path = 'face-tracking/models/detector.tflite'  # for face detection
            base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
            options = mp.tasks.vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
            self.detector = mp.tasks.vision.FaceDetector.create_from_options(options)

        else:
            model_path = 'face-tracking/models/ssd_mobilenet_v2.tflite'  # for object detection
            base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
            options = mp.tasks.vision.ObjectDetectorOptions(base_options=base_options, max_results=5,
                                                        score_threshold=0.2,
                                                        running_mode=mp.tasks.vision.RunningMode.IMAGE)
            self.detector = mp.tasks.vision.ObjectDetector.create_from_options(options)

        # create the HumanInput object
        self.inputs = HumanInput(use_voice=self.use_voice, use_openai=self.use_openai, verbose=self.verbose)

        if self.use_serial:
            # Open serial port to Arduino
            self.arduino = serial.Serial(port=port, baudrate=9600, timeout=0.1)

    def start(self):
        """ Start the tracking process """
        # Start the main loop
        asyncio.run(self.main())

    async def main(self):
        """ Start main tasks and coroutines in a single main function """
        # Start the tracking and serial communication task
        self.serial_task = asyncio.create_task(self.serial_comm())

        # Start the observing task
        self.tracking_task = asyncio.create_task(self.observe_environment(0.03))

        # Start the human input task
        self.human_input_task = asyncio.create_task(self.handle_human_input())

        # Wait for all tasks to complete (this will never happen)
        print('Starting tracking coroutines')
        await self.tracking_task
        await self.serial_task
        await self.human_input_task
        # await asyncio.sleep(0)

    async def observe_environment(self, rate=0.05):
        """ This function will track an object and send commands to the Arduino to move the pan/tilt mechanism """
        while not self.exit_flag:

            # Run detection from camera
            results, image = self.get_camera_image(alpha=1)

            self.tracking_object_center = None
            if results.detections:
                for i, detected_object in enumerate(results.detections):

                    # draw box around object
                    bbox = detected_object.bounding_box
                    bbox_start = (bbox.origin_x, bbox.origin_y)
                    bbox_end = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
                    cv2.rectangle(image, bbox_start, bbox_end, (255, 255, 255), 2)

                    # Display label_id over object bounding box
                    idx = i + 1
                    cv2.putText(image, f'{idx}:', (bbox.origin_x - 10, bbox.origin_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 2)

                    # Display the class name above the object bounding box
                    object_name = detected_object.categories[0].category_name
                    #print(f'Object {idx}: {object_name}')
                    cv2.putText(image, f'{object_name}',
                                (bbox.origin_x + 20, bbox.origin_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 2)

                    # Use the index if face tracking
                    if self.use_face_detection:
                        object_name = str(idx)
                    else:
                        object_name = str(object_name)

                    # Draw a red boundary box over the detected object that matches the name in self.tracking_object
                    if self.tracking_object is not None and object_name == str(self.tracking_object):
                        cv2.rectangle(image, bbox_start, bbox_end, (0, 0, 255), 2)

                        # Get the center of the detected object
                        self.tracking_object_center = (bbox.origin_x + bbox.width / 2, bbox.origin_y + bbox.height / 2)

                        # Draw a circle at the center of the image
                        cv2.circle(image, (int(self.tracking_object_center[0]), int(self.tracking_object_center[1])), 5,
                                   (0, 255, 0), -1)


                        # Draw a green rectangle over the center of the image
                        cv2.rectangle(image, (int(self.image_center[0] - 5), int(self.image_center[1] - 5)),
                                    (int(self.image_center[0] + 5), int(self.image_center[1] + 5)), (0, 255, 0), 2)

            # display image with all drawings at the end
            cv2.imshow('Object Detection', image)
            if cv2.waitKey(5) & 0xFF == 27:
                self.exit_flag = True
                break

            await asyncio.sleep(rate)

    def get_camera_image(self, alpha=3):
        """ Get an image from the camera and run object detection on it"""
        success, image = self.cap.read()
        if success:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            results = self.detector.detect(mp_image)  # Run object detection on the image

            # Adjust image brightness in case the room is too dark or bright
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)

            # get the image center point
            self.image_center = (image.shape[1] / 2, image.shape[0] / 2)
            return results, image

    async def handle_human_input(self, rate=1):
        """ This function will handle the human input from the keyboard (not right now) or voice """
        i = 1
        j = 0
        while not self.exit_flag:

            # Testing for cycling through labels
            # if False:
            #     j = j + 1
            #     if j > 30:
            #         j = 0
            #         self.tracking_object = i
            #         i = i + 1
            #
            #     if i > 3:
            #         i = 0

            if self.inputs.ai_output:
                self.tracking_object = self.inputs.ai_output

            if self.inputs.ai_output == 'reset':
                self.reset_position = True

            await asyncio.sleep(1)

    async def serial_comm(self, rate=0.05):
        """ Uses PID controllers to adjust the pan, tilt angles to send to the Arduino using the image_center and
            tracking_object_center properties as inputs and the PAN, TILT angles as the outputs with adjustable PID
            coefficients
        """
        while not self.exit_flag:

            if self.use_serial and self.tracking_object_center is not None:
                # Update the PID values for pan and tilt using the image_center and tracking_object_center
                pan_tilt_err = tuple(map(lambda i, j: i - j, self.image_center, self.tracking_object_center))
                self.pan = -self.pan_pid(
                    pan_tilt_err[0])  # Needs negative sign because the pan servo is mounted upside down
                self.tilt = self.tilt_pid(
                    pan_tilt_err[1])  # Needs negative sign because the tilt servo is mounted upside down
                if True:
                    print("object distance to center (px): ", pan_tilt_err)
                    print("pan: ", self.pan)
                    print("tilt: ", self.tilt)

                # If we are tracking an object, then send commands to the Arduino to move the pan/tilt mechanism
                self.arduino.write(f'<PAN,{self.pan}>\n'.encode('utf-8'))
                self.arduino.write(f'<TILT,{self.tilt}>\n'.encode('utf-8'))

            await asyncio.sleep(rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--port', type=str, required=True, help='Port to establish serial communication ')
    args = parser.parse_args()

    args.port = 'COM7'

    app = Tracking('Ultimate Assistant Camera Man', port=args.port, use_voice=True, use_openai=True, use_serial=True)
    try:
        app.start()
    except KeyboardInterrupt:
        print('Keyboard interrupt')
        sys.exit(0)
