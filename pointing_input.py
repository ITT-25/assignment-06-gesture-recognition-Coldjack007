import recognizer
from recognizer import Point
import pyglet
from pyglet.window import mouse
import math
import cv2
import mediapipe as mp
from pynput.mouse import Controller
from PIL import Image
import sys
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

GESTURE_NAME = "circle"
GESTURE_NUMBER = "10"

video_id = 0
if len(sys.argv) > 1:
    video_id = int(sys.argv[1])

# Create a video capture object for the webcam
cap = cv2.VideoCapture(video_id)

WINDOW_HEIGHT = 400
WINDOW_WIDTH = 420
RANDOM_Y_OFFSET = 80 # For some reason, the y coordinate of the mediapipe is 80 off

SAVE_FILENAME = GESTURE_NAME + GESTURE_NUMBER + ".xml"

points = []
array_user_input = []

window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)
dollar = recognizer.DollarRecognizer()
batch = pyglet.graphics.Batch()

pointer = Controller()

record_finger = False
record_input = False

class User_input:
    def __init__(self, x, y):
        self.Bubble = pyglet.shapes.Circle(x, y, 2.0, color=(255,0,0), batch=batch)

class Unistroke_input:
    def __init__(self, x, y):
        self.Bubble = pyglet.shapes.Circle(x, y, 2.0, color=(0,0,255), batch=batch)
        
###########################################################################

DEBUG = True
NUM_HANDS = 2
DETECTION_CONFIDENCE = 0.9
TRACKING_CONFIDENCE = 0.7

class HandDetection():
    def __init__(self, num_hands=2, detection_confidence=0.9, tracking_confidence=0.7):
        self.detector = mp.solutions.hands.Hands(
            max_num_hands=num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

    def draw_landmarks(self, img, landmarks):
        # draws the landmarks, caution: overwrites the image data
        mp.solutions.drawing_utils.draw_landmarks(
            img, landmarks, mp.solutions.hands.HAND_CONNECTIONS
        )   

    def detect(self, img):
        hand_data = {}
        h, w, _ = img.shape #sometimes this makes the program crash? AttributeError: 'NoneType' object has no attribute 'shape'
        detections = self.detector.process(img)

        # was detection successful?
        success = detections.multi_hand_landmarks and detections.multi_handedness
        if not success:
            return False, hand_data

        for hand_landmarks, handedness in zip(detections.multi_hand_landmarks, detections.multi_handedness):
            # left or right hand?
            handedness_label = handedness.classification[0].label

            # collect coordinates within image
            img_coords = []
            for lm in hand_landmarks.landmark:
                x_px = int(lm.x * w)
                y_px = int(lm.y * h)
                img_coords.append((x_px, y_px))

            # hand data, e.g.: {"Left" : (landmark coords within image, landmark data to draw them)}
            hand_data[handedness_label] = (img_coords, hand_landmarks)
     
        return True, hand_data

################################################################################

def cv2glet(img,fmt):
    '''Assumes image is in BGR color space. Returns a pyimg object'''
    if fmt == 'GRAY':
        rows, cols = img.shape
        channels = 1
    else:
        rows, cols, channels = img.shape

    raw_img = Image.fromarray(img).tobytes()

    top_to_bottom_flag = -1
    bytes_per_row = channels*cols
    pyimg = pyglet.image.ImageData(width=cols, 
                                   height=rows, 
                                   fmt=fmt, 
                                   data=raw_img, 
                                   pitch=top_to_bottom_flag*bytes_per_row)
    return pyimg

#######################################################################
   
header = ET.Element("Gesture", {
    "Name": GESTURE_NAME,
    "Subject": "1",
    "Speed": "medium",
    "Number": GESTURE_NUMBER,
    "NumPts": str(len(points)),
    "Millseconds": "715",
    "AppName": "Gestures",
    "AppVer": "3.5.0.0",
    "Date": "Wednesday, June 25, 2025",
    "TimeOfDay": "19:40"
})

########################################################################

@window.event
def on_mouse_press(x, y, button, modifiers):
    global points, array_user_input
    points.clear()
    array_user_input.clear()

@window.event
def on_mouse_release(x, y, button, modifiers):
    if len(points) >= 10:
        for i in points:
            array_user_input.append(User_input(i.X, i.Y))
        result = dollar.recognize(points, False)
        print("Result: " + result.Name + " (" + str(round_it(result.Score,2)) + ") in " + str(result.Time) + " ms.")
        for i in points:
            ET.SubElement(header, "Point", {
                "X": str(int(i.X)),
                "Y": str(int(i.Y))
            })
        xml_bytes = ET.tostring(header, encoding='utf-8')
        parsed = minidom.parseString(xml_bytes)
        pretty_xml_as_str = parsed.toprettyxml(indent="  ")
        with open(SAVE_FILENAME, "w", encoding="utf-8") as f:
            f.write(pretty_xml_as_str)
    else:
        print("Too few points made. Please try again.")

@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    global points
    if buttons & mouse.LEFT:
        points.append(Point(x, y))

def round_it(n, d):
    d = math.pow(10, d)
    result = (n * d) / d
    return round(result)

@window.event
def on_key_press(symbol, modifiers):
    global record_finger
    if symbol == pyglet.window.key.R:
        record_finger = not record_finger
    if symbol == pyglet.window.key.Q:
        os._exit(0)

detector = HandDetection(NUM_HANDS, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE)

@window.event
def on_draw():
    window.clear()
    ret, frame = cap.read()

    #Mediapipe
    # preprocess
    #frame = cv2.flip(frame, 1)
    # detection
    detection_success, data = detector.detect(frame)
    # make detection visible
    items = data.items()
    x = 0
    y = 0
    for handedness, hand_data in items:
        #print(f'{handedness} hand detected. landmarks: {hand_data[0]}')
        detector.draw_landmarks(frame, hand_data[1])
        x, y = hand_data[0][8]
        y -= 80
        #percentage = y / WINDOW_HEIGHT
        #y = (1.0 - percentage) * WINDOW_HEIGHT
        #y = int(y)
    offset_x, offset_y = window.get_location()
    x += offset_x
    y += offset_y
    if record_finger:
        pointer.position = (x, y)

    img = cv2glet(frame, 'BGR')
    img.blit(0, 0, 0)
    batch.draw()

pyglet.app.run()