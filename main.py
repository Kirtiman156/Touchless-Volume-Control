import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER                                          
from comtypes import CLSCTX_ALL         
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import pyautogui  
import time  


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Audio setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol, maxVol = volRange[0], volRange[1]         

# Webcam setup
cam = cv2.VideoCapture(0)       
cam.set(3, 640)
cam.set(4, 480)

# Gesture variables
lastPlayPauseTime = time.time()
isPlaying = False
volumeControlActive = False
previousLength = 0

# Define drawing specs for better visibility
landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=5, circle_radius=4)
connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3)

# Mediapipe hand tracking model with lower thresholds for better detection
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,  
    min_tracking_confidence=0.5) as hands:

    while cam.isOpened():
        success, image = cam.read()
        if not success:
            print("Failed to capture video")
            continue

        # Flip image horizontally for more intuitive interaction
        image = cv2.flip(image, 1)
        
        # Convert color and process with MediaPipe
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Status display
        cv2.putText(image, "Waiting for hand...", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Process hands if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    image, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec,
                    connection_drawing_spec
                )
                
                # Status display update
                cv2.putText(image, "Hand detected", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Get landmark points
                landmarks = []
                for lm in hand_landmarks.landmark:
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append([cx, cy])
                
                # Check for volume gesture - thumb and index finger pinch
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                
                # Draw volume control points
                cv2.circle(image, (thumb_tip[0], thumb_tip[1]), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(image, (index_tip[0], index_tip[1]), 10, (255, 0, 255), cv2.FILLED)
                cv2.line(image, (thumb_tip[0], thumb_tip[1]), (index_tip[0], index_tip[1]), (255, 0, 255), 3)
                
                # Calculate distance between thumb and index finger
                length = math.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])
                
                # Check if significant change in length to activate volume control
                if abs(length - previousLength) > 3:
                    # Map distance to volume range (adjust these values based on your hand size)
                    vol = np.interp(length, [20, 200], [minVol, maxVol])
                    volBar = np.interp(length, [20, 200], [400, 150])
                    volPer = np.interp(length, [20, 200], [0, 100])
                    
                    # Set system volume
                    volume.SetMasterVolumeLevel(vol, None)
                    volumeControlActive = True
                    
                    # Display volume info
                    cv2.putText(image, f"Volume: {int(volPer)}%", (10, 70), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                              
                    # Draw volume bar
                    cv2.rectangle(image, (50, 150), (85, 400), (0, 0, 0), 3)
                    cv2.rectangle(image, (50, int(volBar)), (85, 400), (0, 0, 255), cv2.FILLED)
                
                # Update previous length
                previousLength = length
                
                # Check for play/pause gesture - closed fist
                # Count extended fingers
                finger_tips = [8, 12, 16, 20]  # Index, middle, ring, pinky fingertips
                finger_bases = [5, 9, 13, 17]  # Base of each finger
                fingers_extended = 0
                
                # Check each finger
                for tip, base in zip(finger_tips, finger_bases):
                    if landmarks[tip][1] < landmarks[base][1]:  # If fingertip is higher than base
                        fingers_extended += 1
                
                # Play/pause with closed fist (no extended fingers)
                current_time = time.time()
                if fingers_extended == 0 and current_time - lastPlayPauseTime > 1.5:
                    pyautogui.press("space")
                    isPlaying = not isPlaying
                    lastPlayPauseTime = current_time
                    
                    # Display play/pause status
                    status = "Paused" if isPlaying else "Playing"
                    cv2.putText(image, f"Media: {status}", (10, 110), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    print(f"Media {status}")
        
        # Show instructions
        cv2.putText(image, "Volume: Pinch thumb & index", (320, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(image, "Play/Pause: Closed fist", (320, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(image, "Press 'q' to quit", (320, 90), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Display frame
        cv2.imshow("Gesture Control", image)
        
        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cam.release()
cv2.destroyAllWindows()
