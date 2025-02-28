import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from screen_brightness_control import set_brightness, get_brightness
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import time
import sys
import win32gui
import win32con

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configure PyAutoGUI
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1
screen_width, screen_height = pyautogui.size()

# Initialize audio controls
try:
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
except Exception as e:
    print(f"Warning: Could not initialize audio controls: {e}")
    volume = None

class GestureController:
    def __init__(self):
        self.prev_hand_y = None
        self.prev_hand_x = None
        self.smooth_factor = 0.5
        self.is_mouse_down = False
        self.prev_gesture = None
        self.gesture_cooldown = time.time()
        self.cooldown_period = 0.3
        self.brightness_base = None
        self.volume_base = None
        self.last_volume_change = time.time()
        self.last_brightness_change = time.time()
        self.volume_cooldown = 0.1
        self.brightness_cooldown = 0.1
        self.scroll_start_y = None
        
    def calculate_finger_states(self, hand_landmarks):
        fingers = []
        
        # Thumb detection
        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip = hand_landmarks.landmark[3]
        thumb_extended = thumb_tip.x < thumb_ip.x if thumb_tip.x < hand_landmarks.landmark[0].x else thumb_tip.x > thumb_ip.x
        fingers.append(1 if thumb_extended else 0)
            
        # Fingers detection
        tips = [8, 12, 16, 20]  # Finger tips
        pips = [6, 10, 14, 18]  # PIP joints
        
        for tip, pip in zip(tips, pips):
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers

    def get_gesture(self, fingers):
        current_time = time.time()
        
        if current_time - self.gesture_cooldown < self.cooldown_period:
            return self.prev_gesture if self.prev_gesture else "NONE"
            
        # [Thumb, Index, Middle, Ring, Pinky]
        if fingers == [0, 1, 0, 0, 0]:      # Index only
            gesture = "MOVE"
        elif fingers == [0, 1, 1, 0, 0]:    # Index + Middle
            gesture = "SCROLL_READY"
        elif fingers == [1, 1, 0, 0, 0]:    # Thumb + Index
            gesture = "LEFT_CLICK"
        elif fingers == [1, 0, 1, 0, 0]:    # Thumb + Middle
            gesture = "RIGHT_CLICK"
        elif fingers == [0, 1, 1, 1, 0]:    # Index + Middle + Ring
            gesture = "BRIGHTNESS"
        elif fingers == [1, 1, 1, 0, 0]:    # Thumb + Index + Middle
            gesture = "VOLUME"
        else:
            gesture = "NONE"
            
        if gesture != self.prev_gesture:
            self.gesture_cooldown = current_time
            
        self.prev_gesture = gesture
        return gesture

    def process_gesture(self, frame, hand_landmarks, gesture):
        try:
            frame_height, frame_width, _ = frame.shape
            
            # Get index finger coordinates
            index_tip = hand_landmarks.landmark[8]
            x = int(np.interp(index_tip.x, [0, 1], [0, screen_width]))
            y = int(np.interp(index_tip.y, [0, 1], [0, screen_height]))
            
            if gesture == "MOVE":
                if self.prev_hand_x is not None:
                    smoothed_x = int(self.prev_hand_x + (x - self.prev_hand_x) * self.smooth_factor)
                    smoothed_y = int(self.prev_hand_y + (y - self.prev_hand_y) * self.smooth_factor)
                    pyautogui.moveTo(smoothed_x, smoothed_y)
                self.prev_hand_x = x
                self.prev_hand_y = y
                
            elif gesture == "LEFT_CLICK":
                pyautogui.click()
                time.sleep(0.2)
                
            elif gesture == "RIGHT_CLICK":
                pyautogui.rightClick()
                time.sleep(0.2)
                
            elif gesture == "SCROLL_READY":
                current_y = hand_landmarks.landmark[8].y
                if self.scroll_start_y is None:
                    self.scroll_start_y = current_y
                    cv2.putText(frame, "Scroll Mode - Move hand up/down", (10, 130),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                else:
                    scroll_amount = int((self.scroll_start_y - current_y) * 200)  # Increased multiplier for faster scrolling
                    
                    if abs(scroll_amount) > 0:  # Removed threshold to make scrolling more responsive
                        direction = "UP" if scroll_amount > 0 else "DOWN"
                        speed = min(abs(scroll_amount), 10)  # Increased max speed
                        cv2.putText(frame, f"Scrolling {direction} (Speed: {speed})", (10, 130),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                        
                        pyautogui.scroll(scroll_amount)
                    
                    # Don't reset scroll_start_y to allow continuous scrolling
                    
            elif gesture == "VOLUME" and volume is not None:
                current_time = time.time()
                if current_time - self.last_volume_change >= self.volume_cooldown:
                    if self.volume_base is None:
                        self.volume_base = hand_landmarks.landmark[4].y
                    else:
                        y_diff = hand_landmarks.landmark[4].y - self.volume_base
                        volume_change = -y_diff
                        current_volume = volume.GetMasterVolumeLevelScalar()
                        new_volume = max(0, min(1, current_volume + volume_change * 0.1))
                        volume.SetMasterVolumeLevelScalar(new_volume, None)
                        volume_percent = int(new_volume * 100)
                        cv2.putText(frame, f"Volume: {volume_percent}%", (10, 90),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                        self.last_volume_change = current_time
                    
            elif gesture == "BRIGHTNESS":
                current_time = time.time()
                if current_time - self.last_brightness_change >= self.brightness_cooldown:
                    if self.brightness_base is None:
                        self.brightness_base = hand_landmarks.landmark[8].y
                    else:
                        try:
                            y_diff = hand_landmarks.landmark[8].y - self.brightness_base
                            current_brightness = get_brightness()[0]
                            brightness_change = int(-y_diff * 50)
                            new_brightness = max(0, min(100, current_brightness + brightness_change))
                            set_brightness(new_brightness)
                            cv2.putText(frame, f"Brightness: {new_brightness}%", (10, 90),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                            self.last_brightness_change = current_time
                        except Exception as e:
                            print(f"Brightness control error: {e}")
            
            # Reset base values when gesture changes
            if gesture not in ["VOLUME", "BRIGHTNESS"]:
                self.volume_base = None
                self.brightness_base = None
                
        except Exception as e:
            print(f"Error in process_gesture: {e}")

def cleanup(cap):
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

def main():
    # Initialize webcam with error handling
    max_retries = 3
    retry_count = 0
    cap = None
    
    while retry_count < max_retries:
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                break
            retry_count += 1
            time.sleep(1)
        except Exception as e:
            print(f"Error initializing webcam (attempt {retry_count + 1}): {e}")
            retry_count += 1
            time.sleep(1)
    
    if cap is None or not cap.isOpened():
        print("Error: Could not initialize webcam after multiple attempts")
        return
    
    # Set lower resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    controller = GestureController()
    
    # Create window and get handle
    cv2.namedWindow('Hand Gesture Control', cv2.WINDOW_NORMAL)
    window_handle = win32gui.FindWindow(None, 'Hand Gesture Control')
    
    # Set window to be always on top
    if window_handle != 0:
        win32gui.SetWindowPos(window_handle, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                            win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
    
    # Initialize MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:
        
        print("\nGesture Control System Started")
        print("\nAvailable gestures (maximum 3 fingers):")
        print("1. Move Cursor: Index finger")
        print("2. Scroll: Index + Middle fingers")
        print("   - Hold the gesture and:")
        print("   - Move hand UP to scroll UP")
        print("   - Move hand DOWN to scroll DOWN")
        print("   - Speed varies with movement distance")
        print("3. Left Click: Thumb + Index finger")
        print("4. Right Click: Thumb + Middle finger")
        print("5. Volume Control: Thumb + Index + Middle fingers (move hand up/down)")
        print("6. Brightness Control: Index + Middle + Ring fingers (move hand up/down)")
        print("\nTo exit the program, press 'q'\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(0)
                    continue
                
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                results = hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, 
                            hand_landmarks, 
                            mp_hands.HAND_CONNECTIONS)
                        
                        finger_states = controller.calculate_finger_states(hand_landmarks)
                        gesture = controller.get_gesture(finger_states)
                        controller.process_gesture(frame, hand_landmarks, gesture)
                        
                        cv2.putText(frame, f"Gesture: {gesture}", (10, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
                cv2.imshow('Hand Gesture Control', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nProgram interrupted by user")
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            cleanup(cap)

if __name__ == "__main__":
    main()