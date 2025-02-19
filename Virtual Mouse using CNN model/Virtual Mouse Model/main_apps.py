import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import pyautogui
import time
import win32gui
import win32con

# Configure PyAutoGUI
pyautogui.FAILSAFE = False  # Disable failsafe for continuous operation
pyautogui.PAUSE = 0.1

class GestureController:
    def __init__(self):
        try:
            # Load the model
            json_file = open("SignLanguageDetectionUsingCNN-main/signlanguagedetectionmodel128x128main_model.json", "r")
            model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(model_json)
            self.model.load_weights("SignLanguageDetectionUsingCNN-main/signlanguagedetectionmodel128x128main_model.h5")
            
            # Initialize video capture
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open video capture device")
            
            # Initialize variables
            self.label = ['move cursor', 'paste', 'Copy', 'right click','double click','blank']
            self.screen_width, self.screen_height = pyautogui.size()
            self.last_mouse_pos = pyautogui.position()
            self.smooth_factor = 0.5
            
            # Movement tracking
            self.movement_threshold = 5
            self.movement_direction = ""
            self.prev_hand_position = None
            self.vertical_threshold = 1

            # Gesture cooldown mechanism
            self.last_action_time = 0
            self.cooldown_time = 2.0
            self.action_performed = False
            self.is_vertical_movement = False
            
            print("Gesture Controller initialized successfully")
        except Exception as e:
            print(f"Error initializing Gesture Controller: {str(e)}")
            raise

    def make_window_topmost(self, window_name):
        """Make the window stay on top of all others"""
        try:
            hwnd = win32gui.FindWindow(None, window_name)
            if hwnd:
                win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                     win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
        except Exception as e:
            print(f"Error setting window topmost: {str(e)}")

    def extract_features(self, image):
        try:
            image = cv2.resize(image, (128, 128))
            feature = np.array(image)
            feature = feature.reshape(1, 128, 128, 1)
            return feature / 255.0
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None

    def get_hand_position(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            moments = cv2.moments(thresh)
            
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                return (cx, cy)
            return None
        except Exception as e:
            print(f"Error getting hand position: {str(e)}")
            return None

    def determine_movement_direction(self, current_pos, prev_pos):
        if prev_pos is None:
            return ""
            
        dx = current_pos[0] - prev_pos[0]
        dy = current_pos[1] - prev_pos[1]
        
        # Threshold for movement detection
        threshold = 1
        
        if abs(dx) > abs(dy):
            if dx > threshold:
                return "Left"
            elif dx < -threshold:
                return "Right"
        else:
            if dy > threshold:
                return "Down"
            elif dy < -threshold:
                return "Up"
        
        return "stop"

    def mouse_move(self, hand_position):
        try:
            if hand_position is None:
                return
            
            global old_top
            if not hasattr(self, 'prev_hand_position'):
                self.prev_hand_position = hand_position
                old_top = hand_position
                return

            diff_x = hand_position[0] - old_top[0]
            diff_y = hand_position[1] - old_top[1]
            
            # Invert coordinates for more intuitive control
            diff_x = -diff_x
            diff_y = -diff_y
            
            # Vertical movement detection
            if abs(diff_y) > self.vertical_threshold:
                self.is_vertical_movement = True
            else:
                self.is_vertical_movement = False
            
            # Dynamic multiplier based on movement magnitude
            if self.is_vertical_movement:
                mul_y = self.get_multiplierY(abs(diff_y))
                # mul_y = self.get_multiplier(abs(diff_y))
            else:
                mul_x = self.get_multiplierX(abs(diff_x))
                # mul_y = self.get_multiplier(abs(diff_y))

            # Update movement direction
            self.movement_direction = self.determine_movement_direction(hand_position, self.prev_hand_position)
            
            # Apply movement with vertical control
            if self.is_vertical_movement:
                pyautogui.moveRel(0, diff_y * mul_y, duration=0.1)
            else:
                pyautogui.moveRel(diff_x * mul_x, 0, duration=0.1)
            
            # Update previous positions
            old_top = hand_position
            self.prev_hand_position = hand_position
            
        except Exception as e:
            print(f"Error in mouse movement: {str(e)}")
    def get_multiplierY(self, diff):
        if diff >= 40:
            return 200
        if diff >= 30:
            return 150
        elif diff >= 25:
            return 100
        elif diff >= 20:
            return 80
        elif diff >= 15:
            return 60
        elif diff >= 10:
            return 50
        elif diff >= 5:
            return 30
        elif diff >= 2:
            return 20
        return 10
    def get_multiplierX(self, diff):
        if diff >= 40:
            return 150
        if diff >= 30:
            return 90
            # return 70
        elif diff >= 25:
            return 70
            # return 50
        elif diff >= 20:
            return 50
            # return 30
        elif diff >= 15:
            return 40
            # return 20
        elif diff >= 10:
            return 35
            # return 15
        elif diff >= 5:
            return 20
            # return 10
        return 10

    def click_mouse(self):
        try:
            x, y = pyautogui.position()
            pyautogui.doubleClick(x, y)
            return True
        except Exception as e:
            print(f"Error performing double click: {str(e)}")
            return False

    def copy_command(self):
        try:
            x, y = pyautogui.position()
            pyautogui.click(x, y)
            pyautogui.hotkey('ctrl', 'c')
            return True
        except Exception as e:
            print(f"Error performing copy command: {str(e)}")
            return False

    def paste_command(self):
        try:
            x, y = pyautogui.position()
            pyautogui.click(x, y)
            pyautogui.hotkey('ctrl', 'v')
            return True
        except Exception as e:
            print(f"Error performing paste command: {str(e)}")
            return False

    def right_click(self):
        try:
            x, y = pyautogui.position()
            pyautogui.rightClick(x, y)
            return True
        except Exception as e:
            print(f"Error performing right click: {str(e)}")
            return False

    def run(self):
        print("Starting gesture detection. Press 'ESC' to exit.")
        global old_top
        old_top = (0, 0)
        
        while True:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Make window topmost
                frame = cv2.flip(frame, 1)
                self.make_window_topmost("Gesture Control")

                # Create detection area
                cv2.rectangle(frame, (0, 40), (680, 480), (0, 165, 255), 0)
                crop_frame = frame[40:680, 0:480]

                # Process frame for prediction
                gray_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
                resized_frame = cv2.resize(gray_frame, (128, 128))
                feature = self.extract_features(resized_frame)
                
                if feature is not None:
                    pred = self.model.predict(feature)
                    prediction_label = self.label[pred.argmax()]
                    accuracy = np.max(pred) * 100
                    
                    hand_position = self.get_hand_position(crop_frame)
                    current_time = time.time()
                    
                    if prediction_label == 'move cursor':
                            self.mouse_move(hand_position)
                    elif current_time - self.last_action_time > self.cooldown_time:
                        self.action_performed = False
                        if prediction_label == 'paste' and not self.action_performed and accuracy>=95:
                            if self.paste_command():
                                self.last_action_time = current_time
                                self.action_performed = True
                                
                        elif prediction_label == 'Copy' and not self.action_performed and accuracy>=95:
                            if self.copy_command():
                                self.last_action_time = current_time
                                self.action_performed = True
                        elif prediction_label == 'double click' and not self.action_performed and accuracy>=95:
                            if self.click_mouse():
                                self.last_action_time = current_time
                                self.action_performed = True
                        elif prediction_label == 'right click' and not self.action_performed and accuracy>=95:
                            if self.right_click():
                                self.last_action_time = current_time
                                self.action_performed = True

                    # Display prediction and movement direction
                    cv2.rectangle(frame, (0, 0), (680, 40), (0, 165, 255), -1)
                    if prediction_label == 'blank':
                        display_text = ""
                    
                    elif prediction_label == 'move cursor':
                        display_text = f"Moving {self.movement_direction} ({accuracy:.2f}%)"
                    else:
                        display_text = f'{prediction_label} ({accuracy:.2f}%)'
                    
                    
                    cv2.putText(frame, display_text, (210, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Always show the frame
                cv2.namedWindow('Gesture Control', cv2.WINDOW_NORMAL)
                cv2.imshow("Gesture Control", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    print("ESC pressed. Exiting...")
                    break
                elif key == ord('q'):  # 'q' key
                    break
                    
            except Exception as e:
                print(f"Error in main loop: {str(e)}")
                break

        print("Cleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        controller = GestureController()
        controller.run()
    except Exception as e:
        print(f"Fatal error: {str(e)}")