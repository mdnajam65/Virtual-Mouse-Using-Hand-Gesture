import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import pyautogui
import time

# Configure PyAutoGUI
pyautogui.FAILSAFE = False  # Disable failsafe for continuous operation
pyautogui.PAUSE = 0.1

class GestureController:
    def __init__(self):
        try:
            # Load the model
            json_file = open("SignLanguageDetectionUsingCNN-main/signlanguagedetectionmodel128x128main_model.json", "r")
            # json_file = open("SignLanguageDetectionUsingCNN-main/signlanguagedetectionmodel48x48.json", "r")
            model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(model_json)
            self.model.load_weights("SignLanguageDetectionUsingCNN-main/signlanguagedetectionmodel128x128main_model.h5")
            # self.model.load_weights("SignLanguageDetectionUsingCNN-main/signlanguagedetectionmodel48x48.h5")
            
            # Initialize video capture
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open video capture device")
            
            # Initialize variables
            self.label = ['move cursor', 'paste', 'Copy','right click', 'double click', 'blank']
            self.screen_width, self.screen_height = pyautogui.size()
            self.last_mouse_pos = pyautogui.position()
            self.smooth_factor = 0.5  # Adjust this value to change mouse movement smoothness
            
            # Movement threshold to prevent small unintended movements
            self.movement_threshold = 10
            
            # Gesture cooldown mechanism
            self.last_action_time = 0
            self.cooldown_time = 1.0  # 1 second cooldown between actions
            
            # Flag to track if an action has been performed
            self.action_performed = False
            
            print("Gesture Controller initialized successfully")
        except Exception as e:
            print(f"Error initializing Gesture Controller: {str(e)}")
            raise

    def extract_features(self, image):
        """Convert and normalize the image for model input"""
        try:
            image = cv2.resize(image, (128, 128))
            feature = np.array(image)
            feature = feature.reshape(1, 128,128,1)
            # feature = feature.reshape(1, 48, 48, 1)
            return feature / 255.0
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None

    def get_hand_position(self, frame):
        """Calculate hand position from the frame"""
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

    def smooth_mouse_movement(self, target_x, target_y):
        """Implement smooth mouse movement"""
        try:
            current_x, current_y = pyautogui.position()
            
            dx = target_x - current_x
            dy = target_y - current_y
            
            if abs(dx) < self.movement_threshold and abs(dy) < self.movement_threshold:
                return
            
            new_x = current_x + (dx * self.smooth_factor)
            new_y = current_y + (dy * self.smooth_factor)
            
            new_x = max(0, min(new_x, self.screen_width))
            new_y = max(0, min(new_y, self.screen_height))
            
            pyautogui.moveTo(int(new_x), int(new_y), duration=0.01)
            # pyautogui.moveTo(int(new_x), int(new_y), duration=0.1)
        except Exception as e:
            print(f"Error in smooth mouse movement: {str(e)}")

    # def mouse_move(self, hand_position):
    #     """Move mouse based on hand position"""
    #     try:
    #         if hand_position is None:
    #             return
            
    #         # screen_x = (hand_position[0] / 600) * self.screen_width
    #         # screen_y = (hand_position[1] / 600) * self.screen_height
    #         screen_x = (hand_position[0]) * (self.screen_width/300)
    #         screen_y = (hand_position[1]) * (self.screen_height/260)
            
    #         self.smooth_mouse_movement(screen_x, screen_y)
    #     except Exception as e:
    #         print(f"Error in mouse movement: {str(e)}")

    def mouse_move(self, hand_position):
        """Move mouse based on hand position"""
        try:
            if hand_position is None:
                return
            
            global old_top
            diff_x = hand_position[0] - old_top[0]
            diff_y = hand_position[1] - old_top[1]
            
            # Invert x and y coordinates
            diff_x = -diff_x
            diff_y = -diff_y
            # Adjust sensitivity
            if abs(diff_x) >= 30:
                mul_x = 70
                # mul_x = 40
            elif abs(diff_x) >= 20:
                mul_x = 40
                # mul_x = 20
            elif abs(diff_x) >= 15:
                mul_x = 30
                # mul_x = 15
            elif abs(diff_x) >= 10:
                mul_x = 20
                # mul_x = 10
            elif abs(diff_x) >= 5:
                mul_x = 10
                # mul_x = 5
            else:
                mul_x = 1

            if abs(diff_y) >= 30:
                mul_y = 70
                # mul_y = 20
            elif abs(diff_y) >= 25:
                mul_y = 50
                # mul_y = 15
            elif abs(diff_y) >= 20:
                mul_y = 30
                # mul_y = 12
            elif abs(diff_y) >= 15:
                mul_y = 20
                # mul_y = 9
            elif abs(diff_y) >= 10:
                mul_y = 15
            else:
                mul_y = 1

            pyautogui.moveRel(diff_x * mul_x, diff_y * mul_y, duration=0.1)
            # vertical_threshold = 5  # Adjust this value
    
            # if abs(diff_y) > vertical_threshold:
            #     pyautogui.moveRel(diff_x * mul_x, diff_y * mul_y, duration=0.01)
            # else:
            #     pyautogui.moveRel(diff_x * mul_x, 0, duration=0.01)
            old_top = hand_position
            
        except Exception as e:
            print(f"Error in mouse movement: {str(e)}")


    def click_mouse(self):
        """Double click at current mouse position"""
        try:
            x, y = pyautogui.position()
            pyautogui.doubleClick(x, y)
            return True
        except Exception as e:
            print(f"Error performing double click: {str(e)}")
            return False

    def copy_command(self):
        """Perform Ctrl+C command"""
        try:
            pyautogui.hotkey('ctrl', 'c')
            return True
        except Exception as e:
            print(f"Error performing copy command: {str(e)}")
            return False

    def paste_command(self):
        """Perform Ctrl+V command"""
        try:
            x, y = pyautogui.position()
            pyautogui.click(x, y)
            pyautogui.hotkey('ctrl', 'v')
            return True
        except Exception as e:
            print(f"Error performing paste command: {str(e)}")
            return False

    def right_click(self):
        """Perform right click at current mouse position"""
        try:
            x, y = pyautogui.position()
            pyautogui.rightClick(x, y)
            return True
        except Exception as e:
            print(f"Error performing right click: {str(e)}")
            return False

    def run(self):
        """Main loop for gesture detection and control"""
        print("Starting gesture detection. Press 'ESC' to exit.")
        global old_top
        old_top = (0, 0)
        
        while True:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                frame = cv2.flip(frame, 1)
                # self.make_window_topmost("Gesture Control")
                cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
                crop_frame = frame[40:300, 0:300]
                # cv2.rectangle(frame, (0, 0), (600, 600), (0, 165, 255), 1)
                # crop_frame = frame[0:600, 0:600]

                gray_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
                resized_frame = cv2.resize(gray_frame, (128, 128))
                # resized_frame = cv2.resize(gray_frame, (48, 48))
                feature = self.extract_features(resized_frame)
                
                if feature is not None:
                    pred = self.model.predict(feature)
                    prediction_label = self.label[pred.argmax()]
                    accuracy = np.max(pred) * 100
                    
                    hand_position = self.get_hand_position(crop_frame)
                    current_time = time.time()
                    
                    if current_time - self.last_action_time > self.cooldown_time:
                        self.action_performed = False
                        if accuracy >= 95.0:
                            if prediction_label == 'move cursor':
                                self.mouse_move(hand_position)
                            elif prediction_label == 'paste' and not self.action_performed:
                                if self.paste_command():
                                    self.last_action_time = current_time
                                    self.action_performed = True
                            elif prediction_label == 'Copy' and not self.action_performed:
                                if self.copy_command():
                                    self.last_action_time = current_time
                                    self.action_performed = True
                            elif prediction_label == 'double click' and not self.action_performed:
                                if self.click_mouse():
                                    self.last_action_time = current_time
                                    self.action_performed = True
                            elif prediction_label == 'right click' and not self.action_performed:
                                if self.right_click():
                                    self.last_action_time = current_time
                                    self.action_performed = True

                    # Display prediction
                    cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
                    if prediction_label == 'blank':
                        cv2.putText(frame, " ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (255, 255, 255), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, f'{prediction_label} {accuracy:.2f}%', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow("Gesture Control", frame)
                
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    print("ESC pressed. Exiting...")
                    break
                if cv2.waitKey(27) & 0xFF == ord('q'):  # Press 'q' to quit
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