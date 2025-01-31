import cv2
import numpy as np
from keras.models import model_from_json
import pyautogui
import shutil
import os
from tkinter import filedialog
import tkinter as tk
import time

# Initialize tkinter
root = tk.Tk()
root.withdraw()  # Hide the main window

# Configure PyAutoGUI
pyautogui.FAILSAFE = False  # Disable failsafe for continuous operation
pyautogui.PAUSE = 0.1

class GestureController:
    def __init__(self):
        try:
            # Load the model
            json_file = open("SignLanguageDetectionUsingCNN-main/signlanguagedetectionmodel48x48.json", "r")
            # json_file = open("SignLanguageDetectionUsingCNN-main/signlanguagedetectionmodel128x128.json", "r")
            model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(model_json)
            self.model.load_weights("SignLanguageDetectionUsingCNN-main/signlanguagedetectionmodel48x48.h5")
            # self.model.load_weights("SignLanguageDetectionUsingCNN-main/signlanguagedetectionmodel128x128.h5")
            
            # Initialize video capture
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open video capture device")
            
            # Initialize variables
            self.label = ['A', 'B', 'C', 'D', 'E', 'blank']
            self.copied_file_path = None
            self.screen_width, self.screen_height = pyautogui.size()
            self.last_mouse_pos = pyautogui.position()
            self.smooth_factor = 0.5  # Adjust this value to change mouse movement smoothness
            
            # Movement threshold to prevent small unintended movements
            self.movement_threshold = 10
            
            # Gesture cooldown mechanism
            self.last_action_time = 0
            self.cooldown_time = 1.0  # 1 second cooldown between actions
            
            print("Gesture Controller initialized successfully")
        except Exception as e:
            print(f"Error initializing Gesture Controller: {str(e)}")
            raise

    def extract_features(self, image):
        """Convert and normalize the image for model input"""
        try:
            feature = np.array(image)
            # feature = feature.reshape(1, 128, 128, 1)
            feature = feature.reshape(1, 48, 48, 1)
            return feature / 255.0
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None

    def get_hand_position(self, frame):
        """Calculate hand position from the frame"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Calculate moments
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
            
            # Calculate distance to move
            dx = target_x - current_x
            dy = target_y - current_y
            
            # Apply movement threshold
            if abs(dx) < self.movement_threshold and abs(dy) < self.movement_threshold:
                return
            
            # Apply smoothing
            new_x = current_x + (dx * self.smooth_factor)
            new_y = current_y + (dy * self.smooth_factor)
            
            # Ensure coordinates are within screen bounds
            new_x = max(0, min(new_x, self.screen_width))
            new_y = max(0, min(new_y, self.screen_height))
            
            # Move mouse
            pyautogui.moveTo(int(new_x), int(new_y), duration=0.1)
        except Exception as e:
            print(f"Error in smooth mouse movement: {str(e)}")

    def mouse_move(self, hand_position):
        """Move mouse based on hand position"""
        try:
            if hand_position is None:
                return
            
            # Map hand coordinates to screen coordinates
            screen_x = (hand_position[0] / 300) * self.screen_width
            screen_y = (hand_position[1] / 260) * self.screen_height
            
            self.smooth_mouse_movement(screen_x, screen_y)
        except Exception as e:
            print(f"Error in mouse movement: {str(e)}")

    def copy_file(self):
        """Open file dialog to select and copy a file"""
        try:
            file_path = filedialog.askopenfilename(title="Select file to copy")
            if file_path:
                self.copied_file_path = file_path
                print(f"File selected for copying: {file_path}")
                return True
        except Exception as e:
            print(f"Error in copy operation: {str(e)}")
        return False

    def paste_file(self):
        """Paste the previously copied file"""
        try:
            if self.copied_file_path and os.path.exists(self.copied_file_path):
                dest_folder = filedialog.askdirectory(title="Select destination folder")
                if dest_folder:
                    dest_path = os.path.join(dest_folder, os.path.basename(self.copied_file_path))
                    shutil.copy2(self.copied_file_path, dest_path)
                    print(f"File pasted to: {dest_path}")
                    return True
        except Exception as e:
            print(f"Error in paste operation: {str(e)}")
        return False

    def open_file(self):
        """Open file dialog to select and open a file"""
        try:
            file_path = filedialog.askopenfilename(title="Select file to open")
            if file_path:
                os.startfile(file_path)
                print(f"Opened file: {file_path}")
                return True
        except Exception as e:
            print(f"Error opening file: {str(e)}")
        return False

    def right_click(self):
        """Perform right-click at current mouse position"""
        try:
            pyautogui.rightClick()
            return True
        except Exception as e:
            print(f"Error performing right click: {str(e)}")
        return False

    def run(self):
        """Main loop for gesture detection and control"""
        print("Starting gesture detection. Press 'ESC' to exit.")
        
        while True:
            try:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Draw rectangle for hand region
                cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
                crop_frame = frame[40:300, 0:300]
                
                # Process frame for gesture detection
                gray_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
                resized_frame = cv2.resize(gray_frame, (48, 48))
                feature = self.extract_features(resized_frame)
                
                if feature is not None:
                    # Get prediction
                    pred = self.model.predict(feature)
                    prediction_label = self.label[pred.argmax()]
                    
                    # Get hand position
                    hand_position = self.get_hand_position(crop_frame)
                    
                    # Current time for cooldown
                    current_time = time.time()
                    
                    # Handle gestures with cooldown
                    if current_time - self.last_action_time > self.cooldown_time:
                        if prediction_label == 'A':
                            self.mouse_move(hand_position)
                        elif prediction_label == 'B' and hand_position:
                            if self.copy_file():
                                self.last_action_time = current_time
                        elif prediction_label == 'C' and hand_position:
                            if self.paste_file():
                                self.last_action_time = current_time
                        elif prediction_label == 'D' and hand_position:
                            if self.open_file():
                                self.last_action_time = current_time
                        elif prediction_label == 'E' and hand_position:
                            if self.right_click():
                                self.last_action_time = current_time

                    # Display prediction
                    cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
                    if prediction_label == 'blank':
                        cv2.putText(frame, " ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (255, 255, 255), 2, cv2.LINE_AA)
                    else:
                        accuracy = "{:.2f}".format(np.max(pred) * 100)
                        cv2.putText(frame, f'{prediction_label} {accuracy}%', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Display frame
                cv2.imshow("Gesture Control", frame)
                
                # Check for exit
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    print("ESC pressed. Exiting...")
                    break

            except Exception as e:
                print(f"Error in main loop: {str(e)}")
                break

        # Cleanup
        print("Cleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        # Create and run the gesture controller
        controller = GestureController()
        controller.run()
    except Exception as e:
        print(f"Fatal error: {str(e)}")