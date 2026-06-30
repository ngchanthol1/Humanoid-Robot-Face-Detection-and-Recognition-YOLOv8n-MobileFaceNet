"""
Integrated Face Recognition Robot Control System - Enhanced GUI
Redesigned with improved visuals, larger buttons, and distinct motor colors

Features:
- Enhanced color-coded GUI interface
- Large prominent buttons
- Distinct colors for each motor (1-16)
- UART connection for Raspberry Pi 5 to STM32F103 via /dev/ttyAMA2
- Proper connection status validation
- Detection Fail after 3 consecutive alerts
- Detection pauses during welcome sequence until robot finishes

Requirements:
- servo_motor_protocols.py (must be in same directory)
- blazeface.tflite
- mobilefacenet.tflite
- mrdavid/ folder with training images
"""

import cv2
import numpy as np
import os
from pathlib import Path
import pickle
import time
from datetime import datetime
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog, font as tkfont
import serial
import serial.tools.list_ports

# Import servo motor protocols
from servo_motor_protocols import (
    ServoMotorProtocols,
    ServoMotorController,
    WelcomeSequence,
    get_rgb_protocol,
    get_angle_protocol
)

# Check for TensorFlow Lite
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Face recognition will be disabled.")


# ==================== COLOR SCHEME ====================
class ColorScheme:
    """Enhanced color scheme for the application"""
    
    # Main background colors
    BG_DARK = "#1a1a2e"
    BG_MEDIUM = "#16213e"
    BG_LIGHT = "#0f3460"
    BG_ACCENT = "#533483"
    
    # Text colors
    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#b8b8b8"
    TEXT_ACCENT = "#00fff5"
    
    # Status colors
    STATUS_CONNECTED = "#00ff88"
    STATUS_DISCONNECTED = "#ff4757"
    STATUS_WARNING = "#ffa502"
    STATUS_INFO = "#3498db"
    
    # Button colors
    BTN_PRIMARY = "#0f3460"
    BTN_PRIMARY_HOVER = "#ff6b81"
    BTN_SECONDARY = "#0f3460"
    BTN_SUCCESS = "#00d26a"
    BTN_DANGER = "#0f3460"
    BTN_WARNING = "#0f3460"
    
    # Motor colors (16 distinct colors)
    MOTOR_COLORS = {
        1: "#45B7D1",
        2: "#45B7D1",
        3: "#45B7D1",
        4: "#45B7D1",
        5: "#45B7D1",
        6: "#45B7D1",
        7: "#45B7D1",
        8: "#45B7D1",
        9: "#45B7D1",
        10: "#45B7D1",
        11: "#45B7D1",
        12: "#45B7D1",
        13: "#45B7D1",
        14: "#45B7D1",
        15: "#45B7D1",
        16: "#45B7D1",
    }

    # RGB display colors
    RGB_COLORS = {
        "RED": "#FF0000",
        "GREEN": "#00FF00",
        "BLUE": "#0000FF"
    }


class FaceRecognitionSystem:
    """
    Face Recognition System using BlazeFace detection and MobileFaceNet recognition
    """
    
    def __init__(self, detection_model_path, recognition_model_path, known_faces_path, debug_mode=False):
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow is required for face recognition")
        
        # Load detection model (BlazeFace)
        self.detector = tf.lite.Interpreter(model_path=detection_model_path)
        self.detector.allocate_tensors()
        self.detection_input_details = self.detector.get_input_details()
        self.detection_output_details = self.detector.get_output_details()
        
        # Load recognition model (MobileFaceNet)
        self.recognizer = tf.lite.Interpreter(model_path=recognition_model_path)
        self.recognizer.allocate_tensors()
        self.recognition_input_details = self.recognizer.get_input_details()
        self.recognition_output_details = self.recognizer.get_output_details()
        
        self.camera = None
        self.debug_mode = debug_mode
        self.known_face_embeddings = []
        self.known_face_name = "Mr. David"
        
        # Recognition thresholds
        self.recognition_threshold = 0.75
        self.strict_threshold = 0.85
        self.min_matching_embeddings = 4
        
        # Display duration settings
        self.welcome_duration = 3
        self.alert_duration = 1
        self.detection_fail_duration = 3
        
        # State tracking
        self.in_welcome_mode = False
        self.welcome_start_time = None
        self.welcome_last_box = None
        self.in_alert_mode = False
        self.alert_start_time = None
        self.alert_last_box = None
        
        # Detection fail tracking
        self.in_detection_fail_mode = False
        self.detection_fail_start_time = None
        self.consecutive_alert_count = 0
        self.max_consecutive_alerts = 3
        
        # Welcome sequence tracking - NEW
        self.welcome_sequence_running = False
        self.welcome_sequence_status = ""
        
        self.on_david_recognized = None
        self.is_running = False
        
        self.load_known_faces(known_faces_path)
    
    def load_known_faces(self, folder_path):
        """Load images from the known faces folder and create embeddings"""
        print(f"Loading known faces from: {folder_path}")
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist!")
            return
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(folder_path).glob(f'*{ext}'))
            image_files.extend(Path(folder_path).glob(f'*{ext.upper()}'))
        
        if len(image_files) == 0:
            print("Warning: No images found in the folder!")
            return
        
        print(f"Found {len(image_files)} images")
        
        for img_path in image_files:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                embedding = self.get_face_embedding(img)
                
                if embedding is not None:
                    self.known_face_embeddings.append(embedding)
                    print(f"Processed: {img_path.name}")
                    
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
        
        print(f"Successfully loaded {len(self.known_face_embeddings)} face embeddings")
        self.save_embeddings()
    
    def save_embeddings(self, filename='known_faces.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump({'embeddings': self.known_face_embeddings, 'name': self.known_face_name}, f)
    
    def load_embeddings(self, filename='known_faces.pkl'):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.known_face_embeddings = data['embeddings']
                self.known_face_name = data['name']
            return True
        return False
    
    def preprocess_for_detection(self, image):
        input_shape = self.detection_input_details[0]['shape']
        height, width = input_shape[1], input_shape[2]
        img_resized = cv2.resize(image, (width, height))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 127.5 - 1.0
        return np.expand_dims(img_normalized, axis=0)
    
    def detect_faces(self, image):
        input_data = self.preprocess_for_detection(image)
        self.detector.set_tensor(self.detection_input_details[0]['index'], input_data)
        self.detector.invoke()
        
        boxes = self.detector.get_tensor(self.detection_output_details[0]['index'])
        scores = self.detector.get_tensor(self.detection_output_details[1]['index'])
        
        faces = []
        h, w = image.shape[:2]
        
        for i in range(len(scores[0])):
            if scores[0][i] > 0.5:
                box = boxes[0][i]
                y1, x1, y2, x2 = int(box[0]*h), int(box[1]*w), int(box[2]*h), int(box[3]*w)
                faces.append((max(0,x1), max(0,y1), min(w,x2), min(h,y2)))
        
        return faces
    
    def preprocess_for_recognition(self, face_image):
        input_shape = self.recognition_input_details[0]['shape']
        height, width = input_shape[1], input_shape[2]
        face_resized = cv2.resize(face_image, (width, height))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_normalized = face_rgb.astype(np.float32) / 255.0
        return np.expand_dims(face_normalized, axis=0)
    
    def get_face_embedding(self, face_image):
        try:
            input_data = self.preprocess_for_recognition(face_image)
            self.recognizer.set_tensor(self.recognition_input_details[0]['index'], input_data)
            self.recognizer.invoke()
            embedding = self.recognizer.get_tensor(self.recognition_output_details[0]['index'])
            return (embedding / np.linalg.norm(embedding))[0]
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    def recognize_face(self, face_image):
        if len(self.known_face_embeddings) == 0:
            return False, "Unknown", 0.0
        
        current_embedding = self.get_face_embedding(face_image)
        if current_embedding is None:
            return False, "Unknown", 0.0
        
        similarities = np.array([np.dot(current_embedding, ke) for ke in self.known_face_embeddings])
        
        max_sim = np.max(similarities)
        mean_sim = np.mean(similarities)
        median_sim = np.median(similarities)
        matches = np.sum(similarities > self.recognition_threshold)
        high_conf = np.sum(similarities > self.strict_threshold)
        
        is_recognized = (
            max_sim > self.strict_threshold and
            mean_sim > (self.recognition_threshold - 0.05) and
            matches >= self.min_matching_embeddings and
            median_sim > (self.recognition_threshold - 0.10)
        )
        
        if high_conf < 3 and matches < self.min_matching_embeddings:
            is_recognized = False
        
        if is_recognized:
            return True, self.known_face_name, np.mean(np.sort(similarities)[-3:])
        return False, "Unknown", max_sim
    
    def initialize_camera(self, camera_index=0):
        self.camera = cv2.VideoCapture(camera_index)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        if not self.camera.isOpened():
            raise Exception("Could not open camera!")
        print("Camera initialized successfully")
    
    def draw_display_message(self, image, box, remaining_time, label_text, big_text, color):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        
        label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(image, (x1, y1 - 40), (x1 + label_size[0] + 20, y1), color, -1)
        cv2.putText(image, label_text, (x1 + 10, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        
        timer_text = f"Display: {int(remaining_time)}s"
        timer_size, _ = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y2 + 5), (x1 + timer_size[0] + 10, y2 + 35), color, -1)
        cv2.putText(image, timer_text, (x1 + 5, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        cl = 20
        ct = 4
        cv2.line(image, (x1, y1), (x1 + cl, y1), color, ct)
        cv2.line(image, (x1, y1), (x1, y1 + cl), color, ct)
        cv2.line(image, (x2, y1), (x2 - cl, y1), color, ct)
        cv2.line(image, (x2, y1), (x2, y1 + cl), color, ct)
        cv2.line(image, (x1, y2), (x1 + cl, y2), color, ct)
        cv2.line(image, (x1, y2), (x1, y2 - cl), color, ct)
        cv2.line(image, (x2, y2), (x2 - cl, y2), color, ct)
        cv2.line(image, (x2, y2), (x2, y2 - cl), color, ct)
        
        h, w = image.shape[:2]
        big_size, _ = cv2.getTextSize(big_text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)
        big_x = (w - big_size[0]) // 2
        cv2.rectangle(image, (big_x - 20, 60), (big_x + big_size[0] + 20, 120), color, -1)
        cv2.putText(image, big_text, (big_x, 105), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255,255,255), 3)
    
    def draw_detection_fail_message(self, image, remaining_time):
        """Draw detection fail message on the entire screen"""
        h, w = image.shape[:2]
        
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 100), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        main_text = "DETECTION FAIL"
        main_size, _ = cv2.getTextSize(main_text, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 4)
        main_x = (w - main_size[0]) // 2
        main_y = h // 2 - 30
        
        cv2.rectangle(image, (main_x - 30, main_y - main_size[1] - 20), 
                     (main_x + main_size[0] + 30, main_y + 20), (0, 0, 200), -1)
        cv2.rectangle(image, (main_x - 30, main_y - main_size[1] - 20), 
                     (main_x + main_size[0] + 30, main_y + 20), (255, 255, 255), 3)
        cv2.putText(image, main_text, (main_x, main_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 4)
        
        sub_text = f"3 consecutive unknown faces detected"
        sub_size, _ = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        sub_x = (w - sub_size[0]) // 2
        sub_y = main_y + 60
        cv2.putText(image, sub_text, (sub_x, sub_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        timer_text = f"Restarting in {int(remaining_time)}s..."
        timer_size, _ = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        timer_x = (w - timer_size[0]) // 2
        timer_y = sub_y + 50
        cv2.putText(image, timer_text, (timer_x, timer_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        icon_size = 40
        icon_thickness = 4
        icon_color = (0, 0, 255)
        
        cv2.line(image, (30, 30), (30 + icon_size, 30 + icon_size), icon_color, icon_thickness)
        cv2.line(image, (30 + icon_size, 30), (30, 30 + icon_size), icon_color, icon_thickness)
        cv2.line(image, (w - 30 - icon_size, 30), (w - 30, 30 + icon_size), icon_color, icon_thickness)
        cv2.line(image, (w - 30, 30), (w - 30 - icon_size, 30 + icon_size), icon_color, icon_thickness)
        cv2.line(image, (30, h - 30 - icon_size), (30 + icon_size, h - 30), icon_color, icon_thickness)
        cv2.line(image, (30 + icon_size, h - 30 - icon_size), (30, h - 30), icon_color, icon_thickness)
        cv2.line(image, (w - 30 - icon_size, h - 30 - icon_size), (w - 30, h - 30), icon_color, icon_thickness)
        cv2.line(image, (w - 30, h - 30 - icon_size), (w - 30 - icon_size, h - 30), icon_color, icon_thickness)
        
        count_text = f"Alert Count: {self.consecutive_alert_count}/{self.max_consecutive_alerts}"
        cv2.putText(image, count_text, (10, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
    
    def draw_welcome_sequence_message(self, image):
        """Draw welcome sequence running message - detection paused"""
        h, w = image.shape[:2]
        
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (100, 50, 50), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        
        main_text = "ROBOT SEQUENCE RUNNING"
        main_size, _ = cv2.getTextSize(main_text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)
        main_x = (w - main_size[0]) // 2
        main_y = h // 2 - 60
        
        cv2.rectangle(image, (main_x - 30, main_y - main_size[1] - 20), 
                     (main_x + main_size[0] + 30, main_y + 20), (200, 100, 0), -1)
        cv2.rectangle(image, (main_x - 30, main_y - main_size[1] - 20), 
                     (main_x + main_size[0] + 30, main_y + 20), (255, 255, 255), 3)
        cv2.putText(image, main_text, (main_x, main_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)
        
        sub_text = "Mr. David Recognized - Executing Welcome!"
        sub_size, _ = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        sub_x = (w - sub_size[0]) // 2
        sub_y = main_y + 50
        cv2.putText(image, sub_text, (sub_x, sub_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        if self.welcome_sequence_status:
            status_text = f"Status: {self.welcome_sequence_status}"
            status_size, _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            status_x = (w - status_size[0]) // 2
            status_y = sub_y + 45
            cv2.putText(image, status_text, (status_x, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        pause_text = "Face Detection PAUSED"
        pause_size, _ = cv2.getTextSize(pause_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        pause_x = (w - pause_size[0]) // 2
        pause_y = h - 80
        cv2.putText(image, pause_text, (pause_x, pause_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
        
        resume_text = "Will resume after sequence completes..."
        resume_size, _ = cv2.getTextSize(resume_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        resume_x = (w - resume_size[0]) // 2
        resume_y = h - 50
        cv2.putText(image, resume_text, (resume_x, resume_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        cv2.circle(image, (50, 50), 25, (255, 165, 0), 3)
        cv2.circle(image, (50, 50), 10, (255, 165, 0), -1)
        cv2.circle(image, (w - 50, 50), 25, (255, 165, 0), 3)
        cv2.circle(image, (w - 50, 50), 10, (255, 165, 0), -1)
        cv2.circle(image, (50, h - 50), 25, (255, 165, 0), 3)
        cv2.circle(image, (50, h - 50), 10, (255, 165, 0), -1)
        cv2.circle(image, (w - 50, h - 50), 25, (255, 165, 0), 3)
        cv2.circle(image, (w - 50, h - 50), 10, (255, 165, 0), -1)
    
    def stop(self):
        self.is_running = False
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
    
    def reset_detection_state(self):
        """Reset all detection states for a fresh start"""
        self.in_welcome_mode = False
        self.welcome_start_time = None
        self.welcome_last_box = None
        self.in_alert_mode = False
        self.alert_start_time = None
        self.alert_last_box = None
        self.in_detection_fail_mode = False
        self.detection_fail_start_time = None
        self.consecutive_alert_count = 0
        print("Detection state reset - starting fresh scan")
    
    def run_detection_loop(self, window_name="Face Recognition"):
        self.is_running = True
        david_recognized_this_cycle = False
        
        self.reset_detection_state()
        
        while self.is_running:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            display_frame = frame.copy()
            
            # Check if welcome sequence is running - PAUSE DETECTION
            if self.welcome_sequence_running:
                self.draw_welcome_sequence_message(display_frame)
                cv2.putText(display_frame, "SEQUENCE RUNNING - Detection Paused", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                
                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue
            
            if self.in_detection_fail_mode:
                elapsed = (datetime.now() - self.detection_fail_start_time).total_seconds()
                remaining = self.detection_fail_duration - elapsed
                
                if remaining > 0:
                    self.draw_detection_fail_message(display_frame, remaining)
                    cv2.putText(display_frame, f"DETECTION FAIL - Restarting in {int(remaining)}s", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    print("Detection fail period ended - restarting detection...")
                    self.reset_detection_state()
            
            elif self.in_welcome_mode:
                elapsed = (datetime.now() - self.welcome_start_time).total_seconds()
                remaining = self.welcome_duration - elapsed
                
                if remaining > 0:
                    face_box = self.welcome_last_box or (160, 120, 480, 360)
                    self.draw_display_message(display_frame, face_box, remaining, 
                                             "Welcome, Mr. David", "WELCOME!", (255, 0, 0))
                    cv2.putText(display_frame, f"WELCOME MODE - {int(remaining)}s", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(display_frame, f"Alert Count: 0/{self.max_consecutive_alerts}", 
                               (10, display_frame.shape[0] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    self.in_welcome_mode = False
                    self.welcome_start_time = None
                    self.welcome_last_box = None
                    self.consecutive_alert_count = 0
                    if self.on_david_recognized and david_recognized_this_cycle:
                        david_recognized_this_cycle = False
                        threading.Thread(target=self.on_david_recognized, daemon=True).start()
            
            elif self.in_alert_mode:
                elapsed = (datetime.now() - self.alert_start_time).total_seconds()
                remaining = self.alert_duration - elapsed
                
                if remaining > 0:
                    face_box = self.alert_last_box or (160, 120, 480, 360)
                    self.draw_display_message(display_frame, face_box, remaining, 
                                             f"Unknown ({self.consecutive_alert_count}/{self.max_consecutive_alerts})", 
                                             "ALERT!", (0, 0, 255))
                    cv2.putText(display_frame, f"ALERT MODE - {int(remaining)}s", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(display_frame, f"Alert Count: {self.consecutive_alert_count}/{self.max_consecutive_alerts}", 
                               (10, display_frame.shape[0] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    self.in_alert_mode = False
                    self.alert_start_time = None
                    self.alert_last_box = None
                    
                    if self.consecutive_alert_count >= self.max_consecutive_alerts:
                        print(f"Maximum consecutive alerts ({self.max_consecutive_alerts}) reached!")
                        print("Entering Detection Fail mode...")
                        self.in_detection_fail_mode = True
                        self.detection_fail_start_time = datetime.now()
            else:
                faces = self.detect_faces(frame)
                
                cv2.putText(display_frame, f"Alert Count: {self.consecutive_alert_count}/{self.max_consecutive_alerts}", 
                           (10, display_frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                
                for face_box in faces:
                    x1, y1, x2, y2 = face_box
                    face_img = frame[y1:y2, x1:x2]
                    if face_img.size == 0:
                        continue
                    
                    is_known, name, confidence = self.recognize_face(face_img)
                    
                    if is_known and name == self.known_face_name:
                        self.in_welcome_mode = True
                        self.welcome_start_time = datetime.now()
                        self.welcome_last_box = face_box
                        david_recognized_this_cycle = True
                        self.consecutive_alert_count = 0
                        print(f"Mr. David recognized! Alert counter reset to 0")
                        self.draw_display_message(display_frame, face_box, self.welcome_duration, 
                                                 "Welcome, Mr. David", "WELCOME!", (255, 0, 0))
                        break
                    else:
                        self.consecutive_alert_count += 1
                        print(f"Unknown face detected! Alert count: {self.consecutive_alert_count}/{self.max_consecutive_alerts}")
                        
                        self.in_alert_mode = True
                        self.alert_start_time = datetime.now()
                        self.alert_last_box = face_box
                        self.draw_display_message(display_frame, face_box, self.alert_duration, 
                                                 f"Unknown ({self.consecutive_alert_count}/{self.max_consecutive_alerts})", 
                                                 "ALERT!", (0, 0, 255))
                        break
                
                if not self.in_welcome_mode and not self.in_alert_mode and not self.in_detection_fail_mode:
                    cv2.putText(display_frame, f"Scanning... Faces: {len(faces)}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("Manual reset triggered")
                self.reset_detection_state()
        
        self.stop()


class StyledButton(tk.Button):
    """Custom styled button with hover effects"""
    
    def __init__(self, master, text, command, bg_color, hover_color, fg_color="#ffffff", 
                 width=15, height=2, font_size=11, **kwargs):
        
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.fg_color = fg_color
        
        super().__init__(
            master,
            text=text,
            command=command,
            bg=bg_color,
            fg=fg_color,
            activebackground=hover_color,
            activeforeground=fg_color,
            relief="flat",
            cursor="hand2",
            width=width,
            height=height,
            font=("Segoe UI", font_size, "bold"),
            bd=0,
            **kwargs
        )
        
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
    
    def _on_enter(self, e):
        if self['state'] != 'disabled':
            self['bg'] = self.hover_color
    
    def _on_leave(self, e):
        if self['state'] != 'disabled':
            self['bg'] = self.bg_color


class IntegratedRobotController:
    """
    Integrated GUI controller for Robot with Face Recognition
    Enhanced version with improved visuals and Raspberry Pi 5 UART support
    """
    
    DEFAULT_PORT = "/dev/ttyAMA2"
    DEFAULT_BAUD = 115200
    
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 Face Recognition Robot Controller - STM32F103")
        self.root.geometry("1600x1000")
        self.root.resizable(True, True)
        self.root.configure(bg=ColorScheme.BG_DARK)
        
        self.serial_port = None
        self.is_connected = False
        self.stm32_detected = False
        
        self.motor_controller = ServoMotorController()
        
        self.face_system = None
        self.face_detection_running = False
        self.face_detection_thread = None
        
        self.current_angles = {i: 0 for i in range(1, 17)}
        self.current_colors = {i: "BLUE" for i in range(1, 17)}
        
        self.motor_widgets = {}
        self.continuous_detection = False
        self.log_text = None
        
        self.title_font = tkfont.Font(family="Segoe UI", size=14, weight="bold")
        self.label_font = tkfont.Font(family="Segoe UI", size=11)
        self.button_font = tkfont.Font(family="Segoe UI", size=10, weight="bold")
        
        self.create_widgets()
        self.update_connection_status()
        self.refresh_ports_silent()
    
    def create_styled_frame(self, parent, text, row, col, colspan=1, rowspan=1):
        """Create a styled labeled frame"""
        frame = tk.LabelFrame(
            parent,
            text=f"  {text}  ",
            bg=ColorScheme.BG_MEDIUM,
            fg=ColorScheme.TEXT_ACCENT,
            font=self.title_font,
            relief="ridge",
            bd=2,
            padx=10,
            pady=10
        )
        frame.grid(row=row, column=col, columnspan=colspan, rowspan=rowspan, 
                  padx=8, pady=8, sticky="nsew")
        return frame
    
    def create_widgets(self):
        """Create all GUI widgets with enhanced styling"""
        
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(3, weight=1)
        
        main_frame = tk.Frame(self.root, bg=ColorScheme.BG_DARK, padx=15, pady=15)
        main_frame.pack(fill="both", expand=True)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        
        # Header
        header_frame = tk.Frame(main_frame, bg=ColorScheme.BG_ACCENT, height=60)
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 15))
        header_frame.grid_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="🤖 HUMANOID ROBOT FACE RECOGNITION",
            bg=ColorScheme.BG_ACCENT,
            fg=ColorScheme.TEXT_PRIMARY,
            font=("Segoe UI", 18, "bold")
        )
        title_label.pack(expand=True)
        
        # Face Recognition Section
        face_frame = self.create_styled_frame(main_frame, "👤 FACE RECOGNITION CONTROL", 1, 0, 2)
        
        path_frame = tk.Frame(face_frame, bg=ColorScheme.BG_MEDIUM)
        path_frame.pack(fill="x", pady=5)
        
        tk.Label(path_frame, text="Detection Model:", bg=ColorScheme.BG_MEDIUM, 
                fg=ColorScheme.TEXT_SECONDARY, font=self.label_font).grid(row=0, column=0, padx=5, sticky="w")
        self.detection_model_var = tk.StringVar(value="blazeface.tflite")
        tk.Entry(path_frame, textvariable=self.detection_model_var, width=25, 
                font=self.label_font, bg=ColorScheme.BG_LIGHT, fg=ColorScheme.TEXT_PRIMARY,
                insertbackground=ColorScheme.TEXT_PRIMARY).grid(row=0, column=1, padx=5)
        
        tk.Label(path_frame, text="Recognition Model:", bg=ColorScheme.BG_MEDIUM, 
                fg=ColorScheme.TEXT_SECONDARY, font=self.label_font).grid(row=0, column=2, padx=5, sticky="w")
        self.recognition_model_var = tk.StringVar(value="mobilefacenet.tflite")
        tk.Entry(path_frame, textvariable=self.recognition_model_var, width=25,
                font=self.label_font, bg=ColorScheme.BG_LIGHT, fg=ColorScheme.TEXT_PRIMARY,
                insertbackground=ColorScheme.TEXT_PRIMARY).grid(row=0, column=3, padx=5)
        
        tk.Label(path_frame, text="Training Folder:", bg=ColorScheme.BG_MEDIUM, 
                fg=ColorScheme.TEXT_SECONDARY, font=self.label_font).grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.training_folder_var = tk.StringVar(value="mrdavid")
        tk.Entry(path_frame, textvariable=self.training_folder_var, width=25,
                font=self.label_font, bg=ColorScheme.BG_LIGHT, fg=ColorScheme.TEXT_PRIMARY,
                insertbackground=ColorScheme.TEXT_PRIMARY).grid(row=1, column=1, padx=5, pady=5)
        
        StyledButton(path_frame, "📁 Browse", self.browse_training_folder,
                    ColorScheme.BTN_SECONDARY, ColorScheme.BG_LIGHT, width=10, height=1,
                    font_size=10).grid(row=1, column=2, padx=5, pady=5)
        
        face_btn_frame = tk.Frame(face_frame, bg=ColorScheme.BG_MEDIUM)
        face_btn_frame.pack(fill="x", pady=15)
        
        self.start_face_btn = StyledButton(
            face_btn_frame,
            "🎥  START FACE DETECTION  🎥",
            self.start_face_detection,
            ColorScheme.BTN_PRIMARY,
            ColorScheme.BTN_PRIMARY_HOVER,
            width=35,
            height=3,
            font_size=16
        )
        self.start_face_btn.pack(side="left", padx=20)
        
        self.stop_face_btn = StyledButton(
            face_btn_frame,
            "⏹  STOP DETECTION",
            self.stop_face_detection,
            ColorScheme.BTN_DANGER,
            "#3498db",
            width=20,
            height=3,
            font_size=12
        )
        self.stop_face_btn.pack(side="left", padx=10)
        self.stop_face_btn.config(state="disabled")
        
        self.continuous_var = tk.BooleanVar(value=True)
        continuous_cb = tk.Checkbutton(
            face_btn_frame,
            text="🔄 Continuous Detection Loop",
            variable=self.continuous_var,
            bg=ColorScheme.BG_MEDIUM,
            fg=ColorScheme.TEXT_ACCENT,
            selectcolor=ColorScheme.BG_LIGHT,
            activebackground=ColorScheme.BG_MEDIUM,
            font=self.label_font
        )
        continuous_cb.pack(side="left", padx=20)
        
        self.face_status = tk.Label(
            face_frame,
            text="● Face Detection: NOT RUNNING",
            bg=ColorScheme.BG_MEDIUM,
            fg=ColorScheme.STATUS_DISCONNECTED,
            font=("Segoe UI", 12, "bold")
        )
        self.face_status.pack(pady=5)
        
        # Connection Section
        conn_frame = self.create_styled_frame(main_frame, "🔌 UART CONNECTION (Raspberry Pi 5 → STM32F103)", 2, 0, 2)
        
        conn_inner = tk.Frame(conn_frame, bg=ColorScheme.BG_MEDIUM)
        conn_inner.pack(fill="x", pady=5)
        
        tk.Label(conn_inner, text="UART Port:", bg=ColorScheme.BG_MEDIUM, 
                fg=ColorScheme.TEXT_SECONDARY, font=self.label_font).grid(row=0, column=0, padx=5)
        
        self.port_var = tk.StringVar(value=self.DEFAULT_PORT)
        self.port_combo = ttk.Combobox(conn_inner, textvariable=self.port_var, width=18, font=self.label_font)
        self.port_combo.grid(row=0, column=1, padx=5)
        
        tk.Label(conn_inner, text="Baud Rate:", bg=ColorScheme.BG_MEDIUM, 
                fg=ColorScheme.TEXT_SECONDARY, font=self.label_font).grid(row=0, column=2, padx=5)
        
        self.baud_var = tk.StringVar(value=str(self.DEFAULT_BAUD))
        baud_combo = ttk.Combobox(conn_inner, textvariable=self.baud_var, width=18, font=self.label_font)
        baud_combo['values'] = ('9600', '19200', '38400', '57600', '115200')
        baud_combo.grid(row=0, column=3, padx=5)
        
        self.connect_btn = StyledButton(
            conn_inner, "🔗 CONNECT", self.connect_serial,
            ColorScheme.BTN_SUCCESS, "#00ff88", width=14, height=2, font_size=11
        )
        self.connect_btn.grid(row=0, column=4, padx=10)
        
        self.disconnect_btn = StyledButton(
            conn_inner, "🔌 DISCONNECT", self.disconnect_serial,
            ColorScheme.BTN_DANGER, "#ff6b81", width=14, height=2, font_size=11
        )
        self.disconnect_btn.grid(row=0, column=5, padx=5)
        self.disconnect_btn.config(state="disabled")
        
        StyledButton(
            conn_inner, "🔄 Refresh", self.refresh_ports,
            ColorScheme.BTN_WARNING, "#3498db", width=14, height=2, font_size=10
        ).grid(row=0, column=6, padx=10)
        
        status_frame = tk.Frame(conn_frame, bg=ColorScheme.BG_MEDIUM)
        status_frame.pack(fill="x", pady=10)
        
        self.status_label = tk.Label(
            status_frame,
            text="● DISCONNECTED",
            bg=ColorScheme.BG_MEDIUM,
            fg=ColorScheme.STATUS_DISCONNECTED,
            font=("Segoe UI", 14, "bold")
        )
        self.status_label.pack(side="left", padx=20)
        
        self.status_detail = tk.Label(
            status_frame,
            text="Port: -- | Baud: -- | STM32: Not Detected",
            bg=ColorScheme.BG_MEDIUM,
            fg=ColorScheme.TEXT_SECONDARY,
            font=self.label_font
        )
        self.status_detail.pack(side="left", padx=20)
        
        # Global Controls
        global_frame = self.create_styled_frame(main_frame, "🎮 GLOBAL CONTROLS", 3, 0)
        
        angle_label = tk.Label(global_frame, text="⚙️ ANGLE CONTROL", bg=ColorScheme.BG_MEDIUM,
                              fg=ColorScheme.TEXT_ACCENT, font=self.title_font)
        angle_label.pack(pady=5)
        
        angle_btns = tk.Frame(global_frame, bg=ColorScheme.BG_MEDIUM)
        angle_btns.pack(pady=5)
        
        StyledButton(angle_btns, "All → -5°", lambda: self.set_all_angles(-5),
                    "#3498db", "#3498db", width=7, height=2).pack(side="left", padx=5)
        StyledButton(angle_btns, "All → 0°", lambda: self.set_all_angles(0),
                    "#3498db", "#5dade2", width=7, height=2).pack(side="left", padx=5)
        StyledButton(angle_btns, "All → 5°", lambda: self.set_all_angles(5),
                    "#3498db", "#3498db", width=7, height=2).pack(side="left", padx=5)
        
        color_label = tk.Label(global_frame, text="🎨 COLOR CONTROL", bg=ColorScheme.BG_MEDIUM,
                              fg=ColorScheme.TEXT_ACCENT, font=self.title_font)
        color_label.pack(pady=(15, 5))
        
        color_btns = tk.Frame(global_frame, bg=ColorScheme.BG_MEDIUM)
        color_btns.pack(pady=5)
        
        StyledButton(color_btns, "🔴 RED", lambda: self.set_all_colors("RED"),
                    "#e74c3c", "#ff6b6b", width=7, height=2).pack(side="left", padx=5)
        StyledButton(color_btns, "🟢 GREEN", lambda: self.set_all_colors("GREEN"),
                    "#27ae60", "#2ecc71", width=7, height=2).pack(side="left", padx=5)
        StyledButton(color_btns, "🔵 BLUE", lambda: self.set_all_colors("BLUE"),
                    "#3498db", "#5dade2", width=7, height=2).pack(side="left", padx=5)
        
        pattern_label = tk.Label(global_frame, text="✨ PATTERNS & SEQUENCES", bg=ColorScheme.BG_MEDIUM,
                                fg=ColorScheme.TEXT_ACCENT, font=self.title_font)
        pattern_label.pack(pady=(15, 5))
        
        pattern_btns = tk.Frame(global_frame, bg=ColorScheme.BG_MEDIUM)
        pattern_btns.pack(pady=5)
        
        StyledButton(pattern_btns, "🌈 Rainbow", self.rainbow_wave,
                    "#9b59b6", "#a569bd", width=7, height=2).pack(side="left", padx=5)
        StyledButton(pattern_btns, "🔄 Cycle", self.color_cycle,
                    "#9b59b6", "#a569bd", width=7, height=2).pack(side="left", padx=5)
        StyledButton(pattern_btns, "👋 Welcome", self.run_welcome_sequence,
                    "#9b59b6", "#a569bd", width=7, height=2).pack(side="left", padx=5)
        
        resetMotor_label = tk.Label(global_frame, text="⚙️ RESET MOTOR", bg=ColorScheme.BG_MEDIUM,
                              fg=ColorScheme.TEXT_ACCENT, font=self.title_font)
        resetMotor_label.pack(pady=5)
        
        resetMotor_btns = tk.Frame(global_frame, bg=ColorScheme.BG_MEDIUM)
        resetMotor_btns.pack(pady=5)

        StyledButton(resetMotor_btns, "⏹️ Reset All", self.reset_all,
                    "#7f8c8d", "#95a5a6", width=29, height=2).pack(side="left", padx=3)
        
        # Motor Controls
        motors_frame = self.create_styled_frame(main_frame, "🤖 INDIVIDUAL MOTOR CONTROLS (1-16)", 3, 1)
        
        motors_container = tk.Frame(motors_frame, bg=ColorScheme.BG_MEDIUM)
        motors_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        for i in range(16):
            motor_num = i + 1
            row = i // 8
            col = i % 8
            self.create_motor_widget(motors_container, motor_num, row, col)
        
        for i in range(4):
            motors_container.grid_columnconfigure(i, weight=1)
            motors_container.grid_rowconfigure(i, weight=1)
        
        # Log Section
        log_frame = self.create_styled_frame(main_frame, "📜 COMMAND LOG", 4, 0, 2)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame, width=120, height=8, wrap=tk.WORD,
            bg=ColorScheme.BG_LIGHT, fg=ColorScheme.TEXT_PRIMARY,
            font=("Consolas", 10), insertbackground=ColorScheme.TEXT_PRIMARY
        )
        self.log_text.pack(fill="both", expand=True, pady=5)
        
        StyledButton(log_frame, "🗑️ Clear Log", self.clear_log,
                    ColorScheme.BTN_SECONDARY, ColorScheme.BG_LIGHT, width=15, height=1).pack(pady=5)
    
    def create_motor_widget(self, parent, motor_num, row, col):
        """Create individual motor control widget with distinct colors"""
        motor_color = ColorScheme.MOTOR_COLORS[motor_num]
        
        motor_frame = tk.Frame(
            parent,
            bg=ColorScheme.BG_LIGHT,
            relief="raised",
            bd=3,
            highlightbackground=motor_color,
            highlightthickness=3
        )
        motor_frame.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")
        
        header = tk.Label(
            motor_frame,
            text=f"  MOTOR {motor_num}  ",
            bg=motor_color,
            fg="#ffffff",
            font=("Segoe UI", 11, "bold")
        )
        header.pack(fill="x", pady=(0, 5))
        
        status_frame = tk.Frame(motor_frame, bg=ColorScheme.RGB_COLORS["BLUE"], 
                               height=35, relief="sunken", bd=2)
        status_frame.pack(fill="x", padx=5, pady=2)
        status_frame.pack_propagate(False)
        
        angle_label = tk.Label(status_frame, text="0°", font=("Segoe UI", 12, "bold"),
                              bg=ColorScheme.RGB_COLORS["BLUE"], fg="#ffffff")
        angle_label.pack(expand=True)
        
        angle_frame = tk.Frame(motor_frame, bg=ColorScheme.BG_LIGHT)
        angle_frame.pack(fill="x", padx=3, pady=2)
        
        angle_entry = tk.Entry(angle_frame, width=3, font=("Segoe UI", 12, "bold"),
                              bg=ColorScheme.BG_MEDIUM, fg=ColorScheme.TEXT_PRIMARY,
                              insertbackground=ColorScheme.TEXT_PRIMARY, justify="center")
        angle_entry.pack(side="left", padx=3)
        angle_entry.insert(0, "0")
        
        set_btn = tk.Button(
            angle_frame, text="SET", font=("Segoe UI", 12, "bold"),
            bg=motor_color, fg="#ffffff", width=1, relief="flat",
            command=lambda m=motor_num: self.set_motor_angle(m),
            cursor="hand2"
        )
        set_btn.pack(side="left", padx=1, pady=2)
        
        quick_frame = tk.Frame(motor_frame, bg=ColorScheme.BG_LIGHT)
        quick_frame.pack(fill="x", padx=3, pady=2)
        
        for angle, color in [(-40, "#7f8c8d"), (0, "#7f8c8d"), (40, "#7f8c8d")]:
            btn = tk.Button(
                quick_frame, text=str(angle), font=("Segoe UI", 12, "bold"),
                bg=color, fg="#ffffff", width=1, relief="flat",
                command=lambda m=motor_num, a=angle: self.quick_angle(m, a),
                cursor="hand2"
            )
            btn.pack(side="left", padx=1)
        
        rgb_frame = tk.Frame(motor_frame, bg=ColorScheme.BG_LIGHT)
        rgb_frame.pack(fill="x", padx=3, pady=2)
        
        for color, emoji, bg_color in [("RED", "🔴", "#e74c3c"), 
                                        ("GREEN", "🟢", "#27ae60"), 
                                        ("BLUE", "🔵", "#3498db")]:
            btn = tk.Button(
                rgb_frame, text=emoji, font=("Segoe UI", 12),
                bg=bg_color, fg="#ffffff", width=1, relief="flat",
                command=lambda m=motor_num, c=color: self.set_motor_color(m, c),
                cursor="hand2"
            )
            btn.pack(side="left", padx=1)
        
        self.motor_widgets[motor_num] = {
            'status_frame': status_frame,
            'angle_label': angle_label,
            'angle_entry': angle_entry,
            'motor_frame': motor_frame,
            'motor_color': motor_color
        }
    
    def browse_training_folder(self):
        folder = filedialog.askdirectory(title="Select Training Images Folder")
        if folder:
            self.training_folder_var.set(folder)
            self.log_message(f"📁 Training folder: {folder}")
    
    def refresh_ports_silent(self):
        """Refresh available serial ports without logging"""
        ports = ["/dev/ttyAMA2", "/dev/ttyAMA0", "/dev/ttyUSB0", "/dev/ttyUSB1", 
                 "COM1", "COM2", "COM3", "COM4", "COM5"]
        system_ports = [p.device for p in serial.tools.list_ports.comports()]
        all_ports = list(set(ports + system_ports))
        self.port_combo['values'] = all_ports if all_ports else [self.DEFAULT_PORT]
    
    def refresh_ports(self):
        """Refresh available serial ports with logging"""
        self.refresh_ports_silent()
        self.log_message("🔄 Serial ports refreshed")
    
    def verify_stm32_connection(self):
        """Verify STM32F103 is detected and responding"""
        if not self.serial_port or not self.serial_port.is_open:
            return False
        
        try:
            self.serial_port.reset_input_buffer()
            self.serial_port.reset_output_buffer()
            
            test_cmd = bytes.fromhex("FFFF01070DE00D")
            self.serial_port.write(test_cmd)
            time.sleep(0.1)
            
            return True
        except Exception as e:
            self.log_message(f"⚠️ STM32 verification failed: {str(e)}")
            return False
    
    def update_connection_status(self):
        """Update the connection status display"""
        port = self.port_var.get()
        baud = self.baud_var.get()
        
        if self.is_connected and self.stm32_detected:
            self.status_label.config(text="● CONNECTED", fg=ColorScheme.STATUS_CONNECTED)
            self.status_detail.config(
                text=f"Port: {port} | Baud: {baud} | STM32: Detected ✓",
                fg=ColorScheme.STATUS_CONNECTED
            )
            self.disconnect_btn.config(state="normal")
            self.connect_btn.config(state="disabled")
        else:
            self.status_label.config(text="● DISCONNECTED", fg=ColorScheme.STATUS_DISCONNECTED)
            if self.is_connected and not self.stm32_detected:
                detail_text = f"Port: {port} | Baud: {baud} | STM32: NOT DETECTED ✗"
                self.status_detail.config(text=detail_text, fg=ColorScheme.STATUS_WARNING)
            else:
                self.status_detail.config(
                    text="Port: -- | Baud: -- | STM32: Not Detected",
                    fg=ColorScheme.TEXT_SECONDARY
                )
            self.disconnect_btn.config(state="disabled")
            self.connect_btn.config(state="normal")
    
    def connect_serial(self):
        """Connect to serial port with validation"""
        port = self.port_var.get()
        baud = self.baud_var.get()
        
        if port != self.DEFAULT_PORT:
            self.log_message(f"⚠️ Warning: Recommended port is {self.DEFAULT_PORT}")
        
        if baud != str(self.DEFAULT_BAUD):
            self.log_message(f"⚠️ Warning: Recommended baud rate is {self.DEFAULT_BAUD}")
        
        try:
            self.serial_port = serial.Serial(
                port=port,
                baudrate=int(baud),
                timeout=1,
                write_timeout=1
            )
            time.sleep(2)
            
            self.is_connected = True
            self.log_message(f"🔗 Serial port {port} opened at {baud} baud")
            
            self.stm32_detected = self.verify_stm32_connection()
            
            if self.stm32_detected:
                self.log_message("✅ STM32F103 detected and responding!")
                self.motor_controller.serial_port = self.serial_port
                self.motor_controller.is_connected = True
            else:
                self.log_message("⚠️ STM32F103 NOT DETECTED - Check power and connections")
                self.log_message("   Disconnect button will remain inactive until STM32 is detected")
                self.serial_port.close()
                self.is_connected = False
                self.serial_port = None
            
            self.update_connection_status()
            
        except serial.SerialException as e:
            self.log_message(f"✗ Serial error: {str(e)}")
            self.is_connected = False
            self.stm32_detected = False
            self.update_connection_status()
            messagebox.showerror("Connection Error", 
                               f"Failed to connect to {port}\n\nError: {str(e)}\n\n"
                               f"Please check:\n"
                               f"1. UART port is {self.DEFAULT_PORT}\n"
                               f"2. Baud rate is {self.DEFAULT_BAUD}\n"
                               f"3. STM32F103 is powered on")
        except Exception as e:
            self.log_message(f"✗ Connection failed: {str(e)}")
            self.is_connected = False
            self.stm32_detected = False
            self.update_connection_status()
    
    def disconnect_serial(self):
        """Disconnect from serial port"""
        try:
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.close()
            
            self.is_connected = False
            self.stm32_detected = False
            self.serial_port = None
            self.motor_controller.is_connected = False
            self.motor_controller.serial_port = None
            
            self.log_message("🔌 Disconnected from serial port")
            self.update_connection_status()
            
        except Exception as e:
            self.log_message(f"✗ Disconnect error: {str(e)}")
    
    def send_hex_command(self, hex_string, description=""):
        """Send hex command via serial"""
        if not self.is_connected or not self.stm32_detected:
            self.log_message("⚠️ Not connected or STM32 not detected")
            return False
        
        try:
            hex_bytes = bytes.fromhex(hex_string)
            self.serial_port.write(hex_bytes)
            self.log_message(f"→ {description}: {hex_string}")
            return True
        except Exception as e:
            self.log_message(f"✗ Send error: {str(e)}")
            return False
    
    def set_motor_angle(self, motor_num):
        """Set motor angle from entry"""
        try:
            entry = self.motor_widgets[motor_num]['angle_entry']
            angle = int(entry.get().strip())
            
            if angle < -160 or angle > 160:
                messagebox.showerror("Invalid Range", "Angle must be between -160 and 160°")
                return
            
            angle_str = str(angle)
            if angle_str in ServoMotorProtocols.ANGLE_PROTOCOLS[motor_num]:
                hex_cmd = ServoMotorProtocols.ANGLE_PROTOCOLS[motor_num][angle_str]
                if self.send_hex_command(hex_cmd, f"Motor {motor_num} → {angle}°"):
                    self.current_angles[motor_num] = angle
                    self.update_motor_display(motor_num)
            else:
                available = [int(a) for a in ServoMotorProtocols.ANGLE_PROTOCOLS[motor_num].keys()]
                nearest = min(available, key=lambda x: abs(x - angle))
                hex_cmd = ServoMotorProtocols.ANGLE_PROTOCOLS[motor_num][str(nearest)]
                if self.send_hex_command(hex_cmd, f"Motor {motor_num} → {nearest}° (nearest)"):
                    self.current_angles[motor_num] = nearest
                    self.update_motor_display(motor_num)
                
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid integer")
    
    def quick_angle(self, motor_num, angle):
        """Quick angle preset"""
        entry = self.motor_widgets[motor_num]['angle_entry']
        entry.delete(0, tk.END)
        entry.insert(0, str(angle))
        self.set_motor_angle(motor_num)
    
    def set_motor_color(self, motor_num, color):
        """Set motor RGB color"""
        hex_cmd = ServoMotorProtocols.RGB_PROTOCOLS[motor_num][color]
        if self.send_hex_command(hex_cmd, f"Motor {motor_num} → {color}"):
            self.current_colors[motor_num] = color
            self.update_motor_display(motor_num)
    
    def update_motor_display(self, motor_num):
        """Update motor status display"""
        widgets = self.motor_widgets[motor_num]
        angle = self.current_angles[motor_num]
        color = self.current_colors[motor_num]
        
        bg_color = ColorScheme.RGB_COLORS.get(color, "#FFFFFF")
        widgets['status_frame'].config(bg=bg_color)
        widgets['angle_label'].config(text=f"{angle}°", bg=bg_color)
    
    def set_all_angles(self, angle, delay=0.05):
        """Set all motors to same angle"""
        for motor_num in range(1, 17):
            entry = self.motor_widgets[motor_num]['angle_entry']
            entry.delete(0, tk.END)
            entry.insert(0, str(angle))
            self.set_motor_angle(motor_num)
            time.sleep(delay)
    
    def set_all_colors(self, color, delay=0.05):
        """Set all motors to same color"""
        for motor_num in range(1, 17):
            self.set_motor_color(motor_num, color)
            time.sleep(delay)
    
    def reset_all(self):
        """Reset all motors"""
        self.set_all_angles(0)
        self.log_message("⏹️ All motors reset to 0°")
    
    def rainbow_wave(self):
        """Rainbow wave effect"""
        def wave():
            colors = ["RED", "GREEN", "BLUE"]
            for cycle in range(2):
                for offset in range(3):
                    for i in range(16):
                        self.set_motor_color(i + 1, colors[(i + offset) % 3])
                    time.sleep(0.5)
            self.log_message("🌈 Rainbow wave completed!")
        threading.Thread(target=wave, daemon=True).start()
    
    def color_cycle(self):
        """Color cycle"""
        def cycle():
            for color in ["RED", "GREEN", "BLUE"]:
                self.set_all_colors(color)
                time.sleep(1)
            self.log_message("🔄 Color cycle completed!")
        threading.Thread(target=cycle, daemon=True).start()
    
    def update_sequence_status(self, status):
        """Update the welcome sequence status for display in detection window"""
        if self.face_system:
            self.face_system.welcome_sequence_status = status
    
    def run_welcome_sequence(self):
        """Run welcome sequence for Mr. David - blocks detection until complete"""
        def sequence():
            # Set flag to pause detection
            if self.face_system:
                self.face_system.welcome_sequence_running = True
                self.face_system.welcome_sequence_status = "Starting..."
            
            self.log_message("🎉 Starting Welcome Sequence! (Detection paused)")
            
            try:
                # Set all RGB to BLUE
                self.update_sequence_status("Setting BLUE LEDs...")
                self.log_message("Setting BLUE...")
                for m in range(1, 17):
                    self.set_motor_color(m, "BLUE")
                    time.sleep(0.03)
                time.sleep(0.5)
                
                # Motion Process 1
                self.update_sequence_status("Motion Process 1/7...")
                self.log_message("Motion Process 1...")
                motion_1 = {1: -1, 2: 0, 3: -1, 4: -2, 5: -1, 6: -1, 7: -9, 8: -4,
                           9: -1, 10: 0, 11: 1, 12: 9, 13: 0, 14: -1, 15: 0, 16: -2}
                for motor, angle in motion_1.items():
                    entry = self.motor_widgets[motor]['angle_entry']
                    entry.delete(0, tk.END)
                    entry.insert(0, str(angle))
                    self.set_motor_angle(motor)
                    time.sleep(0.03)
                time.sleep(0.5)
                
                # Motion Process 2
                self.update_sequence_status("Motion Process 2/7...")
                self.log_message("Motion Process 2...")
                motion_2 = {1: -1, 2: -45, 3: -1, 4: -3, 5: 44, 6: -1, 7: -9, 8: -4,
                           9: -1, 10: 0, 11: 2, 12: 9, 13: 0, 14: -1, 15: 0, 16: -2}
                for motor, angle in motion_2.items():
                    entry = self.motor_widgets[motor]['angle_entry']
                    entry.delete(0, tk.END)
                    entry.insert(0, str(angle))
                    self.set_motor_angle(motor)
                    time.sleep(0.03)
                time.sleep(0.5)
                
                # Motion Process 3
                self.update_sequence_status("Motion Process 3/7...")
                self.log_message("Motion Process 3...")
                motion_3 = {1: -90, 2: -45, 3: 0, 4: 89, 5: 44, 6: 0, 7: -10, 8: -4,
                           9: -2, 10: 0, 11: 3, 12: 10, 13: -1, 14: -1, 15: 0, 16: -3}
                for motor, angle in motion_3.items():
                    entry = self.motor_widgets[motor]['angle_entry']
                    entry.delete(0, tk.END)
                    entry.insert(0, str(angle))
                    self.set_motor_angle(motor)
                    time.sleep(0.03)
                time.sleep(0.5)

                # Motion Process 4
                self.update_sequence_status("Motion Process 4/7...")
                self.log_message("Motion Process 4...")
                motion_4 = {1: -90, 2: -45, 3: 90, 4: 88, 5: 44, 6: -90, 7: -10, 8: -4,
                           9: -3, 10: 0, 11: 4, 12: 10, 13: -1, 14: -1, 15: 0, 16: -5}
                for motor, angle in motion_4.items():
                    entry = self.motor_widgets[motor]['angle_entry']
                    entry.delete(0, tk.END)
                    entry.insert(0, str(angle))
                    self.set_motor_angle(motor)
                    time.sleep(0.03)
                time.sleep(0.5)

                # Motion Process 5
                self.update_sequence_status("Motion Process 5/7...")
                self.log_message("Motion Process 5...")
                motion_5 = {1: -90, 2: -45, 3: 90, 4: 88, 5: 44, 6: -90, 7: -10, 8: -10,
                           9: -3, 10: 0, 11: 4, 12: 10, 13: 10, 14: -1, 15: 0, 16: -5}
                for motor, angle in motion_5.items():
                    entry = self.motor_widgets[motor]['angle_entry']
                    entry.delete(0, tk.END)
                    entry.insert(0, str(angle))
                    self.set_motor_angle(motor)
                    time.sleep(0.03)
                time.sleep(0.5)

                # Motion Process 6
                self.update_sequence_status("Motion Process 6/7...")
                self.log_message("Motion Process 6...")
                motion_6 = {1: -90, 2: -45, 3: 90, 4: 87, 5: 43, 6: -90, 7: -10, 8: 5,
                           9: -4, 10: 0, 11: 4, 12: 9, 13: -7, 14: 0, 15: -2, 16: -6}
                for motor, angle in motion_6.items():
                    entry = self.motor_widgets[motor]['angle_entry']
                    entry.delete(0, tk.END)
                    entry.insert(0, str(angle))
                    self.set_motor_angle(motor)
                    time.sleep(0.03)
                time.sleep(0.5)

                # Motion Process 7
                self.update_sequence_status("Motion Process 7/7...")
                self.log_message("Motion Process 7...")
                motion_7 = {1: -90, 2: -45, 3: 90, 4: 87, 5: 43, 6: -90, 7: -5, 8: 5,
                           9: -4, 10: 0, 11: 4, 12: 0, 13: -7, 14: 0, 15: -2, 16: -6}
                for motor, angle in motion_7.items():
                    entry = self.motor_widgets[motor]['angle_entry']
                    entry.delete(0, tk.END)
                    entry.insert(0, str(angle))
                    self.set_motor_angle(motor)
                    time.sleep(0.03)
                time.sleep(0.03)
                
                # Rainbow wave
                self.update_sequence_status("Rainbow wave effect...")
                self.log_message("Rainbow wave (1 cycle)...")
                colors = ["RED", "GREEN", "BLUE"]
                for cycle in range(1):
                    for offset in range(3):
                        for i in range(16):
                            self.set_motor_color(i + 1, colors[(i + offset) % 3])
                        time.sleep(0.5)
                
                # BLUE for 1 second
                self.update_sequence_status("Setting BLUE (1 sec)...")
                self.log_message("Setting BLUE for 1 second...")
                for m in range(1, 17):
                    self.set_motor_color(m, "BLUE")
                    time.sleep(0.03)
                time.sleep(1)
                
                # Reset to 0° for 3 seconds
                self.update_sequence_status("Resetting to 0° (3 sec)...")
                self.log_message("Resetting to 0° for 3 seconds...")
                self.set_all_angles(0)
                time.sleep(3)
                
                self.log_message("✅ Welcome sequence completed!")
                
            finally:
                # Clear flag to resume detection
                if self.face_system:
                    self.face_system.welcome_sequence_running = False
                    self.face_system.welcome_sequence_status = ""
                self.log_message("🔄 Face detection resuming...")
        
        threading.Thread(target=sequence, daemon=True).start()
    
    def start_face_detection(self):
        """Start face detection"""
        if not TF_AVAILABLE:
            messagebox.showerror("Error", "TensorFlow required for face recognition")
            return
        
        detection_model = self.detection_model_var.get()
        recognition_model = self.recognition_model_var.get()
        training_folder = self.training_folder_var.get()
        
        for path, name in [(detection_model, "Detection model"), 
                           (recognition_model, "Recognition model"),
                           (training_folder, "Training folder")]:
            if not os.path.exists(path):
                messagebox.showerror("Error", f"{name} not found: {path}")
                return
        
        try:
            self.face_system = FaceRecognitionSystem(
                detection_model, recognition_model, training_folder, debug_mode=False
            )
            self.face_system.initialize_camera(0)
            self.face_system.on_david_recognized = self.on_david_recognized
            
            self.face_detection_running = True
            self.face_status.config(text="● Face Detection: RUNNING", fg=ColorScheme.STATUS_CONNECTED)
            self.start_face_btn.config(state="disabled")
            self.stop_face_btn.config(state="normal")
            
            self.log_message("🎥 Face detection started!")
            
            def detection_loop():
                while self.face_detection_running:
                    self.face_system.run_detection_loop("Face Recognition - Press Q to close")
                    if self.continuous_var.get() and self.face_detection_running:
                        self.log_message("🔄 Restarting detection...")
                        if not self.face_system.camera or not self.face_system.camera.isOpened():
                            self.face_system.initialize_camera(0)
                    else:
                        break
                self.root.after(0, self.on_detection_stopped)
            
            self.face_detection_thread = threading.Thread(target=detection_loop, daemon=True)
            self.face_detection_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start: {str(e)}")
            self.log_message(f"✗ Face detection error: {str(e)}")
    
    def stop_face_detection(self):
        """Stop face detection"""
        self.face_detection_running = False
        if self.face_system:
            self.face_system.stop()
        self.log_message("⏹️ Face detection stopped")
    
    def on_detection_stopped(self):
        """Called when detection stops"""
        self.face_status.config(text="● Face Detection: STOPPED", fg=ColorScheme.STATUS_DISCONNECTED)
        self.start_face_btn.config(state="normal")
        self.stop_face_btn.config(state="disabled")
    
    def on_david_recognized(self):
        """Callback when Mr. David is recognized"""
        self.log_message("🎉 Mr. David recognized! Starting robot sequence...")
        if not self.is_connected or not self.stm32_detected:
            self.log_message("⚠️ Robot not connected - skipping sequence")
            if self.face_system:
                self.face_system.welcome_sequence_running = False
            return
        self.run_welcome_sequence()
    
    def log_message(self, message):
        """Add message to log"""
        if self.log_text is None:
            print(f"[LOG] {message}")
            return
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
    
    def clear_log(self):
        """Clear log"""
        if self.log_text:
            self.log_text.delete(1.0, tk.END)
    
    def on_closing(self):
        """Handle window close"""
        self.face_detection_running = False
        if self.face_system:
            self.face_system.stop()
        if self.is_connected:
            self.disconnect_serial()
        self.root.destroy()


def main():
    root = tk.Tk()
    
    style = ttk.Style()
    style.theme_use('clam')
    
    app = IntegratedRobotController(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
