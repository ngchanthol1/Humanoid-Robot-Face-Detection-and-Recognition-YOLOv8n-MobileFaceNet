"""
Face Recognition System for Raspberry Pi 5
Uses BlazeFace for detection and MobileFaceNet for recognition
"""

import cv2
import numpy as np
import tensorflow as tf
import os
from pathlib import Path
import pickle
import time
from datetime import datetime, timedelta

class FaceRecognitionSystem:
    def __init__(self, detection_model_path, recognition_model_path, known_faces_path, debug_mode=False):
        """
        Initialize the face recognition system
        
        Args:
            detection_model_path: Path to blazeface.tflite
            recognition_model_path: Path to mobilefacenet.tflite
            known_faces_path: Path to folder containing Mr. David's images
            debug_mode: If True, show detailed recognition information
        """
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
        
        # Initialize camera
        self.camera = None
        
        # Debug mode for detailed recognition info
        self.debug_mode = debug_mode
        
        # Storage for known face embeddings
        self.known_face_embeddings = []
        self.known_face_name = "Mr. David"
        
        # Recognition thresholds - STRICT settings to avoid false positives
        # Higher threshold = more strict = fewer false positives
        self.recognition_threshold = 0.75  # Minimum similarity to recognize (increased from 0.6)
        self.strict_threshold = 0.85  # High confidence threshold
        self.min_matching_embeddings = 4  # Must match at least 3 training images
        
        # Display mode settings - SEPARATE DURATIONS
        self.welcome_duration = 15  # Duration for Mr. David (recognized) - 1 minute
        self.alert_duration = 2    # Duration for Unknown person - 30 seconds
        
        # Welcome mode (for Mr. David - recognized face)
        self.in_welcome_mode = False  # Track if currently in welcome mode
        self.welcome_start_time = None  # When welcome mode started
        self.welcome_last_box = None  # Last detected face position for Mr. David
        
        # Alert mode (for Unknown - unrecognized face)
        self.in_alert_mode = False  # Track if currently in alert mode
        self.alert_start_time = None  # When alert mode started
        self.alert_last_box = None  # Last detected face position for unknown person
        
        # Load and encode known faces
        self.load_known_faces(known_faces_path)
        
    def load_known_faces(self, folder_path):
        """
        Load images from the known faces folder and create embeddings
        
        Args:
            folder_path: Path to folder containing Mr. David's images
        """
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
                # Read image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Get face embedding
                embedding = self.get_face_embedding(img)
                
                if embedding is not None:
                    self.known_face_embeddings.append(embedding)
                    print(f"Processed: {img_path.name}")
                    
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
        
        print(f"Successfully loaded {len(self.known_face_embeddings)} face embeddings for {self.known_face_name}")
        
        # Require minimum number of training images for reliable recognition
        if len(self.known_face_embeddings) < 5:
            print(f"\n⚠️  WARNING: Only {len(self.known_face_embeddings)} training images found!")
            print("   For best results, please add at least 5-10 clear photos of Mr. David")
            print("   More training images = better recognition accuracy")
        
        # Save embeddings for faster loading next time
        self.save_embeddings()
    
    def save_embeddings(self, filename='known_faces.pkl'):
        """Save known face embeddings to file"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'embeddings': self.known_face_embeddings,
                'name': self.known_face_name
            }, f)
        print(f"Embeddings saved to {filename}")
    
    def load_embeddings(self, filename='known_faces.pkl'):
        """Load known face embeddings from file"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.known_face_embeddings = data['embeddings']
                self.known_face_name = data['name']
            print(f"Loaded {len(self.known_face_embeddings)} embeddings from {filename}")
            return True
        return False
    
    def preprocess_for_detection(self, image):
        """
        Preprocess image for BlazeFace detection
        
        Args:
            image: Input image (BGR format from OpenCV)
        Returns:
            Preprocessed image ready for detection model
        """
        # Get input shape from model
        input_shape = self.detection_input_details[0]['shape']
        height, width = input_shape[1], input_shape[2]
        
        # Resize image
        img_resized = cv2.resize(image, (width, height))
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] or [-1, 1] depending on model
        img_normalized = img_rgb.astype(np.float32) / 127.5 - 1.0
        
        # Add batch dimension
        img_input = np.expand_dims(img_normalized, axis=0)
        
        return img_input
    
    def detect_faces(self, image):
        """
        Detect faces in image using BlazeFace
        
        Args:
            image: Input image (BGR format from OpenCV)
        Returns:
            List of face bounding boxes [(x1, y1, x2, y2), ...]
        """
        # Preprocess image
        input_data = self.preprocess_for_detection(image)
        
        # Run detection
        self.detector.set_tensor(self.detection_input_details[0]['index'], input_data)
        self.detector.invoke()
        
        # Get output
        # BlazeFace typically outputs: detection boxes and scores
        boxes = self.detector.get_tensor(self.detection_output_details[0]['index'])
        scores = self.detector.get_tensor(self.detection_output_details[1]['index'])
        
        # Filter detections by confidence threshold
        confidence_threshold = 0.5
        faces = []
        
        h, w = image.shape[:2]
        
        for i in range(len(scores[0])):
            if scores[0][i] > confidence_threshold:
                # Convert normalized coordinates to pixel coordinates
                box = boxes[0][i]
                
                # BlazeFace format: [ymin, xmin, ymax, xmax] (normalized)
                y1 = int(box[0] * h)
                x1 = int(box[1] * w)
                y2 = int(box[2] * h)
                x2 = int(box[3] * w)
                
                # Ensure coordinates are within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                faces.append((x1, y1, x2, y2))
        
        return faces
    
    def preprocess_for_recognition(self, face_image):
        """
        Preprocess face image for MobileFaceNet recognition
        
        Args:
            face_image: Cropped face image
        Returns:
            Preprocessed face ready for recognition model
        """
        # Get input shape from model
        input_shape = self.recognition_input_details[0]['shape']
        height, width = input_shape[1], input_shape[2]
        
        # Resize face
        face_resized = cv2.resize(face_image, (width, height))
        
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        face_normalized = face_rgb.astype(np.float32) / 255.0
        
        # Add batch dimension
        face_input = np.expand_dims(face_normalized, axis=0)
        
        return face_input
    
    def get_face_embedding(self, face_image):
        """
        Get face embedding using MobileFaceNet
        
        Args:
            face_image: Face image (can be full image, will be preprocessed)
        Returns:
            Face embedding vector
        """
        try:
            # Preprocess face
            input_data = self.preprocess_for_recognition(face_image)
            
            # Run recognition
            self.recognizer.set_tensor(self.recognition_input_details[0]['index'], input_data)
            self.recognizer.invoke()
            
            # Get embedding
            embedding = self.recognizer.get_tensor(self.recognition_output_details[0]['index'])
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding[0]
            
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    def recognize_face(self, face_image):
        """
        Recognize if face belongs to Mr. David - STRICT VERSION
        Uses multiple validation checks to avoid false positives
        
        Args:
            face_image: Cropped face image
        Returns:
            Tuple: (is_known, name, confidence)
        """
        if len(self.known_face_embeddings) == 0:
            return False, "Unknown", 0.0
        
        # Get embedding for current face
        current_embedding = self.get_face_embedding(face_image)
        
        if current_embedding is None:
            return False, "Unknown", 0.0
        
        # Compare with ALL known embeddings and collect similarities
        similarities = []
        
        for known_embedding in self.known_face_embeddings:
            # Calculate cosine similarity
            similarity = np.dot(current_embedding, known_embedding)
            similarities.append(similarity)
        
        # Convert to numpy array for analysis
        similarities = np.array(similarities)
        
        # Get statistics
        max_similarity = np.max(similarities)
        mean_similarity = np.mean(similarities)
        median_similarity = np.median(similarities)
        
        # Count how many embeddings exceed threshold
        matches_above_threshold = np.sum(similarities > self.recognition_threshold)
        high_confidence_matches = np.sum(similarities > self.strict_threshold)
        
        # STRICT RECOGNITION CRITERIA - ALL must be satisfied:
        # 1. Maximum similarity must exceed strict threshold
        # 2. Average similarity must be reasonably high
        # 3. Must match minimum number of training images
        # 4. Median similarity should be high (not just one outlier)
        
        is_recognized = (
            max_similarity > self.strict_threshold and  # Best match is very strong
            mean_similarity > (self.recognition_threshold - 0.05) and  # Average is good
            matches_above_threshold >= self.min_matching_embeddings and  # Matches multiple images
            median_similarity > (self.recognition_threshold - 0.10)  # Median is reasonable
        )
        
        # Extra validation: Check consistency
        # If only 1-2 images match strongly but others don't, it's suspicious
        if high_confidence_matches < 3 and matches_above_threshold < self.min_matching_embeddings:
            is_recognized = False
        
        # Debug output
        if self.debug_mode:
            print(f"\n--- Face Recognition Debug ---")
            print(f"Max similarity: {max_similarity:.3f}")
            print(f"Mean similarity: {mean_similarity:.3f}")
            print(f"Median similarity: {median_similarity:.3f}")
            print(f"Matches above {self.recognition_threshold}: {matches_above_threshold}/{len(similarities)}")
            print(f"High confidence matches (>{self.strict_threshold}): {high_confidence_matches}")
            print(f"Recognition: {'✅ RECOGNIZED' if is_recognized else '❌ UNKNOWN'}")
            print(f"-----------------------------")
        
        if is_recognized:
            # Use average of top 3 similarities as confidence
            top_3_avg = np.mean(np.sort(similarities)[-3:])
            return True, self.known_face_name, top_3_avg
        else:
            # Return the max similarity even for unknown (for debugging)
            return False, "Unknown", max_similarity
    
    def initialize_camera(self, camera_index=0):
        """
        Initialize USB camera
        
        Args:
            camera_index: Camera device index (usually 0)
        """
        self.camera = cv2.VideoCapture(camera_index)
        
        # Set camera properties for better performance
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.camera.isOpened():
            raise Exception("Could not open camera!")
        
        print("Camera initialized successfully")
    
    def draw_display_message(self, image, box, remaining_time, label_text, big_text, color):
        """
        Draw prominent display message (works for both welcome and alert modes)
        
        Args:
            image: Image to draw on
            box: Bounding box coordinates (x1, y1, x2, y2)
            remaining_time: Seconds remaining in display mode
            label_text: Text to show above box (e.g., "Welcome, Mr. David" or "Unknown")
            big_text: Large text at top (e.g., "WELCOME!" or "ALERT!")
            color: BGR color tuple (e.g., (255, 0, 0) for blue, (0, 0, 255) for red)
        """
        x1, y1, x2, y2 = box
        
        # Draw bounding box with thick border
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        
        # Main label
        label = label_text
        
        # Calculate text size
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        
        # Draw label background (larger)
        cv2.rectangle(image, (x1, y1 - 40), (x1 + label_size[0] + 20, y1), color, -1)
        
        # Draw label text (white, bold)
        cv2.putText(image, label, (x1 + 10, y1 - 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw timer below the box
        timer_text = f"Display active: {int(remaining_time)}s remaining"
        timer_size, _ = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Timer background
        cv2.rectangle(image, (x1, y2 + 5), (x1 + timer_size[0] + 10, y2 + 35), color, -1)
        
        # Timer text
        cv2.putText(image, timer_text, (x1 + 5, y2 + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw decorative corners for emphasis
        corner_length = 20
        corner_thickness = 4
        
        # Top-left corner
        cv2.line(image, (x1, y1), (x1 + corner_length, y1), color, corner_thickness)
        cv2.line(image, (x1, y1), (x1, y1 + corner_length), color, corner_thickness)
        
        # Top-right corner
        cv2.line(image, (x2, y1), (x2 - corner_length, y1), color, corner_thickness)
        cv2.line(image, (x2, y1), (x2, y1 + corner_length), color, corner_thickness)
        
        # Bottom-left corner
        cv2.line(image, (x1, y2), (x1 + corner_length, y2), color, corner_thickness)
        cv2.line(image, (x1, y2), (x1, y2 - corner_length), color, corner_thickness)
        
        # Bottom-right corner
        cv2.line(image, (x2, y2), (x2 - corner_length, y2), color, corner_thickness)
        cv2.line(image, (x2, y2), (x2, y2 - corner_length), color, corner_thickness)
        
        # Add a centered large message at the top of the frame
        h, w = image.shape[:2]
        big_text_size, _ = cv2.getTextSize(big_text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)
        big_text_x = (w - big_text_size[0]) // 2
        
        # Background for big text
        cv2.rectangle(image, (big_text_x - 20, 60), 
                     (big_text_x + big_text_size[0] + 20, 120), color, -1)
        
        # Big text
        cv2.putText(image, big_text, (big_text_x, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
    
    def draw_face_box(self, image, box, is_known, name, confidence):
        """
        Draw bounding box and label on image
        
        Args:
            image: Image to draw on
            box: Bounding box coordinates (x1, y1, x2, y2)
            is_known: Whether face is recognized
            name: Name to display
            confidence: Recognition confidence
        """
        x1, y1, x2, y2 = box
        
        # Choose color based on recognition
        if is_known:
            color = (255, 0, 0)  # Blue for Mr. David
            label = f"Welcome, {name}"
        else:
            color = (0, 0, 255)  # Red for unknown
            label = "Unknown"
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - 30), (x1 + label_size[0] + 10, y1), color, -1)
        
        # Draw label text
        cv2.putText(image, label, (x1 + 5, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw confidence score
        conf_text = f"{confidence:.2f}"
        cv2.putText(image, conf_text, (x1, y2 + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def run(self):
        """
        Main loop: Capture frames, detect faces, recognize, and display
        With separate display durations: 60s for Mr. David, 30s for Unknown
        """
        print("\n=== Face Recognition System Started ===")
        print(f"Recognition threshold: {self.recognition_threshold:.2f}")
        print(f"Strict threshold: {self.strict_threshold:.2f}")
        print(f"Min matching images: {self.min_matching_embeddings}")
        print(f"Training images loaded: {len(self.known_face_embeddings)}")
        print(f"Welcome duration (Mr. David): {self.welcome_duration} seconds")
        print(f"Alert duration (Unknown): {self.alert_duration} seconds")
        print("\nControls:")
        print("  Press 'q' to quit")
        print("  Press 's' to save current frame")
        print("  Press 'r' to reset current mode (welcome or alert)")
        if self.debug_mode:
            print("  Press 'd' to toggle debug output")
        print("======================================\n")
        
        frame_count = 0
        
        while True:
            # Capture frame
            ret, frame = self.camera.read()
            
            if not ret:
                print("Failed to capture frame")
                break
            
            # Create a copy for display
            display_frame = frame.copy()
            
            # Check if we're in welcome mode (Mr. David recognized)
            if self.in_welcome_mode:
                # Calculate time elapsed since welcome started
                elapsed_time = (datetime.now() - self.welcome_start_time).total_seconds()
                remaining_time = self.welcome_duration - elapsed_time
                
                if remaining_time > 0:
                    # Still in welcome mode - display welcome message
                    
                    # Use last detected box or detect current position
                    if self.welcome_last_box is not None:
                        face_box = self.welcome_last_box
                    else:
                        # Try to detect face position
                        faces = self.detect_faces(frame)
                        if len(faces) > 0:
                            face_box = faces[0]
                            self.welcome_last_box = face_box
                        else:
                            # No face detected, use center of frame
                            h, w = frame.shape[:2]
                            face_box = (w//4, h//4, 3*w//4, 3*h//4)
                    
                    # Draw welcome message (blue)
                    self.draw_display_message(display_frame, face_box, remaining_time, 
                                             "Welcome, Mr. David", "WELCOME!", (255, 0, 0))
                    
                    # Display info
                    info_text = f"WELCOME MODE - Time remaining: {int(remaining_time)}s"
                    cv2.putText(display_frame, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                else:
                    # Welcome period ended - exit welcome mode
                    print(f"✅ Welcome period ended. Resuming detection...\n")
                    self.in_welcome_mode = False
                    self.welcome_start_time = None
                    self.welcome_last_box = None
            
            # Check if we're in alert mode (Unknown person detected)
            elif self.in_alert_mode:
                # Calculate time elapsed since alert started
                elapsed_time = (datetime.now() - self.alert_start_time).total_seconds()
                remaining_time = self.alert_duration - elapsed_time
                
                if remaining_time > 0:
                    # Still in alert mode - display alert message
                    
                    # Use last detected box or detect current position
                    if self.alert_last_box is not None:
                        face_box = self.alert_last_box
                    else:
                        # Try to detect face position
                        faces = self.detect_faces(frame)
                        if len(faces) > 0:
                            face_box = faces[0]
                            self.alert_last_box = face_box
                        else:
                            # No face detected, use center of frame
                            h, w = frame.shape[:2]
                            face_box = (w//4, h//4, 3*w//4, 3*h//4)
                    
                    # Draw alert message (red)
                    self.draw_display_message(display_frame, face_box, remaining_time, 
                                             "Unknown", "ALERT!", (0, 0, 255))
                    
                    # Display info
                    info_text = f"ALERT MODE - Time remaining: {int(remaining_time)}s"
                    cv2.putText(display_frame, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    # Alert period ended - exit alert mode
                    print(f"⚠️  Alert period ended. Resuming detection...\n")
                    self.in_alert_mode = False
                    self.alert_start_time = None
                    self.alert_last_box = None
            
            else:
                # Normal detection mode
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                # Track if any face triggered a mode
                mode_triggered = False
                
                # Process each detected face
                for face_box in faces:
                    x1, y1, x2, y2 = face_box
                    
                    # Extract face region
                    face_img = frame[y1:y2, x1:x2]
                    
                    if face_img.size == 0:
                        continue
                    
                    # Recognize face
                    is_known, name, confidence = self.recognize_face(face_img)
                    
                    if is_known and name == self.known_face_name:
                        # Mr. David recognized - enter welcome mode
                        self.in_welcome_mode = True
                        self.welcome_start_time = datetime.now()
                        self.welcome_last_box = face_box
                        mode_triggered = True
                        
                        print(f"🎉 Mr. David recognized! (Confidence: {confidence:.3f})")
                        print(f"   Entering WELCOME MODE for {self.welcome_duration} seconds...\n")
                        
                        # Draw welcome message immediately
                        self.draw_display_message(display_frame, face_box, self.welcome_duration, 
                                                 "Welcome, Mr. David", "WELCOME!", (255, 0, 0))
                        
                        # Break to start welcome mode
                        break
                    else:
                        # Unknown person - enter alert mode
                        self.in_alert_mode = True
                        self.alert_start_time = datetime.now()
                        self.alert_last_box = face_box
                        mode_triggered = True
                        
                        print(f"⚠️  Unknown person detected! (Confidence: {confidence:.3f})")
                        print(f"   Entering ALERT MODE for {self.alert_duration} seconds...\n")
                        
                        # Draw alert message immediately
                        self.draw_display_message(display_frame, face_box, self.alert_duration, 
                                                 "Unknown", "ALERT!", (0, 0, 255))
                        
                        # Break to start alert mode
                        break
                
                # Display info on frame
                if not mode_triggered:
                    info_text = f"Faces detected: {len(faces)} | Scanning..."
                    cv2.putText(display_frame, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Face Recognition System', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                # Save current frame
                filename = f'captured_frame_{frame_count}.jpg'
                cv2.imwrite(filename, display_frame)
                print(f"📸 Frame saved as {filename}")
                frame_count += 1
            elif key == ord('r'):
                # Reset current mode (manual override)
                if self.in_welcome_mode:
                    print("🔄 Welcome mode manually reset\n")
                    self.in_welcome_mode = False
                    self.welcome_start_time = None
                    self.welcome_last_box = None
                elif self.in_alert_mode:
                    print("🔄 Alert mode manually reset\n")
                    self.in_alert_mode = False
                    self.alert_start_time = None
                    self.alert_last_box = None
            elif key == ord('d'):
                # Toggle debug mode
                self.debug_mode = not self.debug_mode
                status = "enabled" if self.debug_mode else "disabled"
                print(f"🔍 Debug mode {status}\n")
        
        # Cleanup
        self.camera.release()
        cv2.destroyAllWindows()
        print("\n✅ System stopped")


def main():
    """
    Main function to run the face recognition system
    """
    import sys
    
    # Check for debug mode argument
    debug_mode = '--debug' in sys.argv or '-d' in sys.argv
    
    # Configuration
    DETECTION_MODEL = 'blazeface.tflite'
    RECOGNITION_MODEL = 'mobilefacenet.tflite'
    KNOWN_FACES_FOLDER = 'mrdavid'
    
    # Check if model files exist
    if not os.path.exists(DETECTION_MODEL):
        print(f"Error: Detection model '{DETECTION_MODEL}' not found!")
        print("Please download blazeface.tflite and place it in the same directory")
        return
    
    if not os.path.exists(RECOGNITION_MODEL):
        print(f"Error: Recognition model '{RECOGNITION_MODEL}' not found!")
        print("Please download mobilefacenet.tflite and place it in the same directory")
        return
    
    if not os.path.exists(KNOWN_FACES_FOLDER):
        print(f"Error: Known faces folder '{KNOWN_FACES_FOLDER}' not found!")
        print("Please create the 'mrdavid' folder and add Mr. David's photos")
        return
    
    try:
        # Initialize system
        print("Initializing Face Recognition System...")
        if debug_mode:
            print("🔍 Debug mode enabled - will show detailed recognition info")
        
        system = FaceRecognitionSystem(
            detection_model_path=DETECTION_MODEL,
            recognition_model_path=RECOGNITION_MODEL,
            known_faces_path=KNOWN_FACES_FOLDER,
            debug_mode=debug_mode
        )
        
        # Initialize camera
        system.initialize_camera(camera_index=0)
        
        # Run the system
        system.run()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
