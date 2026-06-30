import cv2
import os
import time

# ===============================
# CONFIGURATION
# ===============================
PERSON_NAME = "david"
SAVE_DIR = f"dataset/{PERSON_NAME}"
CAMERA_INDEX = 0
FRAME_WIDTH = 1920        # Max supported by camera
FRAME_HEIGHT = 1920
CAPTURE_INTERVAL = 1
MAX_IMAGES = 120

# FACE CROP SETTINGS FOR MOBILEFACENET
FACE_SIZE = 1080           # MobileFaceNet standard input size (112x112)
MARGIN = 0.3              # 30% margin around detected face for better recognition

# IMAGE QUALITY SETTINGS
PNG_COMPRESSION = 0       # 0-9 (0 = no compression, largest file)

# ===============================
# CREATE SAVE DIRECTORY
# ===============================
os.makedirs(SAVE_DIR, exist_ok=True)

# ===============================
# LOAD FACE DETECTOR (BUILT-IN PATH)
# ===============================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise IOError("❌ Haar Cascade not loaded!")

# ===============================
# FACE CROP FUNCTION
# ===============================
def crop_face(frame, x, y, w, h, margin=0.3, target_size=1080):
    """
    Crop face with margin and resize for MobileFaceNet training.
    
    Args:
        frame: Input image
        x, y, w, h: Face bounding box coordinates
        margin: Extra margin around face (0.3 = 30%)
        target_size: Output size (112 for MobileFaceNet)
    
    Returns:
        Cropped and resized face image
    """
    img_h, img_w = frame.shape[:2]
    
    # Calculate margin in pixels
    margin_x = int(w * margin)
    margin_y = int(h * margin)
    
    # Expand bounding box with margin
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(img_w, x + w + margin_x)
    y2 = min(img_h, y + h + margin_y)
    
    # Crop face region
    face_crop = frame[y1:y2, x1:x2]
    
    # Make it square (important for MobileFaceNet)
    crop_h, crop_w = face_crop.shape[:2]
    if crop_h != crop_w:
        size = max(crop_h, crop_w)
        square_face = cv2.copyMakeBorder(
            face_crop,
            top=(size - crop_h) // 2,
            bottom=(size - crop_h + 1) // 2,
            left=(size - crop_w) // 2,
            right=(size - crop_w + 1) // 2,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
        face_crop = square_face
    
    # Resize to target size (112x112 for MobileFaceNet)
    face_resized = cv2.resize(face_crop, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    
    return face_resized

# ===============================
# OPEN CAMERA
# ===============================
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)

# Confirm actual resolution
actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"📷 Capturing at: {actual_w} x {actual_h}")

count = 1  # Start from 1 for david_001
last_capture = time.time()

print(f"▶ Face detection active. Saving {FACE_SIZE}x{FACE_SIZE} face crops.")
print("▶ Press 'q' to quit.")

# ===============================
# CAPTURE LOOP
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera read failed")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(120, 120)
    )

    # Create display frame with face boxes
    display_frame = frame.copy()
    
    # Draw rectangles on display frame
    for (x, y, w, h) in faces:
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Save CROPPED FACE when detected
    if len(faces) > 0 and (time.time() - last_capture) >= CAPTURE_INTERVAL:
        
        # Get the largest face (most likely the main subject)
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        # Crop and resize face for MobileFaceNet
        face_crop = crop_face(frame, x, y, w, h, margin=MARGIN, target_size=FACE_SIZE)
        
        # Save as PNG: david_001.png, david_002.png, etc.
        filename = f"{PERSON_NAME}_{count:03d}.png"
        filepath = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(filepath, face_crop, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
        
        # Get file size
        file_size = os.path.getsize(filepath)
        size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.2f} MB"
        
        print(f"✅ Saved: {filepath} ({FACE_SIZE}x{FACE_SIZE}, {size_str})")
        count += 1
        last_capture = time.time()
        
        # Show the cropped face in a separate window
        cv2.imshow("Cropped Face", cv2.resize(face_crop, (1080, 1080)))

    # Show live feed with face boxes
    cv2.imshow("Face Collection - Press 'q' to quit", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count > MAX_IMAGES:
        break

# ===============================
# CLEANUP
# ===============================
cap.release()
cv2.destroyAllWindows()
print(f"✅ Face collection finished. {count - 1} images saved to {SAVE_DIR}/")

"""
## Key Changes for MobileFaceNet Training:

| Feature | Description |
|---------|-------------|
| **Face Size** | 112x112 pixels (MobileFaceNet standard) |
| **Margin** | 30% extra padding around face |
| **Square Crop** | Ensures aspect ratio is 1:1 |
| **Largest Face** | Selects the biggest detected face |
| **High Quality Resize** | Uses LANCZOS4 interpolation |
| **Face Preview** | Shows cropped face in separate window |

## Output:
```
dataset/david/david_001.png  (112x112)
dataset/david/david_002.png  (112x112)
dataset/david/david_003.png  (112x112)
...
"""