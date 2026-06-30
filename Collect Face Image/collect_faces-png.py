import cv2
import os
import time

# ===============================
# CONFIGURATION
# ===============================
PERSON_NAME = "david"
SAVE_DIR = f"dataset/{PERSON_NAME}"
CAMERA_INDEX = 0
FRAME_WIDTH = 360        # Max supported by camera
FRAME_HEIGHT = 360
CAPTURE_INTERVAL = 1
MAX_IMAGES = 120

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
# OPEN CAMERA
# ===============================
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 5)

# Confirm actual resolution
actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"📷 Capturing at: {actual_w} x {actual_h}")

count = 1  # Start from 1 for david_001
last_capture = time.time()

print("▶ Face detection active (no boxes). Press 'q' to quit.")

# ===============================
# CAPTURE LOOP
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera read failed")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection (trigger only)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(120, 120)
    )

    # Save FULL FRAME when face detected
    if len(faces) > 0 and (time.time() - last_capture) >= CAPTURE_INTERVAL:
        
        # Save as PNG with naming: david_001, david_002, etc.
        filename = f"{PERSON_NAME}_{count:03d}.png"
        filepath = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(filepath, frame, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
        
        # Get file size
        file_size = os.path.getsize(filepath)
        size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.2f} MB"
        
        print(f"✅ Saved: {filepath} ({size_str})")
        count += 1
        last_capture = time.time()

    # Show live feed (no overlays)
    cv2.imshow("Face Collection - Full Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count > MAX_IMAGES:
        break

# ===============================
# CLEANUP
# ===============================
cap.release()
cv2.destroyAllWindows()
print(f"✅ Face collection finished. {count - 1} images saved.")

"""

## Key Changes:

| Change | Before | After |
|--------|--------|-------|
| Name prefix | `mrdavid` | `david` |
| File format | `.jpg` | `.png` |
| Numbering format | `0000` (4 digits) | `001` (3 digits) |
| Starting count | `0` | `1` |
| Quality | Default JPEG | PNG with 0 compression (max quality) |

## Output Files:
```
dataset/david/david_001.png
dataset/david/david_002.png
dataset/david/david_003.png
...
dataset/david/david_010.png

"""