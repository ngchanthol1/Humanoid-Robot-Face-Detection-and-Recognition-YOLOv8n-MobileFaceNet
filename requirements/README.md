# Face Recognition System for Raspberry Pi 5

A real-time face recognition system using BlazeFace for detection and MobileFaceNet for recognition, specifically designed for Raspberry Pi 5 with USB camera.

## Features

- ✅ Real-time face detection using BlazeFace
- ✅ Face recognition using MobileFaceNet
- ✅ Color-coded bounding boxes (Blue for known, Red for unknown)
- ✅ Welcome message for recognized faces
- ✅ Multi-face detection and recognition
- ✅ Optimized for Raspberry Pi 5

## Hardware Requirements

- Raspberry Pi 5
- USB RGB Camera
- Monitor (for displaying the GUI)
- Keyboard (for system control)

## Software Requirements

- Raspberry Pi OS (64-bit recommended)
- Python 3.8 or higher
- Libraries listed in requirements.txt

## Project Structure

```
face-recognition-system/
│
├── face_recognition_system.py    # Main program
├── requirements.txt               # Python dependencies
├── blazeface.tflite              # Face detection model (download separately)
├── mobilefacenet.tflite          # Face recognition model (download separately)
├── mrdavid/                      # Folder with Mr. David's photos
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── known_faces.pkl               # Generated embeddings (created automatically)
```

## Installation Steps

### Step 1: Update Your Raspberry Pi

```bash
sudo apt update
sudo apt upgrade -y
```

### Step 2: Install System Dependencies

```bash
sudo apt install -y python3-pip python3-opencv
sudo apt install -y libatlas-base-dev libhdf5-dev libhdf5-serial-dev
sudo apt install -y libharfbuzz0b libwebp7 libtiff5 libjasper-dev
sudo apt install -y libqtgui4 libqt4-test libqtcore4
```

### Step 3: Clone or Create Project Directory

```bash
mkdir ~/face-recognition-system
cd ~/face-recognition-system
```

### Step 4: Copy the Python Files

Copy the following files to your project directory:
- `face_recognition_system.py`
- `requirements.txt`

### Step 5: Install Python Dependencies

```bash
pip3 install -r requirements.txt
```

**For Raspberry Pi specifically, use:**
```bash
pip3 install tensorflow-lite-runtime
pip3 install opencv-python
pip3 install numpy
```

### Step 6: Download Model Files

You need to download the TFLite models:

**Option 1: Download from TensorFlow Hub**
```bash
# BlazeFace model
wget https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite -O blazeface.tflite

# For MobileFaceNet, you may need to convert from other sources or use alternatives
```

**Option 2: Manually place the models**
- Download `blazeface.tflite` and place it in the project directory
- Download `mobilefacenet.tflite` and place it in the project directory

### Step 7: Prepare Training Images

Create a folder named `mrdavid` and add Mr. David's photos:

```bash
mkdir mrdavid
```

Then copy multiple photos of Mr. David into this folder:
- Use at least 5-10 different photos
- Include different angles, lighting conditions, and expressions
- Use clear, front-facing photos for best results
- Supported formats: .jpg, .jpeg, .png, .bmp

**Example:**
```
mrdavid/
├── photo1.jpg
├── photo2.jpg
├── photo3.jpg
├── photo4.jpg
└── photo5.jpg
```

## Running the System

### Basic Usage

```bash
python3 face_recognition_system.py
```

### Keyboard Controls

- **'q'** - Quit the application
- **'s'** - Save current frame as image

### First Run

On first run, the system will:
1. Load the detection and recognition models
2. Process all images in the `mrdavid` folder
3. Create face embeddings
4. Save embeddings to `known_faces.pkl` for faster loading next time
5. Initialize the camera
6. Start real-time face recognition

### Expected Output

```
Initializing Face Recognition System...
Loading known faces from: mrdavid
Found 10 images
Processed: photo1.jpg
Processed: photo2.jpg
...
Successfully loaded 10 face embeddings for Mr. David
Embeddings saved to known_faces.pkl
Camera initialized successfully

=== Face Recognition System Started ===
Press 'q' to quit
Press 's' to save current frame
======================================
```

## Visual Indicators

- **Blue Bounding Box** + "Welcome, Mr. David" = Recognized face
- **Red Bounding Box** + "Unknown" = Unrecognized face
- **Confidence Score** displayed below each face

## Troubleshooting

### Camera Not Working

```bash
# Check available cameras
ls /dev/video*

# Test camera with v4l2
v4l2-ctl --list-devices

# If using different camera index, modify code:
system.initialize_camera(camera_index=0)  # Try 1, 2, etc.
```

### TensorFlow Lite Import Error

If you get import errors with TensorFlow:

```bash
# Uninstall regular TensorFlow
pip3 uninstall tensorflow

# Install TFLite runtime only (lighter)
pip3 install --index-url https://google-coral.github.io/py-repo/ tflite_runtime
```

Then modify the code:
```python
# Replace: import tensorflow as tf
# With: import tflite_runtime.interpreter as tflite
```

### Low FPS / Performance Issues

1. Reduce camera resolution:
```python
self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
```

2. Process every N frames instead of every frame

3. Use hardware acceleration if available

### Model File Not Found

Make sure model files are in the same directory:
```bash
ls -la *.tflite
```

### No Faces in Training Folder

Check that images are properly formatted:
```bash
ls -la mrdavid/
file mrdavid/*.jpg
```

## Customization

### Change Recognition Threshold

In `face_recognition_system.py`, modify:
```python
self.recognition_threshold = 0.6  # Lower = more strict, Higher = more lenient
```

### Add More People

1. Create additional folders (e.g., `mrjohn`, `msjane`)
2. Modify the code to load multiple people
3. Update the recognition logic to handle multiple known faces

### Change Display Settings

Modify colors and text in the `draw_face_box()` method:
```python
color = (255, 0, 0)  # BGR format
label = "Your Custom Text"
```

## Performance Tips for Raspberry Pi 5

1. **Use 64-bit OS** for better performance
2. **Enable GPU acceleration** if available
3. **Close unnecessary applications** while running
4. **Use good lighting** for better detection accuracy
5. **Position camera at eye level** for best results

## License

This project is for educational purposes. Make sure to comply with model licenses and privacy regulations.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure model files are correctly downloaded
4. Check camera permissions

## Credits

- BlazeFace: Google MediaPipe
- MobileFaceNet: Face recognition model
- OpenCV: Computer vision library
