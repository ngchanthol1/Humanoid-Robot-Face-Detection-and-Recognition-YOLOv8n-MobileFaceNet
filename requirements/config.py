"""
Face Recognition System Configuration
Edit these values to adjust recognition sensitivity
"""

# ============================================
# RECOGNITION THRESHOLDS
# ============================================

# Main recognition threshold (0.0 to 1.0)
# Higher = More Strict = Fewer False Positives
# Lower = More Lenient = More False Positives
# Recommended range: 0.70 - 0.85
RECOGNITION_THRESHOLD = 0.75

# Strict threshold for high confidence matches (0.0 to 1.0)
# Should be higher than RECOGNITION_THRESHOLD
# Recommended range: 0.75 - 0.90
STRICT_THRESHOLD = 0.80

# Minimum number of training images that must match
# Higher = More Strict
# Lower = More Lenient
# Recommended: 3-5 (but depends on total training images)
MIN_MATCHING_EMBEDDINGS = 3

# ============================================
# TRAINING DATA REQUIREMENTS
# ============================================

# Minimum number of training images required
# More images = Better accuracy
# Recommended: At least 5-10 images
MIN_TRAINING_IMAGES = 5

# ============================================
# CAMERA SETTINGS
# ============================================

# Camera index (usually 0 for built-in, 1+ for USB cameras)
CAMERA_INDEX = 0

# Camera resolution
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Camera FPS
CAMERA_FPS = 30

# ============================================
# DISPLAY SETTINGS
# ============================================

# Colors (BGR format)
KNOWN_FACE_COLOR = (255, 0, 0)  # Blue for Mr. David
UNKNOWN_FACE_COLOR = (0, 0, 255)  # Red for unknown

# Text labels
KNOWN_FACE_LABEL = "Welcome, Mr. David"
UNKNOWN_FACE_LABEL = "Unknown"

# Font settings
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# ============================================
# ADVANCED SETTINGS
# ============================================

# Face detection confidence threshold
DETECTION_CONFIDENCE = 0.5

# Enable/disable debug mode by default
DEBUG_MODE_DEFAULT = False

# ============================================
# MODEL PATHS
# ============================================

DETECTION_MODEL = 'blazeface.tflite'
RECOGNITION_MODEL = 'mobilefacenet.tflite'
KNOWN_FACES_FOLDER = 'mrdavid'

# ============================================
# NOTES FOR ADJUSTMENT
# ============================================

"""
IF YOU'RE GETTING TOO MANY FALSE POSITIVES (strangers recognized as Mr. David):
1. INCREASE RECOGNITION_THRESHOLD (try 0.80 or 0.85)
2. INCREASE STRICT_THRESHOLD (try 0.85 or 0.90)
3. INCREASE MIN_MATCHING_EMBEDDINGS (try 4 or 5)

IF MR. DAVID IS NOT BEING RECOGNIZED:
1. DECREASE RECOGNITION_THRESHOLD (try 0.70 or 0.65)
2. DECREASE STRICT_THRESHOLD (try 0.75 or 0.70)
3. DECREASE MIN_MATCHING_EMBEDDINGS (try 2)
4. Add more training photos of Mr. David
5. Ensure training photos are clear and well-lit
6. Use photos from different angles

RECOMMENDED SETTINGS FOR MAXIMUM SECURITY (avoid false positives):
- RECOGNITION_THRESHOLD = 0.80
- STRICT_THRESHOLD = 0.85
- MIN_MATCHING_EMBEDDINGS = 4
- At least 10 training images

RECOMMENDED SETTINGS FOR CONVENIENCE (easier to recognize):
- RECOGNITION_THRESHOLD = 0.70
- STRICT_THRESHOLD = 0.75
- MIN_MATCHING_EMBEDDINGS = 2
- At least 5 training images
"""
