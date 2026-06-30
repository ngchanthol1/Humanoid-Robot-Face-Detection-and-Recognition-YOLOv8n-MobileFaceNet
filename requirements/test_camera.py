"""
Camera Test Script
Quick test to verify USB camera is working on Raspberry Pi 5
"""

import cv2
import sys

def test_camera():
    """Test if camera is accessible and working"""
    
    print("========================================")
    print("Camera Test for Raspberry Pi 5")
    print("========================================\n")
    
    # Try different camera indices
    for camera_index in range(3):
        print(f"Testing camera index {camera_index}...")
        
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"  ❌ Camera {camera_index} not accessible\n")
            continue
        
        # Try to read a frame
        ret, frame = cap.read()
        
        if ret:
            height, width = frame.shape[:2]
            print(f"  ✅ Camera {camera_index} is working!")
            print(f"     Resolution: {width}x{height}")
            print(f"     FPS: {cap.get(cv2.CAP_PROP_FPS)}")
            
            # Display frame
            print("\nDisplaying camera feed...")
            print("Press 'q' to quit, 's' to save test image\n")
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Add text overlay
                cv2.putText(frame, f"Camera {camera_index} Test", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                cv2.imshow('Camera Test', frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f'camera_test_{camera_index}.jpg'
                    cv2.imwrite(filename, frame)
                    print(f"  Image saved as {filename}")
            
            cap.release()
            cv2.destroyAllWindows()
            
            print("\n✅ Camera test successful!")
            print(f"Use camera_index={camera_index} in the main program\n")
            return True
        else:
            print(f"  ❌ Camera {camera_index} opened but cannot read frames\n")
            cap.release()
    
    print("❌ No working camera found!")
    print("\nTroubleshooting:")
    print("1. Check if camera is connected: ls /dev/video*")
    print("2. Check camera permissions: sudo usermod -a -G video $USER")
    print("3. Reboot the Raspberry Pi")
    print("4. Try a different USB port")
    
    return False


def check_dependencies():
    """Check if required packages are installed"""
    
    print("Checking dependencies...\n")
    
    try:
        import numpy
        print(f"✅ NumPy version: {numpy.__version__}")
    except ImportError:
        print("❌ NumPy not installed!")
        print("   Install: pip3 install numpy")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not installed!")
        print("   Install: pip3 install opencv-python")
        return False
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
    except ImportError:
        try:
            import tflite_runtime
            print(f"✅ TFLite Runtime installed")
        except ImportError:
            print("❌ TensorFlow/TFLite not installed!")
            print("   Install: pip3 install tensorflow-lite-runtime")
            return False
    
    print()
    return True


if __name__ == "__main__":
    print()
    
    # Check dependencies first
    if not check_dependencies():
        print("\n⚠️  Please install missing dependencies first!")
        sys.exit(1)
    
    # Test camera
    if test_camera():
        print("========================================")
        print("You can now run the face recognition system!")
        print("Command: python3 face_recognition_system.py")
        print("========================================")
    else:
        print("========================================")
        print("Please fix camera issues before proceeding")
        print("========================================")
