"""
Model Download Helper
Script to help download and verify TFLite models
"""

import os
import urllib.request
import hashlib

def download_file(url, filename):
    """Download file from URL with progress indicator"""
    
    print(f"Downloading {filename}...")
    
    try:
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f"\rProgress: {percent}% ", end='')
        
        urllib.request.urlretrieve(url, filename, progress_hook)
        print(f"\n✅ Successfully downloaded {filename}")
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading {filename}: {e}")
        return False


def verify_file(filename, expected_size=None):
    """Verify that file exists and is valid"""
    
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        return False
    
    file_size = os.path.getsize(filename)
    print(f"✅ {filename} exists ({file_size:,} bytes)")
    
    if expected_size and abs(file_size - expected_size) > 1000:
        print(f"⚠️  Warning: File size differs from expected ({expected_size:,} bytes)")
    
    return True


def main():
    """Main function to download models"""
    
    print("="*50)
    print("TFLite Model Download Helper")
    print("="*50)
    print()
    
    # BlazeFace model from MediaPipe
    print("1. Downloading BlazeFace model...")
    blazeface_url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    
    if not os.path.exists("blazeface.tflite"):
        if download_file(blazeface_url, "blazeface.tflite"):
            verify_file("blazeface.tflite")
    else:
        print("✅ blazeface.tflite already exists")
        verify_file("blazeface.tflite")
    
    print()
    
    # MobileFaceNet - this is trickier as it's not as readily available
    print("2. MobileFaceNet model...")
    print("   ⚠️  MobileFaceNet requires manual download")
    print()
    print("   Options to get mobilefacenet.tflite:")
    print("   a) Download from model zoo repositories")
    print("   b) Convert from ONNX/PyTorch using converter")
    print("   c) Use alternative: Try this URL (if available):")
    print("      https://github.com/sirius-ai/MobileFaceNet_TF/tree/master/arch/model")
    print()
    
    if not os.path.exists("mobilefacenet.tflite"):
        print("   ❌ mobilefacenet.tflite not found")
        print()
        print("   Manual download steps:")
        print("   1. Search for 'mobilefacenet tflite model download'")
        print("   2. Download the .tflite file")
        print("   3. Rename it to 'mobilefacenet.tflite'")
        print("   4. Place it in this directory")
    else:
        print("   ✅ mobilefacenet.tflite already exists")
        verify_file("mobilefacenet.tflite")
    
    print()
    print("="*50)
    print("Alternative: Using OpenCV DNN Face Recognition")
    print("="*50)
    print()
    print("If you have trouble getting MobileFaceNet, you can use")
    print("OpenCV's built-in face recognition models:")
    print()
    print("Download these files:")
    print("1. Face detection:")
    print("   https://github.com/opencv/opencv/tree/master/data/haarcascades")
    print("   → haarcascade_frontalface_default.xml")
    print()
    print("2. Face recognition (OpenFace):")
    print("   https://github.com/opencv/opencv_3rdparty/tree/dnn_samples_face_detector_20180205_fp16")
    print("   → opencv_face_detector.caffemodel")
    print("   → opencv_face_detector.prototxt")
    print()
    
    # Download OpenCV Haar Cascade as alternative
    print("Downloading OpenCV Haar Cascade (backup option)...")
    haar_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    
    if not os.path.exists("haarcascade_frontalface_default.xml"):
        download_file(haar_url, "haarcascade_frontalface_default.xml")
    else:
        print("✅ haarcascade_frontalface_default.xml already exists")
    
    print()
    print("="*50)
    print("Summary")
    print("="*50)
    
    files_status = {
        "blazeface.tflite": os.path.exists("blazeface.tflite"),
        "mobilefacenet.tflite": os.path.exists("mobilefacenet.tflite"),
        "haarcascade_frontalface_default.xml": os.path.exists("haarcascade_frontalface_default.xml")
    }
    
    for filename, exists in files_status.items():
        status = "✅" if exists else "❌"
        print(f"{status} {filename}")
    
    print()
    
    if files_status["blazeface.tflite"] and files_status["mobilefacenet.tflite"]:
        print("🎉 All required models are ready!")
        print("You can now run: python3 face_recognition_system.py")
    else:
        print("⚠️  Some models are missing. See instructions above.")


if __name__ == "__main__":
    main()
