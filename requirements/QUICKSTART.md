# QUICK START GUIDE
## Face Recognition System - Raspberry Pi 5

### 📋 BEFORE YOU START

You need:
1. ✅ Raspberry Pi 5 with Raspberry Pi OS
2. ✅ USB RGB Camera connected
3. ✅ Monitor and keyboard
4. ✅ Internet connection (for installation)

---

### 🚀 INSTALLATION (One-Time Setup)

**Step 1: Download and place files**
```bash
cd ~
mkdir face-recognition-system
cd face-recognition-system
```

Place these files in the folder:
- `face_recognition_system.py`
- `requirements.txt`
- `setup.sh`
- `test_camera.py`

**Step 2: Run automatic setup**
```bash
chmod +x setup.sh
./setup.sh
```

**Step 3: Download model files**

You need to manually download:
- `blazeface.tflite` → Face detection model
- `mobilefacenet.tflite` → Face recognition model

Place them in the `face-recognition-system` folder

**Step 4: Add Mr. David's photos**
```bash
# Create folder
mkdir mrdavid

# Copy 5-10 clear photos of Mr. David into the 'mrdavid' folder
# Use .jpg, .jpeg, or .png files
```

**Step 5: Test camera**
```bash
python3 test_camera.py
```

If camera works, you're ready!

---

### ▶️ RUNNING THE SYSTEM

```bash
cd ~/face-recognition-system
python3 face_recognition_system.py
```

**Controls:**
- Press `Q` to quit
- Press `S` to save screenshot

---

### 🎨 WHAT YOU'LL SEE

**Blue Box + "Welcome, Mr. David"**
- Face is recognized as Mr. David
- Confidence score shown below

**Red Box + "Unknown"**
- Face is not recognized
- Stranger/unknown person

---

### ⚠️ TROUBLESHOOTING

**Camera not working?**
```bash
# Check camera
ls /dev/video*

# Test with:
python3 test_camera.py
```

**Models not found?**
```bash
# Check files exist
ls -la *.tflite

# Should show:
# blazeface.tflite
# mobilefacenet.tflite
```

**No training images?**
```bash
# Check Mr. David's photos
ls -la mrdavid/

# Should show at least 5 .jpg files
```

**Python packages missing?**
```bash
pip3 install -r requirements.txt
```

**Low performance?**
- Close other programs
- Reduce camera resolution
- Use better lighting
- Make sure you're using Raspberry Pi 5 (not older models)

---

### 📊 ADJUSTING RECOGNITION SENSITIVITY

Edit `face_recognition_system.py`:

```python
# Line ~44
self.recognition_threshold = 0.6

# Lower value (0.4) = More strict, fewer false positives
# Higher value (0.8) = More lenient, more false positives
```

---

### 📁 PROJECT STRUCTURE

```
face-recognition-system/
├── face_recognition_system.py  ← Main program
├── requirements.txt            ← Dependencies
├── setup.sh                    ← Setup script
├── test_camera.py             ← Camera test
├── blazeface.tflite           ← Detection model
├── mobilefacenet.tflite       ← Recognition model
├── mrdavid/                   ← Training photos
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── ...
└── known_faces.pkl            ← Auto-generated embeddings
```

---

### 🔗 USEFUL COMMANDS

**Check Python version:**
```bash
python3 --version
# Should be 3.8 or higher
```

**Check installed packages:**
```bash
pip3 list | grep -E "numpy|opencv|tensorflow"
```

**Re-train face embeddings:**
```bash
# Delete old embeddings
rm known_faces.pkl

# Run program (will recreate embeddings)
python3 face_recognition_system.py
```

**Save video recording:**
```bash
# Modify code or press 'S' key to save frames
```

---

### 📞 NEED HELP?

1. Read full README.md for detailed instructions
2. Check troubleshooting section above
3. Verify all files are in correct locations
4. Make sure camera has good lighting
5. Ensure Mr. David's photos are clear and front-facing

---

### ✅ CHECKLIST BEFORE RUNNING

- [ ] Raspberry Pi 5 is powered on
- [ ] USB camera is connected
- [ ] Model files (.tflite) are in folder
- [ ] Mr. David's photos are in 'mrdavid' folder
- [ ] All dependencies installed (setup.sh completed)
- [ ] Camera test passed (test_camera.py)

**If all checked, you're ready to run!**

```bash
python3 face_recognition_system.py
```

---

**Happy Face Recognition! 😊**
