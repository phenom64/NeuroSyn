# NeuroSyn

<a href="https://syndromatic.com"><img src="model-v4/NSEmedia/syn app light.png" alt="Syndromatic Logo" width="100"/></a>

**NeuroSyn** is a home-based physiotherapy assistant that uses a surface EMG armband to:

- Calibrate muscle–signal baselines with a guided “Calibration Assistant”  
- Monitor real-time muscle activation strength via a progress bar  
- Classify physiotherapy exercises (e.g., elbow flexion, shoulder abduction, leg raises) with an AI model  
- Deliver on-screen cues, countdowns, and corrective feedback in an intuitive GUI  
- Log timestamped predictions and status messages in a CRT-style console  

## 🔍 Key Features

1. **Calibration Assistant**  
   - Click “OK” to start, per-exercise countdown, auto-advance on completion  
2. **Real-Time Signal Processing**  
   - Z-score normalization & RMS feature extraction  
   - Configurable confidence threshold for model predictions  
3. **Exercise Classification**  
   - TensorFlow/Keras model supporting multi-class recognition of rehab movements  
   - Easily extendable via `ICON_PATHS` & custom icons in `NSEmedia/gestures/`  
4. **Patient-Centric UI**  
   - High-contrast, large text (ITC Garamond Condensed)  
   - Gamified visual cues & progress tracking  
   - Dot-Matrix font logging overlay  
5. **Edge-AI Ready**  
   - Prepared for on-device inference (TinyML / TensorFlow Lite)  
   - Optimized for low-latency, low-power microcontrollers  

## 💾 Installation

```bash
git clone https://github.com/phenom64/NeuroSyn.git
cd NeuroSyn/model-v4
pip install -r requirements.txt
```
1. Set MYO_ADDRESS in constants.py to your armband's identifier (MAC Address)
2. [For developers/Forks] Place exercise icons under NSEmedia/gestures/ and update ICON_PATHS in constants.py

## 🚀 Launch

```bash
python NSE-interfaceFX.py
```
1. Calibration: Follow on-screen prompts on Calibration Assistant
2. Exercise Session: Perform movements; observe live feedback
   
## 🌟 Project Lineage

This repository is a fork of the Artemis Project originally developed by the University of Manchester’s Robotics Society. As a key member of this society, I architected and implemented its cross-platform PyQt6 user interface for the Artemis Project. NeuroSyn extends my work with physiotherapy-focused calibration, classification, and feedback modules for my final-year dissertation in Product Design Engineering.<br> <br>

<small><sub>
TM & &copy; 2025. Syndromatic Inc. All rights reserved.<br>
Designed by Kavish Krishnakumar, Dylan Simpson, Szymon Arciszewski, and Mark Matvijcsuk in Manchester.
</sub></small>

