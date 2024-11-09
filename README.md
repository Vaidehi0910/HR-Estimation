# HR-Estimation
![HR-Estimation-github](https://github.com/user-attachments/assets/d4df1ce7-c119-4061-ae7e-9a56e3b0e6d4)
![HR-Estimation-github-mask](https://github.com/user-attachments/assets/45b81361-733f-458e-9995-88487692a15e)

# Abstract
- Heart Rate (HR) is one of the most important Physiological parameter and a vital indicator of people‘s physiological state
- A non-contact based system to measure Heart Rate: real-time application using camera
- Principal: extract heart rate information from facial skin color variation caused by blood circulation 
- Application: monitoring drivers‘ physiological state
- Contribution: Checks if the person is wearing mask or not. If yes, use forehead area, otherwise use forehead + lower face areas to extarct information

# Methods 
- Detect face and mask, align and get ROI using facial landmarks using mediapipe
- Apply band pass filter with fl = 0.8 Hz and fh = 3 Hz, which are 48 and 180 bpm respectively
- Average color value of ROI in each frame is calculate pushed to a data buffer which is 150 in length
- FFT the data buffer. The highest peak is Heart rate 

# Requirements
```
pip install -r requirements.txt
```

# Implementation
```
python GUI.py
```

# Reference
- Base of the project: https://github.com/habom2310/Heart-rate-measurement-using-camera/tree/master
- Real Time Heart Rate Monitoring From Facial RGB Color Video Using Webcam by H. Rahman, M.U. Ahmed, S. Begum, P. Funk
- Remote Monitoring of Heart Rate using Multispectral Imaging in Group 2, 18-551, Spring 2015 by Michael Kellman Carnegie (Mellon University), Sophia Zikanova (Carnegie Mellon University) and Bryan Phipps (Carnegie Mellon University)
- Non-contact, automated cardiac pulse measurements using video imaging and blind source separation by Ming-Zher Poh, Daniel J. McDuff, and Rosalind W. Picard
- Camera-based Heart Rate Monitoring by Janus Nørtoft Jensen and Morten Hannemose
- Graphs plotting is based on https://github.com/thearn/webcam-pulse-detector

# Note
- Application can only detect HR for 1 people at a time
- Sudden change can cause incorrect HR calculation. In the most case, HR can be correctly detected after 10 seconds being stable infront of the camera
- This github project is for study purpose only.
