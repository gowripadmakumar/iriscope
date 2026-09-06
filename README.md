<img width="1280" height="640" alt="git (1)" src="https://github.com/user-attachments/assets/8920b256-2ba8-4988-b824-5351134eb4bd" />



# Iriscope 🎯


## Basic Details
### Team Name: Irizz


### Team Member
- Team Lead: Gowri Padmakumar - St. Joseph's College(Autonomous), Devagiri

### Project Description
IRISCOPE is a computer-vision application that detects the iris from an eye image and analyzes its texture to estimate the number of visible radial structures. Built using Python, OpenCV, and MediaPipe, it turns an intentionally useless question into a fun, technically real computer-vision project.


### The Problem (that doesn't exist)
Nobody knows exactly how many visible radial lines their iris has; and, more importantly, nobody really needs to know.

### The Solution (that nobody asked for)
IRISCOPE solves this completely unnecessary problem by detecting and counting the approximate visible radial structures in your iris. 👁️
Because apparently, this needed to be measured.

### Live Demo
https://iriscopever2.onrender.com

## Technical Details
### Technologies/Components Used

For Software:
Languages used: Python, HTML, CSS, JavaScript
Frameworks used: Flask
Libraries used: OpenCV, NumPy
Tools used: VS Code, Git, GitHub, Web Browser

For Hardware:
Main components: Laptop/PC, built-in or USB webcam
Specifications: Webcam capable of capturing eye images; minimum 720p recommended
Tools required: USB connection (if using an external webcam)
Implementation

For Software:

# Installation
git clone <repository-url>
cd IRISCOPE
pip install -r requirements.txt

# Run
python run.py
Then open:
http://127.0.0.1:5000/


### Project Documentation
For Software:

# Screenshots (Add at least 3)
https://drive.google.com/drive/folders/1J4dP0zwbsfH3OV8OD8VKeIHc55tKwEc5?usp=drive_link

# Diagrams
https://drive.google.com/file/d/1oUH6MRfMSVcKkK8c5dkh6lSNznpEp8n4/view?usp=sharing
```
flowchart TD
    A[Start] --> B[Capture image<br/>Upload or camera capture]
    B --> C[Face + eye detection<br/>Haar cascades, upper face region]
    C -->|no face/eye found| Z1[Face/eye missing]
    C --> D[Pupil + iris boundary<br/>Darkest blob, brightness jump]
    D -->|pupil not found| Z2[Pupil not found]
    D --> E[Masking + usability check<br/>Reflection and eyelid exclusion]
    E -->|too little usable area| Z3[Too occluded]
    E --> F[Polar normalization<br/>64x360 unwrap, CLAHE]
    F -->|too little valid texture| Z4[Low texture]
    F --> G[Structure detection<br/>Edge coverage, merge, split]
    G --> H[Result<br/>Count, quality score, stage images]
```

### Project Demo
# Video
https://drive.google.com/file/d/1mgyUYnn9k655S0fzdWnPRp-Nyntp5GOT/view?usp=sharing
A little look at IRISCOPE 👁️✨
Here’s a quick preview of the website and its overall interface.



## Team Contributions
Gowri Padmakumar: Idea conceptualization, UI/UX design, frontend development, Flask backend development, computer-vision implementation, iris detection and radial structure analysis, testing, debugging, documentation, GitHub management, and deployment.


---
Made with ❤️ at TinkerHub Useless Projects 

![Static Badge](https://img.shields.io/badge/TinkerHub-24?color=%23000000&link=https%3A%2F%2Fwww.tinkerhub.org%2F)
![Static Badge](https://img.shields.io/badge/UselessProjects--26-26?link=https%3A%2F%2Ftinkerhub.org%2Fevents%2F1M8ORET9A1%2Fuseless-projects-3.0)



README.md
Displaying README.md.
