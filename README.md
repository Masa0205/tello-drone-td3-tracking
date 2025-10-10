# tello-drone-td3-tracking
This project demonstrates autonomous visual tracking using a **Tello** drone controlled by reiforcement learnig algorithm (**TD3**).

## Overview
A custom **gymnasium** environment was designed to simulate the control process. The Tello vision was simply reproduced in gymnasium environment (960 × 720). 

The TD3 agent was trained to move the detected point from a random initial position to the center of the image.

After training, the learned policy was transferred to the Tello drone for real-world control.

## Architecture
### 1. Gymnasium Environment
- Anyone can create a custom environment using the Gymnasium library.
- In this study, a 2D environment was designed to simply reproduce the Tello’s visual input.
- The environment consists of two main elements: a detected point (agent) and a target point (center area), representing the position to be aligned.

### 2. TD3 Agent
- TD3 (Twin Delayed Deep Deterministic Policy Gradient) addresses some of the limitations of DDPG.
  
  - Twin: TD3 architecture employs two critic networks, and the smaller Q-value is used during training to mitigate overestimation.

  - Dlayed:The actor network is updated less frequently than the critics, which helps stabilize learning.
 
## 3. Real Drone Deployment (Tello SDK)
- The Tello drone is controlled by python library **DJITelloPy**.
- The Action output from the trained policy ([x, y]).
  - Action **x** was converted the lateral action in Gym environment to yaw velocity on the Tello.
  - Action **y** was converted the vertical action in Gym environent to z-axis velocity on the Tello.
- In this study, OpenCV ArUco markers were used for target detection.
However, as long as the target can be detected in the captured image and represented as a point coordinate, the tracking system can work with other detection methods, such as YOLO.

## Demonstration
### The trained agent in Gymnasium environment
![train](https://github.com/user-attachments/assets/f09c0c86-c8f3-418f-adec-00e917decda7)


### Real-World Deployment Experiments (Tello drone vision)
![dronecam](https://github.com/user-attachments/assets/e330d8b2-56d8-47e6-a494-834c31228bb6)

## Future Work

- Extension to 3D position tracking (including forward/backward motion).

- Incorporation of depth estimation or optical flow for richer state representation.

- Comparison with other policy-based algorithms (e.g., PPO, SAC).
