# Title: NN Robo-vision Literature Review

### Group Members: Francisco Morales, Luis Mendoza, and Haram Yoon

## Related Works

### Research Articles:

* [Appearance-Based Gaze Estimation With Deep Learning: A Review and Benchmark](https://ieeexplore.ieee.org/document/10508472/) This paper goes over utilizing deep learning models for gaze estimation. It emphasizes how effective CNNs and transformers are at extracting gaze related features and how important it is to do image pre-processing to allow for model generalization. It also talks about how future research should focus on personalized calibration for users and identifying solutions for domain adaptation and variations in head poses/illumination.


* [Gaze Estimation Using Neural Network And Logistic Regression ](https://academic.oup.com/comjnl/article/65/8/2034/6269131?login=true) This research directly addresses the core challenge we're tackling that most existing eye-tracking solutions require expensive specialized hardware and complex setups, making them inaccessible to many users. The researchers propose a two-stage method combining deep learning with logistic regression that works on basic mobile platform cameras, without requiring additional hardware or expert knowledge using an automated data collection system and introduced a new annotation method to improve prediction accuracy. This aligns perfectly with our project's goals and their success shows the feasibility of the future direction of our project proposal. 

* [Gaze Estimation Based on Convolutional Structure and Sliding Window-Based Attention Mechanism](https://pmc.ncbi.nlm.nih.gov/articles/PMC10346721/) This research introduces two innovative approaches for gaze estimation: a pure Swin Transformer model (SwinT-GE) and a hybrid model combining convolutional structures with Swin Transformer (Res-Swin-GE)1. The Res-Swin-GE model demonstrated a 7.5% improvement over existing state-of-the-art methods on the Eyediap dataset and competitive performance on the MpiiFaceGaze dataset. These findings are particularly useful for us as it shows how combining CNN and transformer architectures can effectively preserve both local spatial features and global relationships in eye tracking. They have shown success with standard webcam-based datasets, which is what we will be using for our project.

* [A Review and Analysis of Eye-Gaze Estimation Systems, Algorithms and Performance Evaluation Methods in Consumer Platforms](https://ieeexplore.ieee.org/abstract/document/8003267) The paper reviews eye-gaze estimation in consumer devices like TVs, handheld devices, and web cameras. It covers geometric eye models, deep learning with CNNs, and hybrid methods. It examines accuracy metrics, datasets, and challenges like head movement and lighting.

* [An Intelligent and Low-Cost Eye-Tracking System for Motorized Wheelchair Control](https://pmc.ncbi.nlm.nih.gov/articles/PMC7412002/pdf/sensors-20-03936.pdf) This paper talks about a real-time, CNN based gaze estimation system for controlling a motorized wheelchair, utilizing infrared cameras for eye-tracking under varying lighting conditions and ultrasonic sensors for avoiding obstacles. The system includes a user specific calibration phase to optimize gaze classification accuracy. This CNN based model alongside feature extraction allows for an accuracy of about 96%, demonstrating its feasibility for precise and responsive robotic movement control.






