# NN Robo-vision
## Group Members: Francisco Morales Puente, Luis Mendoza, Haram Yoon

# Introduction
## Introductory sentence
Approximately 12.2% of Americans experience mobility disabilities, often requiring lifelong assistance [1]. These individulas have lost a great degree of their independence and individualism, and in order to regain that have to rely on technologyh. However, much of the world's current assistive technologies, like joystick-controlled wheelchairs or voice-command systems, pose limitations for individuals with limited dexterity [2].

## Supporting paragraphs (e.g., citation or quote)

 Eye-tracking offers an alternative control system, which allows for mobility without external limb movement. However, existing eye-tracking solutions often require expensive equipment like commercial eye-trackers or infrared cameras [5]. Luckily, CNNs yip yap about CNNS and how we can use them [4 and 3] alongside innovative approach from [6].

## Thesis
Our research aims to develop an affordable and accurate gaze estimator by leveraging neural networks trained on a diverse population to allow for high-accuracy between a diverse set of individuals, correctly identify general directions for user's gaze, and map the estimated gaze to a set of coordinates that will then idenitfy a user's current gaze on a screen.

# First body section
## Topic related to first point of thesis
## Supporting paragraphs

# Second body section
## Topic related to second point of thesis
## Supporting paragraphs

# Third body section
## Topic related to third point of thesis
## Supporting paragraphs

# Conclusion
## Restated thesis
Our research demonstrates that eye-tracking can serve as a cost-effective and accessible alternative for mobility assistance. By leveraging neural networks trained on a diverse population, we aim to improve gaze estimation accuracy across different individuals, enables precise gaze direction estimation, and even allows for coordinate mapping; laying the foundation for broader applications in assistive technology.

## Highlighted points
We anticipate that our research will yield several key outcomes and insights. Our approach focuses on developing a convolutional neural network capable of classifying gaze direction (forward, left, right, up, down) and eye status (blinking/open) with a sastisfactory target accuracy using standard webcams rather than specialized hardware. This would validate our approach of making assistive technology more accessible and affordable. After curating a diverse dataset collection process, we will use this diverse data in order to ensure the model performs consistently across various eye structures, lighting conditions, and head positions. We expect the solution to achieve real-time responsiveness, making it viable for practical mobility assistance applications, even with limited computational resources. Finally, we hypothesize that the preprocessing and normalization techniques we implement will be crucial for achieving high accuracy without specialized equipment in order to show effective eye-tracking doesn't necessarily require expensive hardware.



# Future work and additional questions
Future work will focus on integrating this model with a robotic system that facilitates real-time control through two-way SSH communication. This setup, combining a laptop, webcam, and Jetson Nano-powered robot, will allow users to navigate and interact with their environment using only their gaze, further expanding accessibility for individuals with mobility impairments. Additional research directions may include personalizing the model for individual users through adaptive calibration techniques, expanding the control interface to include more complex commands through combined eye movements, and implementing the system on mobile platforms to provide greater flexibility. Another future direction we could explore is the integration of multimodal inputs, such as voice commands or facial expressions, to enhance the system's capabilities and user experience. Finally, we could conduct comprehensive user studies with individuals who have mobility disabilities to see our effective approach is within real-world usage and gain valuable feedback.


# References:
[1] “Disability impacts all of us infographic,” Centers for Disease Control and Prevention, https://www.cdc.gov/disability-and-health/articles-documents/disability-impacts-all-of-us-infographic.html?CDC_AAref_Val=https%3A%2F%2Fwww.cdc.gov%2Fncbddd%2Fdisabilityandhealth%2Finfographic-disability-impacts-all.html (accessed Mar. 4, 2025). 

[2] M. Dahmani et al., “An intelligent and low-cost eye-tracking system for motorized wheelchair control,” Sensors, vol. 20, no. 14, p. 3936, Jul. 2020. doi:10.3390/s20143936 

[3] A. Kar and P. Corcoran, “A review and analysis of eye-gaze estimation systems, algorithms and performance evaluation methods in consumer platforms,” IEEE Access, vol. 5, pp. 16495–16519, 2017. doi:10.1109/access.2017.2735633 

[4] Y. Cheng, H. Wang, Y. Bao, and F. Lu, “Appearance-based gaze estimation with Deep Learning: A review and benchmark,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 12, pp. 7509–7528, Dec. 2024. doi:10.1109/tpami.2024.3393571 

[5] Y. Xia, B. Liang, Z. Li, and S. Gao, “Gaze estimation using neural network and logistic regression,” The Computer Journal, vol. 65, no. 8, pp. 2034–2043, May 2021. doi:10.1093/comjnl/bxab043 

[6] Y. Li, J. Chen, J. Ma, X. Wang, and W. Zhang, “Gaze estimation based on convolutional structure and sliding window-based attention mechanism,” Sensors, vol. 23, no. 13, p. 6226, Jul. 2023. doi:10.3390/s23136226 



# IGNORE THIS: 
## Related Works
### Research Articles:

* [Appearance-Based Gaze Estimation With Deep Learning: A Review and Benchmark](https://ieeexplore.ieee.org/document/10508472/) This paper goes over utilizing deep learning models for gaze estimation. It emphasizes how effective CNNs and transformers are at extracting gaze related features and how important it is to do image pre-processing to allow for model generalization. It also talks about how future research should focus on personalized calibration for users and identifying solutions for domain adaptation and variations in head poses/illumination.


* [Gaze Estimation Using Neural Network And Logistic Regression ](https://academic.oup.com/comjnl/article/65/8/2034/6269131?login=true) This research directly addresses the core challenge we're tackling that most existing eye-tracking solutions require expensive specialized hardware and complex setups, making them inaccessible to many users. The researchers propose a two-stage method combining deep learning with logistic regression that works on basic mobile platform cameras, without requiring additional hardware or expert knowledge using an automated data collection system and introduced a new annotation method to improve prediction accuracy. This aligns perfectly with our project's goals and their success shows the feasibility of the future direction of our project proposal. 

* [Gaze Estimation Based on Convolutional Structure and Sliding Window-Based Attention Mechanism](https://pmc.ncbi.nlm.nih.gov/articles/PMC10346721/) This research introduces two innovative approaches for gaze estimation: a pure Swin Transformer model (SwinT-GE) and a hybrid model combining convolutional structures with Swin Transformer (Res-Swin-GE)1. The Res-Swin-GE model demonstrated a 7.5% improvement over existing state-of-the-art methods on the Eyediap dataset and competitive performance on the MpiiFaceGaze dataset. These findings are particularly useful for us as it shows how combining CNN and transformer architectures can effectively preserve both local spatial features and global relationships in eye tracking. They have shown success with standard webcam-based datasets, which is what we will be using for our project.

* [A Review and Analysis of Eye-Gaze Estimation Systems, Algorithms and Performance Evaluation Methods in Consumer Platforms](https://ieeexplore.ieee.org/abstract/document/8003267) The paper reviews eye-gaze estimation in consumer devices like TVs, handheld devices, and web cameras. It covers geometric eye models, deep learning with CNNs, and hybrid methods. It examines accuracy metrics, datasets, and challenges like head movement and lighting.

* [An Intelligent and Low-Cost Eye-Tracking System for Motorized Wheelchair Control](https://pmc.ncbi.nlm.nih.gov/articles/PMC7412002/pdf/sensors-20-03936.pdf) This paper talks about a real-time, CNN based gaze estimation system for controlling a motorized wheelchair, utilizing infrared cameras for eye-tracking under varying lighting conditions and ultrasonic sensors for avoiding obstacles. The system includes a user specific calibration phase to optimize gaze classification accuracy. This CNN based model alongside feature extraction allows for an accuracy of about 96%, demonstrating its feasibility for precise and responsive robotic movement control.

* [Disability Impacts All of Us infographic](https://www.cdc.gov/disability-and-health/articles-documents/disability-impacts-all-of-us-infographic.html?CDC_AAref_Val=https://www.cdc.gov/ncbddd/disabilityandhealth/infographic-disability-impacts-all.html) A CDC infographic that shares statistics on disabilities in the United States.

