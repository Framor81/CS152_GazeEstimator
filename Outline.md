# NN Robo-vision
## Group Members: Francisco Morales Puente, Luis Mendoza, Haram Yoon

# Introduction
## Introductory sentence
Approximately 12.2% of Americans experience mobility disabilities, often requiring lifelong assistance [1]. These individuals face significant challenges in maintaining independence and autonomy, frequently relying on assistive technologies to regain control over their environment. However, many existing technologies, such as joystick-controlled wheelchairs or voice-command systems, are limited for individuals with restricted dexterity [2].

## Supporting paragraphs (e.g., citation or quote)

Eye tracking presents a promising alternative, enabling individuals to navigate and interact with their surroundings without the need for external limb movement. However, current solutions often rely on overly expensive commercial eye-trackers or infrared cameras, which are inaccessible to many users [5]. We believe that convolutional Neural Networks (CNNs) offer an opportunity to enhance affordability and accessibility by utilizing computer vision techniques for gaze estimation [3]. Recent works show that deep learning methods can improve gaze estimation accuracy, even in challenging conditions [4]. Additionally, by integrating innovative approaches like a sliding window-based attention mechanism, we could enhance the adaptability and precision of gaze-based control systems [6].
## Thesis
Our research aims to develop an affordable and accurate gaze estimator using neural networks trained on a diverse population. The system is designed to have high accuracy across a wide range of demographics, correctly identify gaze directions, and map the estimated gaze to a set of coordinates that determine the user’s focus on a screen. By integrating CNN-based techniques and the attention mechanisms described, we seek to improve the accuracy, affordability, and accessibility of gaze-based control systems.

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
