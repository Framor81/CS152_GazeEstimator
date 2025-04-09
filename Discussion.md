# NN Robo Vision
## Group Members: Francisco Morales Puente, Haram Yoon, Luis Mendoza

# Revised Introduction and Related Works

Approximately 12.2% of Americans experience mobility disabilities, often requiring lifelong assistance [^1]. These individuals face significant challenges in maintaining independence and autonomy, frequently relying on assistive technologies to regain control over their environment. However, many existing technologies, such as joystick-controlled wheelchairs or voice-command systems, are limited for individuals with restricted dexterity [^2].

Eye tracking presents a promising alternative, enabling individuals to navigate and interact with their surroundings without the need for external limb movement. However, current solutions often rely on overly expensive commercial eye-trackers, infrared cameras, or high-quality images which are inaccessible to many users [^5]. We believe that convolutional Neural Networks (CNNs) offer an opportunity to enhance affordability and accessibility by utilizing computer vision techniques for gaze estimation. CNN-based methods have already shown great adaptability to variations in lighting and head poses, which makes them suitable for real-world applications. Additionaly, using CNNs for gaze tracking eliminates the need for extensive user calibration, further increasing its accesibility [^3]. Recent works show that deep learning methods can improve gaze estimation accuracy, even in challenging conditions [^4]. Nonetheless, CNN based methods can lose important spatial information through pooling layers, limiting their ability to capture both local and global eye features. This has motivated researchers to investigate new architectures that preserve such details. In particular the Swin Transformer, has shown powerful modeling capabilities while keeping computational complexity lower with a window-based self-attention [^6]. By integrating innovative approaches like this sliding window-based attention mechanism, we could enhance the adaptability and precision of gaze-based control systems.

Our research aims to develop an affordable and accurate gaze estimator using neural networks trained on a diverse population. The system is designed to have high accuracy across a wide range of demographics, correctly identify gaze directions, and map the estimated gaze to a set of coordinates that determine the user’s focus on a screen. By integrating CNN-based techniques and the attention mechanisms described, we seek to improve the accuracy, affordability, and accessibility of gaze-based control systems.

# Methods Outline

1. We will aim to create our dataset by collecting images from peers using a google form. The form collects 6 images for each of our current classifications: Looking Up, Down, Left, Right, Forward, and Eyes Closed. This dataset will most likely be unreasonably small due to the effort needed for users to upload their own images, but we hope for at least a few hundred photos to utilize as maybe a test dataset.

2. We will then also in tandem utilize a separate dataset that already contains images of eyes and that have been labeled like [MRL Eye Dataset](https://data.mendeley.com/datasets/vy4n28334m/1) and [CEW Dataset](https://www.kaggle.com/datasets/imadeddinedjerarda/mrl-eye-dataset). The Mendeley eye_tracker_data dataset provides 7,500 evenly distributed eye images labeled in the five directional categories you need (left, right, up, down, and straight), captured using standard camera equipment. Additionally this, the MRL Eye Dataset offers over 300 MB of images specifically focused on distinguishing between open and closed eyes, providing blinking detection capability. Together, these datasets allows us to focus on building and training our nueral network rather without complex data preprocessing, while still ensuring sufficient diversity in eye shapes, lighting conditions, and head positions to create a robust model. In addition to this, we will create a Google form to collect images from peers to add to our diverse dataset and gain experience collecting and preprocessing data ourselves. This google form will ask for 6 images of each of the following categories: Looking Up, Down, Left, Right, Forward, and Eyes Closed. [Custom Google Form](https://docs.google.com/forms/d/e/1FAIpQLSekYRNat-GCG9UIBXVq8NYL8qZlgwyBSAy3fE4BrAxWiGq0SA/viewform)

3. We are utilizing an existing face detection model from dlib that applies landmarks to a user's face. We can then identify where a user's eyes are by focusing on certain landmarks. However, this model does not currently have funtionality for gaze identification which makes it perfect to work with.

4. We have implemented an existing python file with code that utilizes the pretrained model to detect a user's eyes and isolate them into a smaller window screen which also has a few linear transformations applied to it to uniformly orient all images. This pythoon file was developed in the Summer of 2024 by Francisco as part of his SURP project for Eyes In Motion.

5. Our project will not focus on finding the best classification accuracy possible instead we will want a decent accuracy where it won't inhibit movement controls when sending commands to a small scale robot through a rpc server.

6. Possible pitfalls might include data being biased for certain individuals or ethnic backgrounds and having a harder time for certain types of eyes. Moreover, it is entirely possible that we might also run into hardware related issues where the small scale robot that we intend to use is unable to properly run.

7. We hope to finish off by creating a small demo webpage where people can try controlling the robot themselves.    


# Discussion 

1. Francisco worked on implementing and creating a swin transformer following along a tutorial [Tutorial Link](https://medium.com/thedeephub/building-swin-transformer-from-scratch-using-pytorch-hierarchical-vision-transformer-using-shifted-91cbf6abc678)

2. We combined two datasets: one consisting of open eyes facing in one of 5 directions - left, right, down, up, and straight - and another dataset that had images of closed eyes. [Eyes Labeled With Direction](https://www.kaggle.com/datasets/imadeddinedjerarda/mrl-eye-dataset) [Closed Eyes](https://www.kaggle.com/datasets/serenaraju/yawn-eye-dataset-new)


# Conclusion
Our research demonstrates that eye-tracking can serve as a cost-effective and accessible alternative for mobility assistance. By leveraging neural networks trained on a diverse population, we aim to improve gaze estimation accuracy across different individuals, enables precise gaze direction estimation, and even allows for coordinate mapping; laying the foundation for broader applications in assistive technology.

We anticipate that our research will yield several key outcomes and insights. Our approach focuses on developing a convolutional neural network capable of classifying gaze direction (forward, left, right, up, down) and eye status (blinking/open) with a sastisfactory target accuracy using standard webcams rather than specialized hardware. This would validate our approach of making assistive technology more accessible and affordable. After curating a diverse dataset collection process, we will use this diverse data in order to ensure the model performs consistently across various eye structures, lighting conditions, and head positions. We expect the solution to achieve real-time responsiveness, making it viable for practical mobility assistance applications, even with limited computational resources. Finally, we hypothesize that the preprocessing and normalization techniques we implement will be crucial for achieving high accuracy without specialized equipment in order to show effective eye-tracking doesn't necessarily require expensive hardware.


# References:
[^1]: “Disability impacts all of us infographic,” Centers for Disease Control and Prevention, https://www.cdc.gov/disability-and-health/articles-documents/disability-impacts-all-of-us-infographic.html?CDC_AAref_Val=https%3A%2F%2Fwww.cdc.gov%2Fncbddd%2Fdisabilityandhealth%2Finfographic-disability-impacts-all.html (accessed Mar. 4, 2025). 

[^2]: M. Dahmani et al., “An intelligent and low-cost eye-tracking system for motorized wheelchair control,” Sensors, vol. 20, no. 14, p. 3936, Jul. 2020. doi:10.3390/s20143936 

[^3]: A. Kar and P. Corcoran, “A review and analysis of eye-gaze estimation systems, algorithms and performance evaluation methods in consumer platforms,” IEEE Access, vol. 5, pp. 16495–16519, 2017. doi:10.1109/access.2017.2735633 

[^4]: Y. Cheng, H. Wang, Y. Bao, and F. Lu, “Appearance-based gaze estimation with Deep Learning: A review and benchmark,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 12, pp. 7509–7528, Dec. 2024. doi:10.1109/tpami.2024.3393571 

[^5]: Y. Xia, B. Liang, Z. Li, and S. Gao, “Gaze estimation using neural network and logistic regression,” The Computer Journal, vol. 65, no. 8, pp. 2034–2043, May 2021. doi:10.1093/comjnl/bxab043 

[^6]: Y. Li, J. Chen, J. Ma, X. Wang, and W. Zhang, “Gaze estimation based on convolutional structure and sliding window-based attention mechanism,” Sensors, vol. 23, no. 13, p. 6226, Jul. 2023. doi:10.3390/s23136226 
