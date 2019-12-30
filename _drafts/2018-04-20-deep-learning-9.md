---
list_title: 深度学习 | Face Verification | Neural Style Transfer
title: Face Verification | Neural Style Transfer
layout: post
mathjax: true
categories: ["AI", "Machine Learning","Deep Learning"]
---

人脸识别的一个挑战是如何在只有一张人脸图片的情况下，识别出当前的人是否是图片中的人。也就是所谓的one Shot Learning，其本质找到两张图片的差异，使这个差值越小越好

$$
d(img1, img2) = degree of difference between images
$$

有了上面的式子，我们的目的便是构建一个CNN网络来生成一个上述模型

## Resources

- Florian Schroff, Dmitry Kalenichenko, James Philbin (2015). [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)
- Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf (2014). [DeepFace: Closing the gap to human-level performance in face verification](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf) 
- The pretrained model we use is inspired by Victor Sy Wang's implementation and was loaded using his code: https://github.com/iwantooxxoox/Keras-OpenFace.
- Our implementation also took a lot of inspiration from the official FaceNet github repository: https://github.com/davidsandberg/facenet 