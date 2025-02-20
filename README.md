CIFAR-10 EfficientNet-Lite GAN - Image Generation with TensorFlow
Overview
This repository contains an implementation of a Generative Adversarial Network (GAN) that generates images based on the CIFAR-10 dataset. The model leverages EfficientNet-Lite for feature extraction and TensorFlow for training and evaluation.

Features
✅ Uses EfficientNet-Lite for feature extraction
✅ Generates high-quality images from CIFAR-10
✅ Built with TensorFlow & Keras
✅ Supports training and inference

Dataset
The model is trained on CIFAR-10, which consists of 60,000 images (32x32 pixels) across 10 classes, including:
🚗 Airplanes, Cars, Birds, Cats, Deer, Dogs, Frogs, Horses, Ships, and Trucks.

Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/fadl1233/CIFAR-10-EfficientNet-Lite-GAN.git
cd CIFAR-10-EfficientNet-Lite-GAN
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Training the GAN
To train the GAN model, run:

bash
Copy
Edit
python train.py
You can adjust hyperparameters in config.py.

Generating Images
After training, generate images using:

bash
Copy
Edit
python generate.py
Generated images will be saved in the outputs/ folder.

Model Architecture
The GAN consists of:
🔹 Generator: Uses transposed convolutions to create images.
🔹 Discriminator: A convolutional network to classify real vs. fake images.
🔹 Feature Extractor: EfficientNet-Lite helps improve feature learning.

Results
Generated images after training for X epochs:
📸 Sample results will be displayed here.

Contributors
👤 FADHL 
📧 fadlcom.2025@gmail.com
🔗 LinkedIn   www.linkedin.com/in/fadhl-ghaleb-730073286

License
This project is licensed under the MIT License.

