# 🎨 Anime Face Generator using DCGAN

## 📌 Project Overview

This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** to generate realistic anime faces from random noise.

The model learns patterns from an anime face dataset and generates new images that resemble real anime characters.

---

## 🚀 Features

* Generate anime faces from random noise
* Deep learning using PyTorch
* GPU support for faster training
* Image generation tracking per epoch
* Loss and score visualization
* Training progress saved as video

---

## 🛠️ Tech Stack

* Python 🐍
* PyTorch 🔥
* Torchvision
* NumPy
* Matplotlib
* OpenCV

---

## 📂 Dataset

* Source: Kaggle Anime Face Dataset
* Link: https://www.kaggle.com/splcher/animefacedataset

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/anime-dcgan.git
cd anime-dcgan
```

### 2. Install Dependencies

#### GPU (Recommended)

```bash
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy matplotlib opendatasets
```

#### CPU

```bash
pip install torch torchvision torchaudio numpy matplotlib opendatasets
```

---

## 📥 Dataset Download

```python
import opendatasets as od
od.download('https://www.kaggle.com/splcher/animefacedataset')
```

---

## 🧠 Model Architecture

### 🔴 Discriminator

* CNN-based binary classifier
* Input: 64×64 image
* Output: Real (1) / Fake (0)

Layers:

* Conv → BatchNorm → LeakyReLU
* Progressive downsampling
* Sigmoid activation

---

### 🟢 Generator

* Transposed CNN (Deconvolution)
* Input: Random noise (latent vector)
* Output: 64×64 RGB image

Layers:

* ConvTranspose → BatchNorm → ReLU
* Tanh output layer

---

## 🔄 Training Process

1. Train Discriminator:

   * Classify real images as **1**
   * Classify fake images as **0**

2. Train Generator:

   * Generate fake images
   * Fool discriminator into predicting **1**

3. Loss Function:

   * Binary Cross Entropy Loss

4. Optimizer:

   * Adam (lr = 0.0002, betas = (0.5, 0.999))

---

## 📊 Results

### 📉 Loss Graph

* Generator and Discriminator losses tracked per epoch

### 📈 Score Graph

* Real vs Fake confidence scores

### 🖼️ Generated Images

* Images saved after each epoch in `/generated`

---

## 🎥 Training Visualization

* Training images compiled into video using OpenCV:

```bash
gans_training.avi
```

---

## 📁 Project Structure

```
anime-dcgan/
│
├── generated/            # Generated images
├── animefacedataset/     # Dataset
├── G.ckpt                # Generator weights
├── D.ckpt                # Discriminator weights
├── notebook.ipynb        # Main implementation
└── README.md
```

---

## ▶️ Run Training

```python
lr = 0.0002
epochs = 10

history = fit(epochs, lr)
```

---

## 💾 Model Saving

```python
torch.save(generator.state_dict(), 'G.ckpt')
torch.save(discriminator.state_dict(), 'D.ckpt')
```

---

## 🔮 Future Improvements

* Train for more epochs (100+ for better quality)
* Use higher resolution images (128×128)
* Implement StyleGAN
* Add conditional GAN (cGAN)
* Deploy as web app (Streamlit / React)

---

## 🧠 Key Learnings

* GAN training stability challenges
* Generator vs Discriminator balancing
* Image normalization importance
* Deep CNN architecture design

---
