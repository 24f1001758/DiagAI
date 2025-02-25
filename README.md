# DiagAI
Chest X-ray Pneumonia Detection 🩺📸

This project trains a deep learning model to classify Chest X-ray images as Normal or Pneumonia using a ResNet-18 model in PyTorch.

📂 Dataset
The dataset used is from Kaggle:
🔗 Chest X-ray Pneumonia Dataset - https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The dataset is structured as follows:
chest_xray/
│── train/
│   ├── NORMAL/       # Normal Chest X-rays
│   ├── PNEUMONIA/    # Pneumonia Chest X-rays
│── test/
│   ├── NORMAL/
│   ├── PNEUMONIA/
│── val/
│   ├── NORMAL/
│   ├── PNEUMONIA/
For quick testing, a smaller dataset (chest_xray_small/) with 200 images is used.
⚙️ Setup & Installation

1️⃣ Clone the Repository
git clone https://github.com/DiagAI/chest-xray-classification.git
cd chest-xray-classification
2️⃣ Install Dependencies
Ensure you have Python 3.8+ and install required libraries:
pip install -r requirements.txt
If OpenCV (cv2) doesn't install, use:
pip install opencv-python
🚀 Training the Model

Run the script to train the ResNet-18 model:
python train.py
The model will train for 5 epochs.
The trained model will be saved as model.pth.

📜 Directory Structure

chest-xray-classification/
│── chest_xray/            # Full dataset
│── chest_xray_small/      # Small dataset (200 images)
│── model.py               # Model architecture
│── train.py               # Training script
│── predict.py             # Inference script
│── requirements.txt       # Dependencies
│── README.md              # This file

📌 Notes

If you encounter FileNotFoundError, ensure the dataset is properly extracted inside chest_xray/.
You can adjust batch size, learning rate, or epochs inside train.py for better accuracy.

🛠 Future Improvements
Use data augmentation for better generalization.
Implement Grad-CAM for visual model interpretation.
Deploy model using Flask API for real-time predictions.
👩‍💻 Author

👤 Soha Farhana
📧 Contact: sohafarhana@gmail.com
