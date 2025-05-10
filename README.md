# **Blenns Architecture Library 📈**  
*Blended Neural Networks for Next Day Candlestick Image Prediction*  

---

## **Table of Contents**  
1. [Overview](#overview)  
2. [Key Features](#key-features)  
3. [Installation](#installation)  
4. [Full Functionality Mode](#full-functionality-mode)  
5. [Limited Functionality Mode](#limited-functionality-mode)  
6. [Model Architecture](#model-architecture)  
7. [Troubleshooting](#troubleshooting)  
8. [License](#license)  

---

## **Overview**  
Blenns is an advanced deep learning framework that predicts next-day candlestick Image by analyzing:  

🟢 **OHLC Images** (Open-High-Low-Close) processed as 64x64 RGB matrices  
📊 **Trading Volumes** integrated as auxiliary input channels  
🧠 **Hybrid CNN-LSTM-Attention** architecture for spatiotemporal pattern recognition  

---

## **Key Features**  

| Feature | Full Mode | Limited Mode |  
|---------|----------|-------------|  
| Custom Model Training | ✅ | ❌ |  
| Pre-trained Predictions | ✅ | ✅ |  
| Candlestick Visualization | ✅ | ✅ |  
| ROC Curve Analysis | ✅ | ❌ |  
| Training Metrics | ✅ | ❌ |  
| Execution Time | 2-5 mins | <30 secs |  

---

## **Installation**  

### **1. Google Colab (Recommended)**  
```python
!git clone https://github.com/DataScienceCoach/BlennsLib
%cd BlennsLib
!pip install -r requirements.txt
!pip install .
```

### **2. Local Python Environment**  
```bash
git clone https://github.com/DataScienceCoach/BlennsLib.git
cd BlennsLib
pip install -r requirements.txt
pip install .
```

---

## **Full Functionality Mode**  
*For complete model training and evaluation*  

```python
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report  # Added critical imports
from blenns.model import BlennsModel
from blenns.utils import (encode_candle_chart, 
                         display_first_two_images,
                         plot_training_validation_loss,
                         plot_roc_curve,
                         plot_predicted_candlestick_image)

# 1. Initialize and fetch data
bm = BlennsModel()
data = bm.fetch_data("NVDA")  # Try "TSLA", "AAPL", etc.

# 2. Visualize candlestick patterns
plt.figure(figsize=(12, 4))
encoded_images, _ = encode_candle_chart(data)
display_first_two_images(encoded_images)
plt.tight_layout()
plt.show()

# 3. Train model
bm.train(data, epochs=10)

# 4. View training performance
plt.figure(figsize=(10, 4))
plot_training_validation_loss(bm.history)
plt.tight_layout()
plt.show()

# 5. Evaluate model
plt.figure(figsize=(10, 4))
plot_roc_curve(bm.model, bm.X_test_img, bm.X_test_vol, bm.y_test)
plt.tight_layout()
plt.show()

# 6. Model evaluation metrics
print("\n=== Model Evaluation ===")
y_pred = (bm.model.predict([bm.X_test_img, bm.X_test_vol]) > 0.5).astype(int)
print("Confusion Matrix:")
print(confusion_matrix(bm.y_test, y_pred))
print("\nClassification Report:")
print(classification_report(bm.y_test, y_pred))

# 7. Final prediction with visual
print("\n=== Final Prediction ===")
prediction = bm.predict_next_day()
plt.figure(figsize=(6, 6))
plot_predicted_candlestick_image(bm.X_test_img[:1])
plt.tight_layout()
plt.show()
```

**Outputs:**  
- First 2 candlestick patterns  
- Training/validation loss curves  
- ROC curve analysis  
- Confusion matrix & classification report  
- Predicted candlestick visualization  

---

## **Limited Functionality Mode**  
*For quick predictions without training*  

```python
%matplotlib inline
import matplotlib.pyplot as plt
from blenns.model import BlennsModel
from blenns.utils import plot_predicted_candlestick_image  # Import visualization function

bm = BlennsModel()
data = bm.fetch_data("AAPL")

bm.train(data, epochs=1)

#bm.predict_next_day()

# Make prediction and show visual
prediction = bm.predict_next_day()
plt.figure(figsize=(6, 6))
plot_predicted_candlestick_image(bm.X_test_img[:1])  # Visualize prediction
plt.show()
```

**Advantages:**  
⚡ No training required  
🖼️ Immediate candlestick visualization  
📉 Works with any US stock ticker  

---

## **Model Architecture**  

```mermaid
graph TD
    A[Input Images] --> B[TimeDistributed CNN]
    C[Volume Data] --> D[LSTM-Attention]
    B --> D
    D --> E[Fully Connected]
    E --> F[Sigmoid Output]
```

**Technical Specifications:**  
- **CNN Layers**: 2x (Conv2D + MaxPooling)  
- **LSTM Units**: 50 with Attention mechanism  
- **Output Activation**: Sigmoid (0=Bearish, 1=Bullish)  
- **Input Shape**: (1, 64, 64, 3) for images + (1,) for volume  

---

## **Troubleshooting**  

| Error | Solution |  
|-------|----------|  
| `ModuleNotFoundError` | Run `!pip install --force-reinstall .` |  
| Plots not showing | Add `%matplotlib inline` and `plt.show()` |  
| CUDA errors | Restart runtime and check GPU availability |  
| Shape mismatches | Verify `data.shape == (N, 64, 64, 3)` |  

---

## **License**  
MIT License - Free for academic and commercial use  

**Repository**: [github.com/DataScienceCoach/BlennsLib](https://github.com/DataScienceCoach/BlennsLib)  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DataScienceCoach/BlennsLib/blob/main/examples/demo.ipynb)  

---
**Disclaimer**: Predictions are for research purposes only. Past performance ≠ future results.
