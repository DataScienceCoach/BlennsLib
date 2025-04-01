# **Blenns Library** 📈  
*A Deep Learning Stock Prediction Tool with Candlestick Images and Volume*  

---

## **Table of Contents**  
1. [Overview](#overview)  
2. [Installation](#installation)  
3. [Quick Start](#quick-start)  
4. [Full Functionality Mode](#full-functionality-mode)  
5. [Limited Functionality Mode](#limited-functionality-mode)  
6. [Troubleshooting](#troubleshooting)  
7. [License](#license)  

---

## **Overview**  
Blenns is a Python library that predicts stock price movements using:  
✅ **Convolutional Neural Networks (CNNs)** for candlestick Image analysis  
✅ **LSTM + Attention** for time-series forecasting  
✅ **Volume data integration** for improved accuracy  

**Two Usage Modes:**  
1. **Full Mode** – Train & predict with custom data (requires GPU for best performance)  
2. **Limited Mode** – Run pre-trained predictions (faster, no GPU required)  

---

## **Installation**  

### **Option 1: Local Installation**  
```bash
git clone https://github.com/DataScienceCoach/BlennsLib.git
cd BlennsLib
pip install -r requirements.txt
pip install .
```

### **Option 2: Google Colab (Recommended)**  
```python
!git clone https://github.com/DataScienceCoach/BlennsLib.git
%cd BlennsLib
!pip install -r requirements.txt
!pip install .
```

---

## **Quick Start**  

### **Basic Prediction (Limited Mode)**  
```python
from blenns.model import BlennsModel

# Initialize and predict
bm = BlennsModel()
data = bm.fetch_data("AAPL")  # Get Apple stock data
prediction = bm.predict_next_day()  # Up/Down prediction
print(prediction)
```

---

## **Full Functionality Mode**  
*(For users who want to train custom models)*  

### **1. Full Training & Prediction**  
```python
from blenns.model import BlennsModel
from blenns.utils import display_first_two_images, plot_training_validation_loss

# Initialize
bm = BlennsModel()

# Fetch data
data = bm.fetch_data("TSLA")  # Try Tesla or any ticker

# Display candlestick patterns (first 2 days)
display_first_two_images(data)  

# Train model (10-50 epochs recommended)
bm.train(data, epochs=10)  

# Show training performance
plot_training_validation_loss(bm.history)  

# Make prediction
prediction = bm.predict_next_day()  
print(f"Tomorrow's prediction: {prediction}")
```

### **2. Advanced Features**  
```python
# Evaluate model performance
bm.evaluate()  

# Plot ROC Curve (requires sklearn)
from blenns.utils import plot_roc_curve
plot_roc_curve(bm.model, bm.X_test_img, bm.X_test_vol, bm.y_test)
```

---

## **Limited Functionality Mode**  
*(For quick predictions without training)*  

### **1. Pre-Trained Predictions**  
```python
from blenns.model import BlennsModel

bm = BlennsModel()
data = bm.fetch_data("MSFT")  # Microsoft stock

# Fast prediction (uses cached model)
prediction = bm.predict_next_day(pretrained=True)  
print(f"Prediction: {'↑ UP' if prediction > 0.5 else '↓ DOWN'}")
```

### **2. Visualization Only**  
```python
from blenns.utils import display_first_two_images

data = bm.fetch_data("GOOGL")  # Alphabet (Google)
display_first_two_images(data)  # View candlestick patterns
```

---

## **Troubleshooting**  

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `!pip install --force-reinstall .` |
| Plots not showing | Add `%matplotlib inline` (Colab) |
| Slow training | Use Google Colab with GPU (`Runtime > Change runtime type`) |
| ROC curve error | Ensure `from sklearn.metrics import roc_curve, auc` exists in `utils.py` |

---

## **License**  
MIT License - Free for personal and commercial use.  

**Contribute:** Found a bug? Open an issue or submit a PR!  

---
**Happy Trading!** 🚀  
*Disclaimer: Predictions are for educational purposes only. Trade at your own risk.*
