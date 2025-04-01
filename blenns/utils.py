# utils.py
from sklearn.metrics import roc_curve, auc  # <-- THIS IS THE CRITICAL FIX
import matplotlib.pyplot as plt
import io
import numpy as np
from PIL import Image
from mplfinance.original_flavor import candlestick_ohlc
from matplotlib.dates import date2num

# [Keep all your existing functions below...]
def encode_candle_chart(data):
    encoded_images = []
    volumes = []
    for index in range(5, len(data)):
        subset = data.iloc[index-5:index+1]
        fig, ax = plt.subplots()
        subset['Date'] = subset['Date'].apply(date2num)
        candlestick_ohlc(ax, subset[['Date', 'Open', 'High', 'Low', 'Close']].values, 
                        width=0.6, colorup='g', colordown='r')
        plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf).resize((64, 64)).convert('RGB')
        encoded_images.append(np.array(img) / 255.0)
        volumes.append(float(data.iloc[index]['Volume']))
        plt.close(fig)
    return np.array(encoded_images, dtype=np.float32), np.array(volumes, dtype=np.float32).reshape(-1, 1)

def display_first_two_images(encoded_images):
    for i in range(2):
        plt.imshow(encoded_images[i])
        plt.axis('off')
        plt.title(f"Candle Chart Image {i+1}")
        plt.show()

def plot_training_validation_loss(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.show()

def plot_roc_curve(model, X_test_img, X_test_vol, y_test):
    y_pred_probs = model.predict([X_test_img, X_test_vol])
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)  # Now works
    roc_auc = auc(fpr, tpr)  # Now works
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()

def plot_predicted_candlestick_image(X_img):
    plt.imshow(X_img[0, 0])
    plt.axis('off')
    plt.title('Predicted Candlestick Image')
    plt.show()
