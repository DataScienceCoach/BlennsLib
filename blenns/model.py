# model.py
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed, concatenate, Attention
from tensorflow.keras.optimizers import Adam
from .utils import encode_candle_chart  # Import from utils

class BlennsModel:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.history = None
        self.X_test_img = None
        self.X_test_vol = None
        self.y_test = None

    def fetch_data(self, ticker="TLRY"):
        """Get stock data from Yahoo Finance"""
        data = yf.download(ticker, start="2024-01-01", end=datetime.now().strftime('%Y-%m-%d'), interval="1d")
        return data.reset_index()

    def normalize_data(self, images, volumes):
        normalized_volumes = self.scaler.fit_transform(volumes)
        return images, normalized_volumes

    def build_model(self):
        img_input = Input(shape=(1, 64, 64, 3), name='Image_Input')
        vol_input = Input(shape=(1,), name='Volume_Input')

        x = TimeDistributed(Conv2D(32, (3, 3), activation='relu'))(img_input)
        x = TimeDistributed(MaxPooling2D((2, 2)))(x)
        x = TimeDistributed(Conv2D(64, (3, 3), activation='relu'))(x)
        x = TimeDistributed(MaxPooling2D((2, 2)))(x)
        x = TimeDistributed(Flatten())(x)

        lstm_output = LSTM(50, return_sequences=True)(x)
        attention_output = Attention()([lstm_output, lstm_output])
        flattened_output = Flatten()(attention_output)

        combined = concatenate([flattened_output, vol_input])
        output = Dense(1, activation='sigmoid')(combined)

        self.model = Model(inputs=[img_input, vol_input], outputs=output)
        self.model.compile(optimizer=Adam(learning_rate=0.001), 
                         loss='binary_crossentropy', 
                         metrics=['accuracy'])

    def train(self, data, epochs=50):
        encoded_images, volumes = encode_candle_chart(data)
        images, vols = self.normalize_data(encoded_images, volumes)
        
        X_img = images[:-1]
        X_vol = vols[:-1]
        y = np.array([1 if images[i+1][32, 32, 1] > images[i][32, 32, 1] else 0 
                    for i in range(len(images) - 1)], dtype=np.float32)

        X_img = X_img.reshape(X_img.shape[0], 1, 64, 64, 3)
        (self.X_train_img, self.X_test_img, 
         self.X_train_vol, self.X_test_vol, 
         y_train, self.y_test) = train_test_split(X_img, X_vol, y, 
                                                 test_size=0.2, 
                                                 random_state=42)

        if not self.model:
            self.build_model()
            
        self.history = self.model.fit(
            [self.X_train_img, self.X_train_vol],
            y_train,
            epochs=epochs,
            validation_data=([self.X_test_img, self.X_test_vol], self.y_test),
            batch_size=32
        )

    def evaluate(self):
        y_pred = (self.model.predict([self.X_test_img, self.X_test_vol]) > 0.5).astype(int)
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, y_pred))
        print("Classification Report:\n", classification_report(self.y_test, y_pred))
        mse = mean_squared_error(self.y_test, y_pred)
        print(f"Mean Squared Error (MSE): {mse:.4f}")

    def predict_next_day(self):
        prediction = self.model.predict([self.X_test_img[:1], self.X_test_vol[:1]])[0][0]
        color = 'Blue' if prediction > 0.5 else 'Grey'
        print(f"Predicted Next Day Candle: {'Upward' if prediction > 0.5 else 'Downward'} ({color})")
        return prediction
