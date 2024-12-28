import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import streamlit as st

# Streamlit Web Application
def main():
    st.title("Stock Market Analysis and Prediction")
    
    # User inputs
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, GOOGL):", "AAPL")
    start_date = st.date_input("Select Start Date:", pd.to_datetime("2015-01-01"))
    end_date = st.date_input("Select End Date:", pd.to_datetime("2023-01-01"))
    future_days = st.slider("Predict Future Days:", min_value=1, max_value=30, value=7)

    if st.button("Fetch and Predict"):
        # Fetch data
        data = yf.download(stock_symbol, start=start_date, end=end_date)

        # Display current price
        latest_price = yf.Ticker(stock_symbol).history(period="1d")['Close'].iloc[-1]
        st.subheader(f"Current Price of {stock_symbol}: ${latest_price:.2f}")

        # Preprocess data
        data = data.dropna()
        close_prices = data['Close'].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices.reshape(-1, 1))

        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]

        # Function to create sequences
        def create_sequences(data, time_step):
            X, y = [], []
            for i in range(len(data) - time_step):
                X.append(data[i:i + time_step, 0])
                y.append(data[i + time_step, 0])
            return np.array(X), np.array(y)

        time_step = 60
        X_train, y_train = create_sequences(train_data, time_step)
        X_test, y_test = create_sequences(test_data, time_step)

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Build the improved LSTM model
        model = Sequential()
        model.add(Bidirectional(LSTM(units=100, return_sequences=True, input_shape=(time_step, 1))))
        model.add(Dropout(0.2))  # Dropout for regularization
        model.add(Bidirectional(LSTM(units=100, return_sequences=False)))
        model.add(Dropout(0.2))
        model.add(Dense(units=50))
        model.add(Dense(units=1))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Early stopping and learning rate scheduler to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.2, min_lr=0.0001)

        # Train the model
        history = model.fit(X_train, y_train, batch_size=64, epochs=100, validation_data=(X_test, y_test), verbose=1, 
                            callbacks=[early_stopping, lr_scheduler])

        # Plot training and validation loss to check for overfitting
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history.history['loss'], label='Training Loss')
        ax.plot(history.history['val_loss'], label='Validation Loss')
        ax.set_title('Model Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()

        # Pass the figure explicitly to st.pyplot()
        st.pyplot(fig)

        # Predict on test data
        predicted_prices = model.predict(X_test)
        predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        #Error metrics calculations
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        # After model prediction
        predicted_prices = model.predict(X_test)
        predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Calculate MSE, RMSE, MAE, R-squared, and MAPE
        mse = mean_squared_error(y_test_actual, predicted_prices)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_actual, predicted_prices)
        r2 = r2_score(y_test_actual, predicted_prices)
        mape = np.mean(np.abs((y_test_actual - predicted_prices) / y_test_actual)) * 100

        # Display the error metrics in Streamlit
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"R-squared (R²): {r2:.2f}")
        st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")


        # Predict future prices
        last_sequence = scaled_data[-time_step:].reshape(1, time_step, 1)
        future_predictions = []
        for _ in range(future_days):
            future_price = model.predict(last_sequence)[0, 0]
            future_predictions.append(future_price)
            last_sequence = np.append(last_sequence[:, 1:, :], [[[future_price]]], axis=1)

        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        # Plot results
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(data.index[train_size + time_step:], y_test_actual, label='Actual Prices', color='blue')
        ax.plot(data.index[train_size + time_step:], predicted_prices, label='Predicted Prices', color='red')
        ax.set_title(f'{stock_symbol} Stock Price Prediction')
        ax.set_xlabel('Date')
        ax.set_ylabel('Stock Price')
        ax.legend()

        # Display future predictions
        future_dates = pd.date_range(data.index[-1], periods=future_days + 1, freq='B')[1:]
        future_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_predictions.flatten()})
        st.write(future_df)

        # Pass the figure explicitly to st.pyplot()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
    
