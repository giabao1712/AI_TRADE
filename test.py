import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import streamlit as st
from datetime import date
import plotly.graph_objects as go
import plotly.express as px
st.set_page_config(
    page_title="AI Trade BY ANONY",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #0e1117;
        border: 1px solid #30333d;
        border-radius: 5px;
        padding: 15px;
        color: white;
    }
    .stProgress > div > div > div > div {
        background-color: #00CC96;
    }
</style>
""", unsafe_allow_html=True)
def add_technical_indicators(df):
    data = df.copy()
    # SMA
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    # MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    data.dropna(inplace=True)
    return data

@st.cache_data
def load_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df = add_technical_indicators(df)
        return df
    except Exception as e:
        return None

def prepare_data(df, sequence_length=60):
    features = ['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD']
    data = df[features].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaler_target.fit(df[['Close']])
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])
        
    X, y = np.array(X), np.array(y)
    
    train_size = int(len(X) * 0.8)
    return X[:train_size], y[:train_size], X[train_size:], y[train_size:], scaler, scaler_target
# BUILD MODEL
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
#TRỰC QUAN HÓA (PLOTLY)
def plot_candlestick(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name='Market Data'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='orange', width=1), name='SMA 20'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='green', width=1), name='SMA 50'))
    
    fig.update_layout(
        title=f'Biểu đồ giá {ticker}',
        yaxis_title='Giá (Currency)',
        template='plotly_dark',
        height=500,
        xaxis_rangeslider_visible=False
    )
    return fig

def plot_prediction(y_test, y_pred, dates):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=y_test.flatten(), mode='lines', name='Thực tế', line=dict(color='#00CC96')))
    fig.add_trace(go.Scatter(x=dates, y=y_pred.flatten(), mode='lines', name='Dự báo AI', line=dict(color='#EF553B', dash='dot')))
    
    fig.update_layout(
        title='So sánh Giá Thực tế vs Dự báo AI',
        template='plotly_dark',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def main():
    with st.sidebar:
        st.header("🎛️ Bảng điều khiển")
        
        POPULAR_STOCKS = {
            "Mỹ - Apple (AAPL)": "AAPL",
            "Mỹ - Tesla (TSLA)": "TSLA",
            "Mỹ - NVIDIA (NVDA)": "NVDA",
            "Mỹ - Bitcoin": "BTC-USD",
            "Mỹ - Gold": "GC=F",
            "VN - Vingroup": "VIC.VN",
            "VN - FPT": "FPT.VN",
            "VN - Hòa Phát": "HPG.VN",
            "VN - Techcombank": "TCB.VN",
            "Khác...": "OTHER"
        }
        
        choice = st.selectbox("Chọn tài sản:", list(POPULAR_STOCKS.keys()))
        ticker = st.text_input("Nhập mã Ticker:", "AAPL") if choice == "Khác..." else POPULAR_STOCKS[choice]
        
        st.caption("Thời gian huấn luyện")
        col_date1, col_date2 = st.columns(2)
        start_date = col_date1.date_input("Từ", date(2020, 1, 1))
        end_date = col_date2.date_input("Đến", date.today())
        
        st.divider()
        st.caption("Thông số Model")
        epochs = st.slider("Epochs", 5, 50, 15)
        batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
        
        run_btn = st.button("🚀 PHÂN TÍCH & DỰ BÁO", type="primary", use_container_width=True)
    st.title("📈 AI Trade BY ANONY")
    st.markdown(f"Phân tích dữ liệu & Dự báo xu hướng cho **{ticker}**")

    if run_btn:
        with st.spinner(f'Đang kết nối dữ liệu máy chủ cho {ticker}...'):
            df = load_data(ticker, str(start_date), str(end_date))
        
        if df is None or len(df) < 60:
            st.error(f"❌ Không tìm thấy dữ liệu hoặc dữ liệu quá ngắn cho {ticker}.")
        else:
            last_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2]
            change = last_price - prev_price
            pct_change = (change / prev_price) * 100
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Giá đóng cửa", f"{last_price:,.2f}", f"{change:,.2f} ({pct_change:.2f}%)")
            col_m2.metric("Khối lượng", f"{df['Volume'].iloc[-1]:,}")
            col_m3.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.2f}")
            col_m4.metric("Dữ liệu mẫu", f"{len(df)} ngày")

            tab1, tab2, tab3 = st.tabs(["📊 Biểu đồ & Dự báo", "🔍 Phân tích Kỹ thuật", "⚙️ Chi tiết Model"])

            with tab1:
                #Biểu đồ nến
                st.plotly_chart(plot_candlestick(df, ticker), use_container_width=True)

                #Huấn luyện Model
                with st.status("🧠 AI đang suy nghĩ và học dữ liệu...", expanded=True) as status:
                    seq_len = 60
                    X_train, y_train, X_test, y_test, scaler, scaler_target = prepare_data(df, seq_len)
                    
                    st.write("🔹 Đang chuẩn bị tập dữ liệu Tensor...")
                    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
                    
                    st.write("🔹 Bắt đầu quá trình Gradient Descent...")
                    progress_bar = st.progress(0)
                    
                    class StreamlitCallback(tf.keras.callbacks.Callback):
                        def on_epoch_end(self, epoch, logs=None):
                            progress_bar.progress((epoch + 1) / epochs)
                    
                    history = model.fit(
                        X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_test, y_test), verbose=0, callbacks=[StreamlitCallback()]
                    )
                    status.update(label="✅ Huấn luyện hoàn tất!", state="complete", expanded=False)

                # Kết quả dự báo Test
                predictions = model.predict(X_test)
                predictions_inv = scaler_target.inverse_transform(predictions)
                y_test_inv = scaler_target.inverse_transform(y_test.reshape(-1, 1))
                
                # Tạo index ngày cho biểu đồ dự báo
                test_dates = df.index[-len(y_test):]
                
                st.subheader("Kết quả Kiểm thử (Backtest)")
                col_chart, col_stat = st.columns([3, 1])
                
                with col_chart:
                    st.plotly_chart(plot_prediction(y_test_inv, predictions_inv, test_dates), use_container_width=True)
                
                with col_stat:
                    rmse = np.sqrt(np.mean((predictions_inv - y_test_inv) ** 2))
                    st.info(f"**Độ sai lệch (RMSE):**\n\n {rmse:.2f}")
                    st.write("RMSE càng nhỏ, mô hình càng chính xác so với biến động giá.")

                # Dự báo tương lai
                st.markdown("---")
                last_seq = df[['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD']].values[-seq_len:]
                last_seq_scaled = scaler.transform(last_seq).reshape(1, seq_len, 5)
                future_pred = model.predict(last_seq_scaled)
                future_price = scaler_target.inverse_transform(future_pred)[0][0]

                # Card dự báo
                st.success(f"🔮 **Dự báo giá phiên tiếp theo:**")
                c1, c2 = st.columns([1, 4])
                with c1:
                    st.metric(label="AI Predicted", value=f"{future_price:,.2f}")
                with c2:
                    if future_price > last_price:
                        st.markdown(f"### 🔼 Tăng trưởng dự kiến: :green[+{((future_price-last_price)/last_price)*100:.2f}%]")
                    else:
                        st.markdown(f"### 🔽 Giảm dự kiến: :red[{((future_price-last_price)/last_price)*100:.2f}%]")

            with tab2:
                st.subheader("Dữ liệu chi tiết & Chỉ báo")
                st.dataframe(df.tail(100).style.highlight_max(axis=0), use_container_width=True)
                
                # Biểu đồ RSI/MACD riêng biệt nếu muốn
                fig_rsi = px.line(df.tail(100), x=df.tail(100).index, y='RSI', title="Chỉ số RSI (14)")
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                st.plotly_chart(fig_rsi, use_container_width=True)

            with tab3:
                st.subheader("Hiệu suất quá trình học")
                loss_df = pd.DataFrame(history.history)
                fig_loss = px.line(loss_df, title="Training vs Validation Loss")
                st.plotly_chart(fig_loss, use_container_width=True)
                
                st.json({
                    "Model": "LSTM (Long Short-Term Memory)",
                    "Optimizer": "Adam",
                    "Input Features": ["Close", "SMA_20", "SMA_50", "RSI", "MACD"],
                    "Epochs": epochs,
                    "Batch Size": batch_size
                })
    else:
        st.info("👈 Vui lòng chọn mã cổ phiếu và nhấn nút **PHÂN TÍCH** ở thanh bên trái.")
        st.markdown("### Hướng dẫn nhanh:")
        st.markdown("""
        1. Chọn mã chứng khoán (VD: **FPT.VN** hoặc **AAPL**).
        2. Chọn khoảng thời gian dữ liệu (Dữ liệu càng dài AI học càng tốt).
        3. Nhấn nút chạy và chờ AI xử lý.
        """)

if __name__ == "__main__":
    main()