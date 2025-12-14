# 📈 AI Trade Prediction - Stock Trend Forecasting

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

**AI Trade BY ANONY** là ứng dụng phân tích kỹ thuật và dự báo giá chứng khoán/tiền điện tử sử dụng trí tuệ nhân tạo (Deep Learning). Ứng dụng tích hợp mô hình **LSTM (Long Short-Term Memory)** để học xu hướng giá từ dữ liệu quá khứ và đưa ra dự đoán cho tương lai.

## ✨ Tính Năng Chính

* **📊 Dữ liệu thị trường Real-time:** Tải dữ liệu trực tiếp từ Yahoo Finance (Hỗ trợ Chứng khoán Mỹ, Việt Nam, Crypto, Forex...).
* **🧠 Mô hình Deep Learning LSTM:**
    * Sử dụng mạng nơ-ron hồi quy (RNN/LSTM) để xử lý chuỗi thời gian.
    * Tự động chuẩn hóa dữ liệu (MinMaxScaler) và huấn luyện trên trình duyệt.
* **📈 Chỉ báo kỹ thuật (Technical Indicators):**
    * **SMA (20, 50):** Đường trung bình động đơn giản.
    * **RSI (14):** Chỉ số sức mạnh tương đối.
    * **MACD:** Phân kỳ hội tụ trung bình động.
* **👀 Trực quan hóa tương tác:**
    * Biểu đồ nến (Candlestick) chuyên nghiệp với Plotly.
    * Biểu đồ so sánh giá Thực tế vs Dự báo (Backtesting).
    * Biểu đồ đánh giá độ mất mát (Loss) trong quá trình huấn luyện.
* **🔮 Dự báo tương lai:** Đưa ra mức giá dự kiến cho phiên tiếp theo và tính toán % tăng trưởng.
* Cách run :streamlit run test.py

## 🛠️ Yêu Cầu Hệ Thống

Để chạy dự án này, bạn cần cài đặt **Python 3.8** trở lên và các thư viện sau:

```text
numpy
pandas
yfinance
tensorflow
scikit-learn
streamlit
plotly

