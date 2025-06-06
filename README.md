# Greenhouse Microclimate Mapping with CNN-RNN

This project implements hybrid neural network models combining CNNs and RNNs (LSTM/BiLSTM) to estimate temperature in greenhouse environments using limited sensors and contextual time series data.

📄 **Based on:**  
**"Combining Convolutional and Recurrent Neural Networks to Improve Greenhouse Microclimate Mapping"**  
Sharifi, Migliorini, Quaglia – IEEE MetroAgriFor 2024  
[IEEE Link](https://ieeexplore.ieee.org/document/10948753)

## 🔍 Summary

- Combines temporal (LSTM) and spatial (CNN) modeling.
- Three hybrid architectures:
  - LSTM → CNN
  - CNN → LSTM
  - Parallel CNN-LSTM
- Tested on real greenhouse data from Verona, Italy.

## 🧠 Input Features

- 26 external weather features
- 3 internal sensor features
- 7 time-based features
- 1 spatial distance feature


## 🚀 Usage

```bash
# Clone and install
git clone https://github.com/yourusername/greenhouse-mapping.git
cd greenhouse-mapping
pip install -r requirements.txt

