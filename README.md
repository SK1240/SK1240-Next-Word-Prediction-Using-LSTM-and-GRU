# 📝 Next Word Prediction using LSTM & GRU
A sleek and educational implementation of **next-word prediction** powered by two advanced **Recurrent Neural Network (RNN)** architectures — **Long Short-Term Memory (LSTM)** and **Gated Recurrent Unit (GRU)**.

This project combines **deep learning** and **natural language processing (NLP)** to predict the most likely next word in a given phrase.

It includes **Streamlit**-based **web applications** for **real-time text inference**, **pretrained models**, and interactive training notebooks for further experimentation.

## 🚀 Project Highlights

| Feature                    | Description                                                                  |
| -------------------------- | ---------------------------------------------------------------------------- |
| 🧩 **Dual Models**         | **LSTM** and **GRU** architectures implemented for comparative next-word prediction. |
| 💻 **Interactive Apps**    | **Two Streamlit applications** for hands-on testing of the models.               |
| 🧠 **Pretrained Models**   | Ready-to-use `.h5` model files and corresponding `tokenizer.pkl` tokenizer.   |
| 📘 **Notebooks Included**  | Full training and retraining workflows for both models.                      |
| 🔄 **End-to-End Pipeline** | Text preprocessing → tokenization → prediction → decoding.                   |

## 📂 Repository Structure

```
Next-Word-Prediction-using-LSTM-GRU/
├── app_LSTM.py               # Streamlit app for LSTM-based inference
├── app_GRU.py                # Streamlit app for GRU-based inference
├── model_LSTM.h5             # Pretrained LSTM model
├── model_GRU.h5              # Pretrained GRU model
├── tokenizer.pkl             # Tokenizer mapping words to indices
├── hamlet.txt                # Sample text dataset used for training
├── experiments.ipynb         # LSTM training and experiment notebook
├── experiments_GRU.ipynb     # GRU training and experiment notebook
└── requirements.txt          # List of required dependencies
```
