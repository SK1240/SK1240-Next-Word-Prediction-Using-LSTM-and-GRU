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

## ⚙️ Setup & Installation

1️⃣ Clone the Repository
```
git clone https://github.com/SK1240/Next-Word-Prediction-Using-LSTM-and-GRU.git
cd Next-Word-Prediction-using-GRU-LSTM
```

2️⃣ Create and Activate a Virtual Environment
```
python -m venv .venv
```
Activate the environment:

* Windows: `.venv\Scripts\activate`

* macOS/Linux: `source .venv/bin/activate`

3️⃣ Install Dependencies
```
pip install -r requirements.txt
```
💡 If you don’t have GPU support, use `tensorflow-cpu` instead of `tensorflow`.

### ▶️ How to Run the Apps

Launch the LSTM-based application:
```
streamlit run app_LSTM.py
```

Launch the GRU-based application:
```
streamlit run app_GRU.py
```
Once executed, Streamlit will start a local server (default: [localhost:8501](http://localhost:8501))

Type a short phrase in the text box, click “**Predict Next Word**”, and view the **model’s** generated suggestion instantly!

## ⚡ Behind the Scenes

Each app follows a streamlined prediction workflow:

* **Load Tokenizer** → Load `tokenizer.pkl` (used during training).

* **Preprocess Input** → Convert text to token indices.

* **Pad Sequences** → Adjust input to model’s expected length.

* **Model Inference** → Predict next token using the trained model.

* **Decode Prediction** → Convert predicted index back to its corresponding word.

## 💡 Usage Notes

* The **tokenizer** and **model** must belong to the same training session.

* Provide meaningful context (`2–5 words`) for accurate predictions.

* To fine-tune or retrain, open the training notebooks, modify parameters or **text corpus**, and re-save the updated model (`.h5`) and tokenizer (`tokenizer.pkl`).


## 🧪 Retraining Process

Each notebook (`experiments.ipynb` and `experiments_GRU.ipynb`) demonstrates:

* **Data Preparation** – Load and clean the text corpus.

* **Tokenization & Sequence Generation** – Map words to integers.

* **Model Construction** – Build **LSTM/GRU** layers using **Keras**.

* **Training & Evaluation** – **Optimize** using **categorical cross-entropy**.

* **Model Saving** – Export trained `.h5` model and `tokenizer.pkl`.

| Notebook                   | Purpose                                |
| -------------------------- | -------------------------------------- |
| 🧩 `experiments.ipynb`     | Training and testing of the **LSTM model** |
| ⚙️ `experiments_GRU.ipynb` | Training and testing of the **GRU model**  |


## 🏁 Summary

This project demonstrates how sequence modeling and neural text generation can be practically implemented using **LSTM** and **GRU** networks.

With minimal setup and intuitive UI, it serves as a foundation for building more advanced language models and predictive NLP applications.

## 📜 License

This repository is released for **educational** and **research purposes**.

Users are encouraged to **test**, **modify**, and extend the project responsibly before any production-level deployment.
