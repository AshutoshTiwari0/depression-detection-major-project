# Depression Detection using NLP

This project demonstrates text classification models to detect signs of depression in short sentences using both **TF-IDF + Logistic Regression** and **Deep Learning with FastText embeddings**.

---

## 📊 Dataset

**Source:** [Depression Detection Dataset (Hugging Face)](https://huggingface.co/datasets)  
**Original size:** 120,000 samples (balanced)  
**Subset used for this project:** 20,000 samples (10k per class)  

**Reason for using subset:**  
- GitHub file size limitations (pickle files >100 MB not allowed)  
- Limited GPU resources for full dataset training  

**Balanced subset file:** `balanced_20k.csv`

---

## ⚙️ Methodology

### 1. Preprocessing
- Cleaned and balanced dataset.  
- Converted text into numerical features:
  - TF-IDF vectorization for Logistic Regression  
  - FastText embeddings for Deep Learning model  

---

### 2. Model Training

#### **A. TF-IDF + Logistic Regression**
- Classifier: Logistic Regression  
- Saved trained model and vectorizer as `model.pkl` (via joblib)

#### **B. Deep Learning Model (FastText embeddings + CNN)**
- **Architecture:**

```python
model_ft = Sequential()
model_ft.add(Embedding(
    input_dim=max_features,
    output_dim=embedding_dim,
    weights=[embedding_matrix],  # FastText embeddings
    input_length=max_len,
    trainable=True
))

model_ft.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model_ft.add(GlobalMaxPooling1D())
model_ft.add(Dense(128, activation='relu'))
model_ft.add(Dropout(0.5))
model_ft.add(Dense(1, activation='sigmoid'))

model_ft.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_ft.build(input_shape=(None, max_len))
model_ft.summary()
```
**Key points:**

1. Uses pre-trained FastText embeddings

2. CNN captures local n-gram features

3. GlobalMaxPooling reduces sequence dimension

4. Dropout prevents overfitting

### 3. Deployment

Streamlit app (app.py) for interactive predictions

Deployment-ready with Procfile, setup.sh, runtime.txt for Heroku

📈 Results (on 4k test set – Logistic Regression)
Metric	Class 0	Class 1	Avg.
Precision	0.90	0.91	0.90
Recall	0.92	0.89	0.90
F1-score	0.91	0.90	0.90
Accuracy	\multicolumn{3}{c}{90%}		

Note: Deep Learning model trained on the subset with FastText embeddings typically performs slightly better than TF-IDF + Logistic Regression and generalizes better on semantically complex sentences.

### 🚀 How to Run
```bash
# Clone the repository
git clone https://github.com/AshutoshTiwari0/depression-detection-minor-project.git
cd depression-detection-minor-project

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```
### 📝 Notes

1.Full dataset (120k) can yield higher accuracy (~96% with Random Forest or Deep Learning) but is not included due to size restrictions.

2.Demonstrates trade-off between model performance and deployment constraints.

3.Small subsets are often sufficient for prototyping; larger datasets improve learning.

### ✨ Future Improvements

1.Try advanced deep learning models (LSTMs, Transformers) for better semantic understanding.

2.Use contextual embeddings (BERT, RoBERTa) instead of Word2vec, FastText etc.

3.Collect more real-world test data to evaluate generalization.

4.Fine-tune transformers based genai models like BERT, GPT.
