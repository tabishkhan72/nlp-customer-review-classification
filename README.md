# **🛍️ NLP Customer Review Classification**  

## **📌 Overview**  

This project applies **Natural Language Processing (NLP)** and **machine learning** techniques to predict whether a customer would **recommend** a product based on their **review**. Using the **Women’s Clothing E-Commerce Reviews Dataset** from Kaggle, we focus specifically on identifying **negative reviews** where customers would **not recommend** the product (**Recommended IND = 0**).  

---

## **📊 Dataset Details**  

- **Source**: [Kaggle - Women’s Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)  
- **Size**: **23,486 customer reviews**  
- **Key Columns**:  
  - **Review Text**: The main text input for NLP processing.  
  - **Recommended IND**: Binary target variable (**1 = Recommended, 0 = Not Recommended**).  
  - **Rating**: Customer-provided rating (1-5).  
  - **Review Title**: Short summary of the review.  

---

## **🔬 Problem Statement**  

The goal is to develop an **NLP-based classification model** that accurately predicts whether a customer **would not recommend** a product.  

**Key Challenges:**  
✔ Handling **imbalanced data** (fewer negative reviews than positive).  
✔ Extracting **meaningful insights** from **text-based reviews**.  
✔ Implementing **effective NLP preprocessing** to improve model accuracy.  

---

## **🛠 Technology Stack**  

| **Component**  | **Technology** |
|---------------|----------------|
| **Programming Language** | Python |
| **Libraries & Frameworks** | Scikit-Learn, TensorFlow, Keras, NLTK, SpaCy |
| **NLP Techniques** | Tokenization, Stopword Removal, TF-IDF, Word Embeddings |
| **Machine Learning Models** | Logistic Regression, Random Forest, SVM, LSTMs |
| **Evaluation Metrics** | Accuracy, Precision, Recall, F1-Score, ROC-AUC |
| **Data Processing** | Pandas, NumPy |

---

## **🚀 Approach & Methodology**  

### **1️⃣ Data Preprocessing**  
✔ Tokenization, stopword removal, and stemming/lemmatization.  
✔ Converting text to numerical features using **TF-IDF** and **Word Embeddings (Word2Vec, GloVe, BERT)**.  
✔ Handling class imbalance using **oversampling** or **weighted loss functions**.  

### **2️⃣ Model Training & Evaluation**  
✔ Tested multiple models including **Logistic Regression, Random Forest, SVM, LSTMs**.  
✔ Fine-tuned hyperparameters to optimize performance.  
✔ Used **cross-validation** and **AUC-ROC curves** for evaluation.  

---

## **📥 Installation & Usage**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/tabishkhan72/nlp-customer-review-classification.git
cd nlp-customer-review-classification
```

### **2️⃣ Install Dependencies**  
Ensure Python **3.8+** is installed, then run:  
```bash
pip install -r requirements.txt
```

### **3️⃣ Train & Evaluate the Model**  
```bash
python train_model.py
```

### **4️⃣ Predict on New Reviews**  
```bash
python predict.py --input "This product was amazing, I would buy again!"
```

---

## **📊 Results & Findings**  

✔ **Best-performing model: [Model Name]** achieving **[X]% F1-score** on test data.  
✔ Improved negative review detection accuracy by **[X]%** after **[preprocessing step/model optimization]**.  
✔ Identified key words influencing **negative recommendations**.  

---

## **📌 Future Enhancements**  

✅ Implement **Transformer models (BERT, RoBERTa) for better contextual understanding**.  
✅ Improve handling of **long and complex reviews**.  
✅ Develop a **real-time API for live sentiment classification**.  

---
