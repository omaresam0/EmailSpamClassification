# Email Spam Classification 
# 1. Project Overview
The project aims to build a machine learning model that can classify emails as either spam or ham (non-spam).
The model uses Logistic Regression and employs various techniques to handle imbalanced data and achieve better accuracy in classifying emails.

# 2. Preprocessing and Label Encoding

**Lowercasing:** The text data is converted to lowercase to ensure consistency.

**Punctuation Removal:** Punctuation marks are removed from the text to eliminate unnecessary noise.

**Tokenization:** The text is tokenized into individual words or tokens to prepare it for further processing.

**Stopword Removal:** Stopwords, such as common words like "and" or "the," are removed from the text as they often don't carry significant meaning.

**Lemmatization:** Words are lemmatized to reduce them to their base or root form. This helps in reducing word variations and improving the consistency of the data.

These preprocessing steps help in cleaning the text data, reducing noise, and improving the quality of features for the machine learning model.

Additionally, the target variable 'Category' (spam or ham) was encoded using Label Encoding to convert it into numerical form    (0 for ham, 1 for spam) for model training.

# 3. Dealing with Imbalanced Data
The dataset was imbalanced where the number of spam instances was much smaller compared to the ham instances, affecting the model's performance.
# 4. Techniques to Handle Imbalanced Data
To address the imbalanced data, three techniques were explored:

**Class Weighting:** We applied class weighting to the Logistic Regression model to assign higher weight to the minority class (spam) during training, effectively balancing the importance of both classes.

**SMOTE (Synthetic Minority Over-sampling Technique):** SMOTE was used to oversample the minority class by creating synthetic samples, thereby increasing the number of spam instances and balancing the dataset.

**Modifying Class Weight Parameter:** We modified the 'class_weight' hyperparameter of the Logistic Regression model to 'balanced,' which automatically adjusts class weights based on the number of instances in each class.

The **class weight parameter** proved to be the simplest which generated same results as the other techniques with fewer lines of code.

Handling imbalanced data was crucial in improving model accuracy and performance. The techniques applied helped in correctly classifying both spam and ham emails, leading to a more balanced and reliable model which resulted in improved accuracy results.

# 5. Evaluation Metrics and Confusion Matrix
The model was evaluated using various metrics, including accuracy, F1-score, recall, and precision. Additionally, The model's performance was visualized using a confusion matrix, which provided insights into true positive, false positive, true negative, and false negative predictions.

# 6. Results
### Imbalanced Dataset ❎
- **Accuracy:** 0.965

- **Precision:** 1.0

- **Recall:** 0.754

- **F1:** 0.860
### Balanced Dataset ✅
- **Accuracy:** 0.981

- **Precision:** 0.952

- **Recall:** 0.909

- **F1:** 0.930

# Conclusion
The Email Spam Classification project demonstrated the importance of handling imbalanced data to achieve accurate and robust model performance. 
