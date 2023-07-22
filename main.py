from imblearn.over_sampling import SMOTE
import nltk
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('','',string.punctuation))

    # Tokenization
    tokens = word_tokenize(text)

    # Removing StopWords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

     # Joining tokens back into a single string
    preprocessed = ' '.join(tokens)

    return preprocessed

data = pd.read_csv('Dataset/mail_data.csv')

# print(data.info())
# print(data.shape())

# Encode the target variable 'Category' into integers (0 for 'spam' and 1 for 'ham')
label_encoder = LabelEncoder()
data['Category'] = label_encoder.fit_transform(data['Category'])

# Check the mapping between numerical labels and classes
# Generating pairs of numerical labels and their corresponding names
# 0:Spam, 1:Ham
# 'label_encoder.classes_' attribute contains ['ham', 'spam']
class_mapping = dict(enumerate(label_encoder.classes_))
print("Class Mapping:", class_mapping)


X = data['Message']
Y = data['Category']



# print(data)
# print(X)
# print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state = 3)

feature_extraction = TfidfVectorizer(min_df = 1, stop_words= 'english')
X_train_Vectorized = feature_extraction.fit_transform(X_train)
X_test_Vectorized = feature_extraction.transform(X_test)

# ~Create the Logistic Regression model with class weighting
#class_weights = {0: len(Y_train) / (2 * (Y_train == 0).sum()), 1: len(Y_train) / (2 * (Y_train == 1).sum())}
#clf = LogisticRegression(class_weight=class_weights)

clf = LogisticRegression(class_weight='balanced')
clf.fit(X_train_Vectorized, Y_train)

# ~Applying resampling using smote after class weighting - No Changes
# oversampler = SMOTE(sampling_strategy='auto', random_state=3)
# X_train_resampled, Y_train_resampled = oversampler.fit_resample(X_train_Vectorized, Y_train)
#clf.fit(X_train_resampled, Y_train_resampled)

# Fitting without balancing the data
# clf = LogisticRegression()
# clf.fit(X_train_Vectorized, Y_train)

# Make Predictions
y_pred = clf.predict(X_test_Vectorized)
# Measuring Accuracy with different evaluation metrics
score = clf.score(X_test_Vectorized, Y_test)
precision = precision_score(Y_test, y_pred)
recall = recall_score(Y_test, y_pred)
f1 = f1_score(Y_test, y_pred)

category_counts = data['Category'].value_counts()
print(category_counts)
category_counts = data['Category'].value_counts()
print("Number of ham emails:", category_counts[0])
print("Number of spam emails:", category_counts[1])


print("Accuracy:",score)
print("Precision:",precision)
print("Recall:",recall)
print("F1:",f1)

sns.countplot(x='Category', data=data)
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.xticks(ticks=[0, 1], labels=['ham', 'spam'])
plt.show()

# confusion matrix 'heatmap'
conf_matrix = confusion_matrix(Y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()