# Importing the Dataset
import pandas as pd
messages = pd.read_csv('C:/Users/INDIA/.spyder-py3/Spam Classifier/sms+spam+collection_Dataset/SMSSpamCollection', sep='\t', names=["label", "message"])

# Data Cleaning and Preprocessing using stemming technique
# Data cleaning and preprocessing
import re
import nltk
# nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])  # Change 'mesage' to 'message'
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
