###Importing the Dataset
import pandas as pd
messages = pd.read_csv('C:/Users/INDIA/.spyder-py3/Spam Classifier/sms+spam+collection_Dataset/SMSSpamCollection', sep='\t', names=["label", "message"])




###Data Cleaning and Preprocessing using stemming technique
import re
import nltk
# nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    #remove everything apart from A~Z ,a~z
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    #using stemming technique
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
###Creating the Bag of Words(Document matrix) model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000)
X = cv.fit_transform(corpus).toarray()

#Converting ham and spam into dummy variables
y = pd.get_dummies(messages['label'])
y = y.iloc[:,1].values
#True indiactes spam & False indicates ham(!spam)


###Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

### Training model using Naive Bayes Classifier-->(classification technique)
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)

###Confusion matrix
from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test, y_pred)


###Accuracy score
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)