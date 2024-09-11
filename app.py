
from flask import Flask, render_template, request
import pickle
import string
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('punkt')

app = Flask(__name__)

def transform_sms(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    return " ".join(y)

tfidf = pickle.load(open('vector.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')  

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
      
        input_sms = request.form['message']
        
        transformed_sms = transform_sms(input_sms)
       
        vector_input = tfidf.transform([transformed_sms])
       
        result = model.predict(vector_input)[0]
       
        if result == 1:
            prediction = "Spam"
        else:
            prediction = "Not Spam"
        
        return render_template('index.html', prediction=prediction)

if __name__=="__main__":
     app.run(host="0.0.0.0", debug = True)




