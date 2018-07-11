from flask import Flask, Response, jsonify, request, json, url_for
import nltk
import bs4
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import string
import numpy as np
import urllib
import requests
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup
from langdetect import detect
from wordcloud import WordCloud
from pytagcloud import create_tag_image, make_tags
from pytagcloud.lang.counter import get_tag_counts
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

app = Flask(__name__,static_url_path='/static')
CORS(app)
SERVER_NAME = 'http://localhost:5000/'

class DetectSVM:


    def __init__(self,url,dewasa):
        self.dataset_dewasa =  pd.read_csv(dewasa, sep='\t', names=["label", "txt"])
        self.dewasa_txt_train, self.dewasa_txt_test, self.dewasa_label_train, self.dewasa_label_test = train_test_split(self.dataset_dewasa['txt'], self.dataset_dewasa['label'],test_size=0.7)
        self.linearSVM()
        self.url = url

    def linearSVM(self):
        def text_process(txt):
            noPunc = [char for char in txt if char not in string.punctuation]
            noPunc = ''.join(noPunc)
            return [kata for kata in noPunc.split() if kata.lower() not in stopwords.words('english')]

        pipeline = Pipeline([
            ('bow', CountVectorizer(analyzer=text_process)),
            ('tfidf', TfidfTransformer()),
            ('classifier', LinearSVC())
        ])
        self.svm = pipeline.fit(self.dewasa_txt_train,self.dewasa_label_train)

    def report2dict(self,cr):
        tmp = list()
        for row in cr.split("\n"):
            parsed_row = [x for x in row.split("  ") if len(x) > 0]
            if len(parsed_row) > 0:
                tmp.append(parsed_row)

        measures = tmp[0]

        D_class_data = defaultdict(dict)
        for row in tmp[1:]:
            class_label = row[0]
            for j, m in enumerate(measures):
                D_class_data[class_label][m.strip()] = float(row[j + 1].strip())
        return D_class_data

    def reportSVM(self):
        prediksi = self.svm.predict(self.dewasa_txt_test)
        cr = classification_report(prediksi,self.dewasa_label_test)
        return pd.DataFrame(self.report2dict(cr)).to_json()

    def scrap(self):
        def cleanMe(soup):
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            return text

        r = requests.get(self.url, headers={'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'}, timeout=15)
        soup = BeautifulSoup(r.content, "html.parser")
        self.cl = cleanMe(soup)
        self.dataScrap = self.cl.replace(',', '\n').replace('.', '\n').split('\n')

    def detectlang(self):
        numID = 0
        numEN = 0
        for num in range(len(self.dataScrap)):
            try:
                dtc = detect(self.dataScrap[num])
                if dtc == 'id':
                    numID = numID + 1
                elif dtc == 'en':
                    numEN = numEN + 1
            except:
                continue
        if numID > numEN:
            return 'Ini Website Indonesia'
        else:
            return 'ini website english'

    def wordc(self):
        wordcloud = WordCloud(background_color="white", width=800, height=400).generate(self.cl)
        plt.figure(figsize=(20, 10), facecolor='k')
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig('static/gambar/wordcloud.png', facecolor='k', bbox_inches='tight')

    def predict(self):
        datanya = self.svm.predict(self.dataScrap)
        presentase_positif_porn = len(datanya[datanya == 'porn']) / len(datanya) * 100
        return presentase_positif_porn
        '''
        if presentase_positif_porn >= 5:
            return 'ini website porno'
        else:
            return 'ini website biasa'
        '''


class DetectKNN:


    def __init__(self,url,dewasa):
        self.dataset_dewasa =  pd.read_csv(dewasa, sep='\t', names=["label", "txt"])
        self.dewasa_txt_train, self.dewasa_txt_test, self.dewasa_label_train, self.dewasa_label_test = train_test_split(self.dataset_dewasa['txt'], self.dataset_dewasa['label'],test_size=0.7)
        self.linearSVM()
        self.url = url

    def linearSVM(self):
        def text_process(txt):
            noPunc = [char for char in txt if char not in string.punctuation]
            noPunc = ''.join(noPunc)
            return [kata for kata in noPunc.split() if kata.lower() not in stopwords.words('english')]

        pipeline = Pipeline([
            ('bow', CountVectorizer(analyzer=text_process)),
            ('tfidf', TfidfTransformer()),
            ('classifier', KNeighborsClassifier())
        ])
        self.svm = pipeline.fit(self.dewasa_txt_train,self.dewasa_label_train)

    def report2dict(self,cr):
        tmp = list()
        for row in cr.split("\n"):
            parsed_row = [x for x in row.split("  ") if len(x) > 0]
            if len(parsed_row) > 0:
                tmp.append(parsed_row)

        measures = tmp[0]

        D_class_data = defaultdict(dict)
        for row in tmp[1:]:
            class_label = row[0]
            for j, m in enumerate(measures):
                D_class_data[class_label][m.strip()] = float(row[j + 1].strip())
        return D_class_data

    def reportSVM(self):
        prediksi = self.svm.predict(self.dewasa_txt_test)
        cr = classification_report(prediksi,self.dewasa_label_test)
        return pd.DataFrame(self.report2dict(cr)).to_json()

    def scrap(self):
        def cleanMe(soup):
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            return text

        r = requests.get(self.url, headers={'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'}, timeout=15)
        soup = BeautifulSoup(r.content, "html.parser")
        self.cl = cleanMe(soup)
        self.dataScrap = self.cl.replace(',', '\n').replace('.', '\n').split('\n')

    def detectlang(self):
        numID = 0
        numEN = 0
        for num in range(len(self.dataScrap)):
            try:
                dtc = detect(self.dataScrap[num])
                if dtc == 'id':
                    numID = numID + 1
                elif dtc == 'en':
                    numEN = numEN + 1
            except:
                continue
        if numID > numEN:
            return 'Ini Website Indonesia'
        else:
            return 'ini website english'

    def wordc(self):
        wordcloud = WordCloud(background_color="white", width=800, height=400).generate(self.cl)
        plt.figure(figsize=(20, 10), facecolor='k')
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig('static/gambar/wordcloud.png', facecolor='k', bbox_inches='tight')

    def predict(self):
        datanya = self.svm.predict(self.dataScrap)
        presentase_positif_porn = len(datanya[datanya == 'porn']) / len(datanya) * 100
        return presentase_positif_porn
        '''
        if presentase_positif_porn >= 5:
            return 'ini website porno'
        else:
            return 'ini website biasa'
        '''


class DetectMLP:


    def __init__(self,url,dewasa):
        self.dataset_dewasa =  pd.read_csv(dewasa, sep='\t', names=["label", "txt"])
        self.dewasa_txt_train, self.dewasa_txt_test, self.dewasa_label_train, self.dewasa_label_test = train_test_split(self.dataset_dewasa['txt'], self.dataset_dewasa['label'],test_size=0.7)
        self.linearSVM()
        self.url = url

    def linearSVM(self):
        def text_process(txt):
            noPunc = [char for char in txt if char not in string.punctuation]
            noPunc = ''.join(noPunc)
            return [kata for kata in noPunc.split() if kata.lower() not in stopwords.words('english')]

        pipeline = Pipeline([
            ('bow', CountVectorizer(analyzer=text_process)),
            ('tfidf', TfidfTransformer()),
            ('classifier', MLPClassifier(hidden_layer_sizes=(30,30,30)))
        ])
        self.svm = pipeline.fit(self.dewasa_txt_train,self.dewasa_label_train)

    def report2dict(self,cr):
        tmp = list()
        for row in cr.split("\n"):
            parsed_row = [x for x in row.split("  ") if len(x) > 0]
            if len(parsed_row) > 0:
                tmp.append(parsed_row)

        measures = tmp[0]

        D_class_data = defaultdict(dict)
        for row in tmp[1:]:
            class_label = row[0]
            for j, m in enumerate(measures):
                D_class_data[class_label][m.strip()] = float(row[j + 1].strip())
        return D_class_data

    def reportSVM(self):
        prediksi = self.svm.predict(self.dewasa_txt_test)
        cr = classification_report(prediksi,self.dewasa_label_test)
        return pd.DataFrame(self.report2dict(cr)).to_json()

    def scrap(self):
        def cleanMe(soup):
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            return text

        r = requests.get(self.url, headers={'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'}, timeout=15)
        soup = BeautifulSoup(r.content, "html.parser")
        self.cl = cleanMe(soup)
        self.dataScrap = self.cl.replace(',', '\n').replace('.', '\n').split('\n')

    def detectlang(self):
        numID = 0
        numEN = 0
        for num in range(len(self.dataScrap)):
            try:
                dtc = detect(self.dataScrap[num])
                if dtc == 'id':
                    numID = numID + 1
                elif dtc == 'en':
                    numEN = numEN + 1
            except:
                continue
        if numID > numEN:
            return 'Ini Website Indonesia'
        else:
            return 'ini website english'

    def wordc(self):
        wordcloud = WordCloud(background_color="white", width=800, height=400).generate(self.cl)
        plt.figure(figsize=(20, 10), facecolor='k')
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig('static/gambar/wordcloud.png', facecolor='k', bbox_inches='tight')

    def predict(self):
        datanya = self.svm.predict(self.dataScrap)
        presentase_positif_porn = len(datanya[datanya == 'porn']) / len(datanya) * 100
        return presentase_positif_porn
        '''
        if presentase_positif_porn >= 5:
            return 'ini website porno'
        else:
            return 'ini website biasa'
        '''

class DetectNB:


    def __init__(self,url,dewasa):
        self.dataset_dewasa =  pd.read_csv(dewasa, sep='\t', names=["label", "txt"])
        self.dewasa_txt_train, self.dewasa_txt_test, self.dewasa_label_train, self.dewasa_label_test = train_test_split(self.dataset_dewasa['txt'], self.dataset_dewasa['label'],test_size=0.7)
        self.linearSVM()
        self.url = url

    def linearSVM(self):
        def text_process(txt):
            noPunc = [char for char in txt if char not in string.punctuation]
            noPunc = ''.join(noPunc)
            return [kata for kata in noPunc.split() if kata.lower() not in stopwords.words('english')]

        pipeline = Pipeline([
            ('bow', CountVectorizer(analyzer=text_process)),
            ('tfidf', TfidfTransformer()),
            ('classifier', MultinomialNB())
        ])
        self.svm = pipeline.fit(self.dewasa_txt_train,self.dewasa_label_train)

    def report2dict(self,cr):
        tmp = list()
        for row in cr.split("\n"):
            parsed_row = [x for x in row.split("  ") if len(x) > 0]
            if len(parsed_row) > 0:
                tmp.append(parsed_row)

        measures = tmp[0]

        D_class_data = defaultdict(dict)
        for row in tmp[1:]:
            class_label = row[0]
            for j, m in enumerate(measures):
                D_class_data[class_label][m.strip()] = float(row[j + 1].strip())
        return D_class_data

    def reportSVM(self):
        prediksi = self.svm.predict(self.dewasa_txt_test)
        cr = classification_report(prediksi,self.dewasa_label_test)
        return pd.DataFrame(self.report2dict(cr)).to_json()

    def scrap(self):
        def cleanMe(soup):
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            return text

        r = requests.get(self.url, headers={'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'}, timeout=15)
        soup = BeautifulSoup(r.content, "html.parser")
        self.cl = cleanMe(soup)
        self.dataScrap = self.cl.replace(',', '\n').replace('.', '\n').split('\n')

    def detectlang(self):
        numID = 0
        numEN = 0
        for num in range(len(self.dataScrap)):
            try:
                dtc = detect(self.dataScrap[num])
                if dtc == 'id':
                    numID = numID + 1
                elif dtc == 'en':
                    numEN = numEN + 1
            except:
                continue
        if numID > numEN:
            return 'Ini Website Indonesia'
        else:
            return 'ini website english'

    def wordc(self):
        wordcloud = WordCloud(background_color="white", width=800, height=400).generate(self.cl)
        plt.figure(figsize=(20, 10), facecolor='k')
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig('static/gambar/wordcloud.png', facecolor='k', bbox_inches='tight')

    def predict(self):
        datanya = self.svm.predict(self.dataScrap)
        presentase_positif_porn = len(datanya[datanya == 'porn']) / len(datanya) * 100
        return presentase_positif_porn
        '''
        if presentase_positif_porn >= 5:
            return 'ini website porno'
        else:
            return 'ini website biasa'
        '''



lokasiDataSet = '/home/fadli/Anaconda Project/My Sentiment Analysist/dataset/dewasa_english.txt'

@app.route("/detect", methods=['POST'])
def detect():
    bacaDS = DetectSVM(request.form['url'],lokasiDataSet)
    report = json.loads(bacaDS.reportSVM())
    bacaDS.scrap()
    bacaDS.wordc()
    return jsonify({
        'bahasa': bacaDS.detectlang(),
        'report': report,
        'result':bacaDS.predict()
    })

@app.route("/detectknn", methods=['POST'])
def detectknn():
    bacaDS = DetectKNN(request.form['url'],lokasiDataSet)
    report = json.loads(bacaDS.reportSVM())
    bacaDS.scrap()
    bacaDS.wordc()
    return jsonify({
        'bahasa': bacaDS.detectlang(),
        'report': report,
        'result':bacaDS.predict()
    })

@app.route("/detectmlp", methods=['POST'])
def detectmlp():
    bacaDS = DetectMLP(request.form['url'],lokasiDataSet)
    report = json.loads(bacaDS.reportSVM())
    bacaDS.scrap()
    bacaDS.wordc()
    return jsonify({
        'bahasa': bacaDS.detectlang(),
        'report': report,
        'result':bacaDS.predict()
    })

@app.route("/detectnb", methods=['POST'])
def detectnb():
    bacaDS = DetectNB(request.form['url'],lokasiDataSet)
    report = json.loads(bacaDS.reportSVM())
    bacaDS.scrap()
    bacaDS.wordc()
    return jsonify({
        'bahasa': bacaDS.detectlang(),
        'report': report,
        'result':bacaDS.predict()
    })

@app.route("/gambar", methods=['GET'])
def gamber():
    nama = request.args.get('nama')
    uri = SERVER_NAME + 'static/gambar/' + nama + '.png'
    return "<img src='"+uri+"' class='img-responsive'>"

if __name__ == '__main__':
    app.run(debug=True)