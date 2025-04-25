import os
import joblib
import pandas as pd
import re

# Ruta relativa al .pkl
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'phishing_clf.pkl')
clf = joblib.load(MODEL_PATH)

keywords = ['https', 'login', '.php', '.html', '@', 'sign']

def extract_features(url):
    data = {}
    for kw in keywords:
        data['keyword_' + kw] = int(kw in url)
    data['lenght'] = len(url) - 2
    parts = url.split('/')
    domain = parts[2] if len(parts) > 2 else ''
    data['lenght_domain'] = len(domain)
    data['isIP'] = int(re.sub(r'\.', '', domain).isnumeric())
    data['count_com'] = url.count('com')
    return pd.DataFrame([data])

def predict_proba(url):
    X_new = extract_features(url)
    return clf.predict_proba(X_new)[0][1]
