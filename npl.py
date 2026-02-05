
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report
from sklearn.svm import LinearSVC
df=pd.read_csv('Instagram-labeled-comments.csv')

# print(df.head())

# داده‌های آموزشی را گسترش دهیم
extra_data = {
    'comment': [
        'نه خوب بود نه بد',
        'معمولی بود',
        'هم خوب بود هم بد',
        'نه خیلی خوب نه خیلی بد',
        'متوسط بود',
        'قابل قبول بود'
    ],
    'sentiment': [0, 0, 0, 0, 0, 0]  # همه خنثی
}

df_extra = pd.DataFrame(extra_data)
df = pd.concat([df, df_extra], ignore_index=True)

import re
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # حذف علائم
    text = re.sub(r'\d+', '', text)      # حذف اعداد
    return text.strip()
    
df['comment'] = df['comment'].apply(clean_text)





persian_stop_words = ['و', 'در', 'به', 'از', 'که', 'این', 'را', 'با', 'برای', 
                      'آن', 'یا', 'است', 'شد', 'های', 'نه', 'ولی', 'هم', 'همه']
vectorize = TfidfVectorizer(
    max_features=14000,
    stop_words=persian_stop_words,
    ngram_range=(1, 2),
    min_df=2
)
X=vectorize.fit_transform(df['comment'])
Y=df['sentiment']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=42,stratify=Y)

model = LinearSVC(class_weight='balanced', random_state=42)
model.fit(X_train,Y_train)

out=model.predict(X_test)

print(f'\n deghat kol : {accuracy_score(Y_test,out)*100:.1f}%')



test_texts = [
    "خیلی عالی بود واقعا لذت بردم",
    "بدترین تجربه زندگی بود", 
    "نه خوب بود نه بد",  # چالش اصلی!
    "قیمت مناسبی داشت ولی کیفیت متوسط",  # چالش دوم!
    "معمولی بود",  # تست جدید
    "هم خوب بود هم بد",  # تست جدید
]

print("\n🔍 تست جملات جدید:")
for text in test_texts:
    cleaned = clean_text(text)
    vec = vectorize.transform([cleaned])
    pred = model.predict(vec)[0]
    feelings = {-1: '😠 منفی', 0: '😐 خنثی', 1: '😊 مثبت'}
    print(f"• '{text[:25]}...' → {feelings[pred]}")

    # در فایل آموزش مدل (قبلی) اضافه کن:
import joblib
joblib.dump(model, 'sentiment_model.joblib')
joblib.dump(vectorize, 'vectorizer.joblib')