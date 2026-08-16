from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_vectorize(train_text, test_text):
    vectorizer = TfidfVectorizer(max_features=10000)
    tfidf_train = vectorizer.fit_transform(train_text)
    tfidf_test =  vectorizer.transform(test_text)

    feature_names = vectorizer.get_feature_names_out()

    print("TF-IDF Completed!")

    return tfidf_train, tfidf_test, feature_names