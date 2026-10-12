import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load dataset
train_data = fetch_20newsgroups(
    subset="train",
    remove=("headers", "footers", "quotes")
)

test_data = fetch_20newsgroups(
    subset="test",
    remove=("headers", "footers", "quotes")
)

x_train_text = train_data.data
y_train = train_data.target
x_test_text = test_data.data
y_test = test_data.target

# 2. Preprocess text for Multinomial NB
vectorizer = CountVectorizer(stop_words="english", min_df=2)

x_train = vectorizer.fit_transform(x_train_text)
x_test = vectorizer.transform(x_test_text)

# 3. Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(x_train, y_train)
y_pred_mnb = mnb.predict(x_test)

# 4. Preprocess text for Bernoulli NB
binary_vectorizer = CountVectorizer(
    stop_words="english", min_df=2, binary=True
)

x_train_binary = binary_vectorizer.fit_transform(x_train_text)
x_test_binary = binary_vectorizer.transform(x_test_text)

# 5. Bernoulli Naive Bayes
bnb = BernoulliNB()
bnb.fit(x_train_binary, y_train)
y_pred_bnb = bnb.predict(x_test_binary)

# 6. Evaluate both models
results = []
models = [
    ("Multinomial NB", y_pred_mnb),
    ("Bernoulli NB", y_pred_bnb)
]

for name, pred in models:
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score( y_test, pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, pred, average="weighted", zero_division=0 )
    f1 = f1_score(y_test, pred, average="weighted", zero_division=0)

    print("\n", name)
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    results.append([name, accuracy, precision, recall, f1])

# 7. Display comparison table
results = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
)

print("\nModel Performance Comparison")
print(results.to_string(index=False))
