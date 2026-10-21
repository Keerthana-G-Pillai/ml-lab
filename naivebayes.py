from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text  import CountVectorizer
from sklearn.metrics import accuracy_score,precision_score,recall_score

a=fetch_20newsgroups(subset="train",remove=("headers","footers","quotes"))
b=fetch_20newsgroups(subset="train",remove=("headers","footers","quotes"))

x_train,y_train=a.data,a.target
x_test,y_test=b.data,b.target

vect=CountVectorizer(stop_words="english",min_df =2)
x_train1=vect.fit_transform(x_train)
x_test1=vect.transform(x_test)

mnb=MultinomialNB()
mnb.fit(x_train1,y_train)
mnb_pred=mnb.predict(x_test1)

vect1=CountVectorizer(stop_words="english",min_df=2,binary=True)
x_train2=vect1.fit_transform(x_train)
x_test2=vect1.transform(x_test)

bnb=BernoulliNB()
bnb.fit(x_train2,y_train)
bnb_pred=bnb.predict(x_test2)

print("Multinomial")
print("Accuracy:", accuracy_score(y_test, mnb_pred))
print("Precision:", precision_score(y_test, mnb_pred, average="weighted", zero_division=0))
print("Recall:", recall_score(y_test, mnb_pred, average="weighted", zero_division=0))

print("Accuracy:", accuracy_score(y_test, bnb_pred))
print("Precision:", precision_score(y_test, bnb_pred, average="weighted", zero_division=0))
print("Recall:", recall_score(y_test, bnb_pred, average="weighted", zero_division=0))

