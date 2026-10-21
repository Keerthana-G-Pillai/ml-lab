
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Load dataset
a = load_breast_cancer()
x = a.data
y = a.target

print("Shape:", x.shape)
print("Classes:", np.unique(y))

# 2. Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# 3. Scale data
s = StandardScaler()
x_train = s.fit_transform(x_train)
x_test = s.transform(x_test)

# 4. MLE model (no regularization)
m1 = LogisticRegression(penalty=None, max_iter=5000, random_state=42)
m1.fit(x_train, y_train)
p1 = m1.predict(x_test)

# 5. MAP L2 model
m2 = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=5000, random_state=42)
m2.fit(x_train, y_train)
p2 = m2.predict(x_test)

# 6. MAP L1 model
m3 = LogisticRegression(penalty="l1", C=1.0, solver="liblinear", max_iter=5000, random_state=42)
m3.fit(x_train, y_train)
p3 = m3.predict(x_test)

# 7. Evaluate MLE
print("\nMLE")
print("Accuracy:", accuracy_score(y_test, p1))
print("Precision:", precision_score(y_test, p1))
print("Recall:", recall_score(y_test, p1))
print("F1 Score:", f1_score(y_test, p1))
print("Confusion Matrix:\n", confusion_matrix(y_test, p1))

# 8. Evaluate MAP L2
print("\nMAP L2")
print("Accuracy:", accuracy_score(y_test, p2))
print("Precision:", precision_score(y_test, p2))
print("Recall:", recall_score(y_test, p2))
print("F1 Score:", f1_score(y_test, p2))
print("Confusion Matrix:\n", confusion_matrix(y_test, p2))

# 9. Evaluate MAP L1
print("\nMAP L1")
print("Accuracy:", accuracy_score(y_test, p3))
print("Precision:", precision_score(y_test, p3))
print("Recall:", recall_score(y_test, p3))
print("F1 Score:", f1_score(y_test, p3))
print("Confusion Matrix:\n", confusion_matrix(y_test, p3))

# 10. Count zero coefficients
print("\nNumber of zero coefficients")
print("MLE:", np.sum(m1.coef_[0] == 0))
print("MAP L2:", np.sum(m2.coef_[0] == 0))
print("MAP L1:", np.sum(m3.coef_[0] == 0))
