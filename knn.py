import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
a = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
x = a.data
y = a.target.astype(int)

# 2. Select 12000 images
x, _, y, _ = train_test_split(
    x, y, train_size=12000, stratify=y, random_state=42
)

# 3. Split data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=2000, stratify=y, random_state=42
)

# 4. Normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0

# 5. Test different K values
r = {}

for i in [1, 3, 5, 7, 9, 15]:
    m = KNeighborsClassifier(n_neighbors=i)
    m.fit(x_train, y_train)

    t = time.time()
    p = m.predict(x_test)
    t = time.time() - t
    acc = accuracy_score(y_test, p)
    r[i] = [acc,t,p]

# 6. Find best K
best = 1

for i in r:
    if r[i][0] > r[best][0]:
        best = i

# 7. Print outputs at the end
print("KNN Results")

for i in r:
    print("K =", i, "| Accuracy =", round(r[i][0] * 100,2), "%\t| Time =", r[i][1], "seconds")

print("\nBest K:", best)
print("Best Accuracy:", r[best][0] * 100, "%")

# 8. Classification report
names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

print("\nClassification Report:")
print(classification_report(y_test, r[best][2], target_names=names))
