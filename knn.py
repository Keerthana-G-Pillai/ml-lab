import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Load Fashion MNIST dataset
data = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
x = data.data
y = data.target.astype(int)

# 2. Select 12000 images
x, _, y, _ = train_test_split(
    x, y, train_size=12000, stratify=y, random_state=42
)

# 3. Split into training and testing data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=2000, stratify=y, random_state=42
)

# 4. Preprocess the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# 5. Test different K values
for k in [1, 3, 5, 7, 9, 15]:

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)

    start = time.time()
    y_pred = model.predict(x_test)
    prediction_time = time.time() - start

    accuracy = accuracy_score(y_test, y_pred)

    print("K =", k, "| Accuracy =", accuracy * 100,
          "% | Prediction time =", prediction_time, "seconds")
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# 7. Find the K value with highest accuracy
best_k = max(results, key=lambda k: results[k]["accuracy"])

print("\nBest K:", best_k)
print("Best accuracy:", round(results[best_k]["accuracy"] * 100, 2), "%")

# 8. Classification report
print("\nClassification Report:")

print(classification_report(
    y_test,
    results[best_k]["predictions"],
    target_names=class_names
))
