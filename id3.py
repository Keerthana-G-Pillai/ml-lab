import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_excel("Online Retail.xlsx")

df = df.dropna(subset=["CustomerID"])
df = df[df["InvoiceNo"].astype(str).str[0] != "C"]
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

# 4. Group data by customer
data = df.groupby("CustomerID").agg(
    TotalSpend=("TotalAmount", "sum"),
    TotalQuantity=("Quantity", "sum"),
    Transactions=("InvoiceNo", "nunique")
).reset_index()

# 5. Create customer segments
data["Segment"] = pd.qcut(data["TotalSpend"],q=3,labels=["Low", "Medium", "High"])

# 6. Select features and target
x = data[["TotalQuantity", "Transactions"]]
y = data["Segment"]

# 7. Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 8. Create and train decision tree
model = DecisionTreeClassifier(criterion="entropy",max_depth=3,random_state=42)
model.fit(x_train, y_train)

# 9. Predict customer segments
y_pred = model.predict(x_test)

# 10. Calculate accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# 11. Display feature importance
importance = pd.Series(
    model.feature_importances_,
    index=x.columns
)

print("\nFeature Importance:")
print(importance.sort_values(ascending=False))
