#datasets
https://www.kaggle.com/datasets/pablomgomez21/drugs-a-b-c-x-y-for-decision-trees
https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset


# Title: Decision Tree Classifier for Drug Classification
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
data = pd.read_csv("/content/drug200.csv")

# Drop duplicates
data.drop_duplicates(inplace=True)

# Label encode categorical variables
le = LabelEncoder()
data["Sex"] = le.fit_transform(data["Sex"])            # Female=0, Male=1
data["BP"] = le.fit_transform(data["BP"])              # High/Low/Normal
data["Cholesterol"] = le.fit_transform(data["Cholesterol"])  # High/Normal

# Define features and target
X = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y = data['Drug']  # No one-hot encoding needed for DecisionTreeClassifier

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# Train Decision Tree model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(x_train, y_train)

# Visualize the decision tree
plt.figure(figsize=(16, 8))
plot_tree(dt, filled=True, feature_names=X.columns, class_names=dt.classes_)
plt.show()

# Evaluate the model
y_pred = dt.predict(x_test)

# Print classification metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))




#Naive Bayes Classifier for Drug Classification
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
data = pd.read_csv("/content/drug200.csv")

# Drop duplicates
data.drop_duplicates(inplace=True)

# Label encode categorical variables
le = LabelEncoder()
data["Sex"] = le.fit_transform(data["Sex"])            # Female=0, Male=1
data["BP"] = le.fit_transform(data["BP"])              # High=0, Low=1, Normal=2
data["Cholesterol"] = le.fit_transform(data["Cholesterol"])  # High=0, Normal=1

# Define features and target
X = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y = data['Drug']  # Target is categorical but NOT one-hot encoded

# Split the data
x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

# Train Naive Bayes classifier
nb = GaussianNB()
nb.fit(x_train, y_train)

# Predict and evaluate
y_pred = nb.predict(x_test)

# Print classification report
print("Classification Report (Naive Bayes):")
print(classification_report(y_test, y_pred))

# Print accuracy
print("Accuracy (Naive Bayes):", accuracy_score(y_test, y_pred))












# Apriori Algorithm for Association Rule Mining
import pandas as pd
from itertools import combinations

# Load and preprocess dataset
df = pd.read_csv("/content/Groceries_dataset.csv")
transactions = df.groupby('Member_number')['itemDescription'].apply(set).tolist()

# Support function
def get_support(itemset):
    return sum(itemset <= t for t in transactions) / len(transactions)

# Initialize
all_items = {item for t in transactions for item in t}
frequent_itemsets = []
k = 1
candidates = [frozenset([item]) for item in all_items]

# Apriori loop
while candidates:
    valid_sets = {i: get_support(i) for i in candidates if get_support(i) >= 0.02}
    if not valid_sets:
        break
    frequent_itemsets.append(valid_sets)
    previous_level = list(valid_sets)
    candidates = list({a | b for a in previous_level for b in previous_level if len(a | b) == k + 1})
    k += 1

# Print frequent itemsets
for level_num, level in enumerate(frequent_itemsets, 1):
    print(f"\nFrequent {level_num}-itemsets:")
    for itemset, support in level.items():
        print(f"{set(itemset)}: {support:.2f}")

# Generate and print association rules
print("\nAssociation Rules:")
for level in frequent_itemsets[1:]:
    for itemset, sup in level.items():
        for i in range(1, len(itemset)):
            for lhs in combinations(itemset, i):
                lhs = frozenset(lhs)
                rhs = itemset - lhs
                conf = sup / get_support(lhs)
                if conf >= 0.5:
                    print(f"{set(lhs)} → {set(rhs)} (Support: {sup:.2f}, Confidence: {conf:.2f})")







# FP-Growth Algorithm for Association Rule Mining
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Load and preprocess the dataset
df = pd.read_csv("/content/Groceries_dataset.csv")
transactions = df.groupby('Member_number')['itemDescription'].apply(list).tolist()

# Encode transactions for FP-Growth
te = TransactionEncoder()
encoded_data = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(encoded_data, columns=te.columns_)

# Apply FP-Growth algorithm
frequent_itemsets = fpgrowth(df_encoded, min_support=0.02, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Print results
print("Frequent Itemsets:\n", frequent_itemsets)

print("\nAssociation Rules:")
for _, row in rules.iterrows():
    print(f"{set(row['antecedents'])} → {set(row['consequents'])} "
          f"(Support: {row['support']:.2f}, Confidence: {row['confidence']:.2f})")












# K-Means Clustering for Customer Segmentation
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load data
df = pd.read_csv('/content/Mall_Customers.csv')
X = df.iloc[:, [3, 4]].values  # Annual Income & Spending Score

# Elbow Method to find optimal clusters
wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    km.fit(X)
    wcss.append(km.inertia_)

# Plot Elbow Graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('No. of Clusters')
plt.ylabel('WCSS')
plt.show()

# Train KMeans with k=5 (from elbow)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)

# Plot clusters
colors = ['blue', 'green', 'red', 'cyan', 'magenta']
for i in range(5):
    plt.scatter(X[y_pred == i, 0], X[y_pred == i, 1], s=50, c=colors[i], label=f'Cluster {i+1}')

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=100, c='yellow', label='Centroids')

plt.title('Customer Segments (K-Means)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()













# DBSCAN Clustering for Customer Segmentation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import seaborn as sns

# Load and preprocess data
df = pd.read_csv('/content/Mall_Customers.csv')
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df.drop(columns=['CustomerID'], inplace=True)

# Normalize numerical features
scaler = MinMaxScaler()
df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']] = scaler.fit_transform(
    df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
)

# Apply DBSCAN
model = DBSCAN(eps=0.13, min_samples=10)
df['cluster'] = model.fit_predict(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

# Silhouette Score (excluding noise)
mask = df['cluster'] != -1
if mask.sum() > 1:
    score = silhouette_score(df.loc[mask, ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']], df['cluster'][mask])
    print(f"Silhouette Score: {score:.2f}")

# 2D plot (excluding noise)
sns.scatterplot(data=df[mask], x='Annual Income (k$)', y='Spending Score (1-100)', hue='cluster', palette='viridis')
plt.title("DBSCAN Clusters (Excl. Noise)")
plt.show()







#sql
create database olp;
use olp;



CREATE TABLE REGION (
    REGION_ID INT PRIMARY KEY,
    REGION_NAME VARCHAR(20)
);

CREATE TABLE SUBREGION (
    SUBREGION_ID INT PRIMARY KEY,
    SUBREGION_NAME VARCHAR(30),
    REGION_ID INT,
    FOREIGN KEY (REGION_ID) REFERENCES REGION(REGION_ID)
);

CREATE TABLE WAREHOUSE (
    CODE INT PRIMARY KEY,
    SUBREGION_ID INT,
    FOREIGN KEY (SUBREGION_ID) REFERENCES SUBREGION(SUBREGION_ID)
);

CREATE TABLE BOOKS (
    ISBN BIGINT PRIMARY KEY,
    TITLE VARCHAR(45),
    PRICE DECIMAL(10,2)
);

CREATE TABLE YEAR (
    YEAR_ID INT PRIMARY KEY,
    YEAR VARCHAR(4)
);

CREATE TABLE MONTH (
    MONTH_ID INT PRIMARY KEY,
    MONTH VARCHAR(15),
    YEAR_ID INT,
    FOREIGN KEY (YEAR_ID) REFERENCES YEAR(YEAR_ID)
);

CREATE TABLE DAY (
	DAY_ID int primary key,
    NUM_DAY INT ,
    day varchar(25),
    MONTH_ID INT,
    FOREIGN KEY (MONTH_ID) REFERENCES MONTH(MONTH_ID)
);



CREATE TABLE FACTS_TICKET (
    SELL_BY_PRODUCT DOUBLE,
    TICKET_NUMBER INT PRIMARY KEY,
    BOOKS_ISBN BIGINT,
    WAREHOUSE_CODE INT,
    DAY_ID INT,
    FOREIGN KEY (BOOKS_ISBN) REFERENCES BOOKS(ISBN),
    FOREIGN KEY (WAREHOUSE_CODE) REFERENCES WAREHOUSE(CODE),
    FOREIGN KEY (DAY_ID) REFERENCES DAY(DAY_ID)
);

































-- region
INSERT INTO region (region_id, region_name) VALUES (1, 'america');
INSERT INTO region (region_id, region_name) VALUES (2, 'europa');
INSERT INTO region (region_id, region_name) VALUES (3, 'asia');
INSERT INTO region (region_id, region_name) VALUES (4, 'africa');
INSERT INTO region (region_id, region_name) VALUES (5, 'oceania');

-- subregion
INSERT INTO subregion (subregion_id, region_id, subregion_name) VALUES (1, 1, 'north america');
INSERT INTO subregion (subregion_id, region_id, subregion_name) VALUES (2, 1, 'central america');
INSERT INTO subregion (subregion_id, region_id, subregion_name) VALUES (3, 1, 'caribe');
INSERT INTO subregion (subregion_id, region_id, subregion_name) VALUES (4, 1, 'south america');
INSERT INTO subregion (subregion_id, region_id, subregion_name) VALUES (5, 2, 'north of europe');
INSERT INTO subregion (subregion_id, region_id, subregion_name) VALUES (6, 2, 'south of europe');
INSERT INTO subregion (subregion_id, region_id, subregion_name) VALUES (7, 2, 'western europe');
INSERT INTO subregion (subregion_id, region_id, subregion_name) VALUES (8, 2, 'eastern europe');
INSERT INTO subregion (subregion_id, region_id, subregion_name) VALUES (9, 3, 'northern of asia');
INSERT INTO subregion (subregion_id, region_id, subregion_name) VALUES (10, 3, 'southern asia');

-- year
INSERT INTO year (year_id, year) VALUES (1, '2017');

-- month
INSERT INTO month (month_id, year_id, month) VALUES (1, 1, 'january');
INSERT INTO month (month_id, year_id, month) VALUES (2, 1, 'february');
INSERT INTO month (month_id, year_id, month) VALUES (3, 1, 'march');
INSERT INTO month (month_id, year_id, month) VALUES (4, 1, 'april');
INSERT INTO month (month_id, year_id, month) VALUES (5, 1, 'may');
INSERT INTO month (month_id, year_id, month) VALUES (6, 1, 'june');
INSERT INTO month (month_id, year_id, month) VALUES (7, 1, 'july');
INSERT INTO month (month_id, year_id, month) VALUES (8, 1, 'august');
INSERT INTO month (month_id, year_id, month) VALUES (9, 1, 'september');
INSERT INTO month (month_id, year_id, month) VALUES (10, 1, 'october');

-- day
INSERT INTO day (day_id, month_id, day, num_day) VALUES (1, 1, 'sunday', 1);
INSERT INTO day (day_id, month_id, day, num_day) VALUES (2, 1, 'monday', 2);
INSERT INTO day (day_id, month_id, day, num_day) VALUES (3, 1, 'tuesday', 3);
INSERT INTO day (day_id, month_id, day, num_day) VALUES (4, 1, 'wednesday', 4);
INSERT INTO day (day_id, month_id, day, num_day) VALUES (5, 1, 'thursday', 5);
INSERT INTO day (day_id, month_id, day, num_day) VALUES (6, 1, 'friday', 6);
INSERT INTO day (day_id, month_id, day, num_day) VALUES (7, 1, 'saturday', 7);
INSERT INTO day (day_id, month_id, day, num_day) VALUES (8, 1, 'sunday', 8);
INSERT INTO day (day_id, month_id, day, num_day) VALUES (9, 1, 'monday', 9);
INSERT INTO day (day_id, month_id, day, num_day) VALUES (10, 1, 'tuesday', 10);

-- warehouse
INSERT INTO warehouse (code, subregion_id) VALUES (1, 1);
INSERT INTO warehouse (code, subregion_id) VALUES (2, 1);
INSERT INTO warehouse (code, subregion_id) VALUES (3, 2);
INSERT INTO warehouse (code, subregion_id) VALUES (4, 2);
INSERT INTO warehouse (code, subregion_id) VALUES (5, 3);
INSERT INTO warehouse (code, subregion_id) VALUES (6, 3);
INSERT INTO warehouse (code, subregion_id) VALUES (7, 4);
INSERT INTO warehouse (code, subregion_id) VALUES (8, 4);
INSERT INTO warehouse (code, subregion_id) VALUES (9, 5);
INSERT INTO warehouse (code, subregion_id) VALUES (10, 5);

-- books
INSERT INTO books (isbn, title, price) VALUES (9706279301, 'A Gentleman Never Keeps Score', 384.00);
INSERT INTO books (isbn, title, price) VALUES (9728304802, 'Texas Glory', 94.50);
INSERT INTO books (isbn, title, price) VALUES (9702833303, 'Rainy day friends', 45.00);
INSERT INTO books (isbn, title, price) VALUES (9703859304, 'Crazy Rich Asians', 1234.50);
INSERT INTO books (isbn, title, price) VALUES (9793937405, 'Tell me you are mine', 23.60);
INSERT INTO books (isbn, title, price) VALUES (9729473006, 'The spy and the traitor', 200.00);
INSERT INTO books (isbn, title, price) VALUES (9703746207, 'Paradaise Sky by Jose Lansdale', 145.00);
INSERT INTO books (isbn, title, price) VALUES (9739482708, 'Meet Camaro from the Nigh Charter', 165.99);
INSERT INTO books (isbn, title, price) VALUES (9706279309, 'Book 9 for A Gentleman', 344.00);
INSERT INTO books (isbn, title, price) VALUES (9728304810, 'Book 10 for Texas', 84.50);


-- Ticket 1: Multi-book (2 books) in north america, Jan 1
INSERT INTO FACTS_TICKET (DAY_ID, WAREHOUSE_CODE, BOOKS_ISBN, SELL_BY_PRODUCT, TICKET_NUMBER) 
VALUES (1, 1, 9706279301, 384.00, 1); -- A Gentleman Never Keeps Score
INSERT INTO FACTS_TICKET (DAY_ID, WAREHOUSE_CODE, BOOKS_ISBN, SELL_BY_PRODUCT, TICKET_NUMBER) 
VALUES (1, 1, 9728304802, 94.50, 2);  -- Texas Glory

-- Ticket 2: Single book in central america, Jan 2
INSERT INTO FACTS_TICKET (DAY_ID, WAREHOUSE_CODE, BOOKS_ISBN, SELL_BY_PRODUCT, TICKET_NUMBER) 
VALUES (2, 3, 9702833303, 45.00, 3); -- Rainy day friends

-- Ticket 3: Multi-book (3 books) in caribe, Jan 3
INSERT INTO FACTS_TICKET (DAY_ID, WAREHOUSE_CODE, BOOKS_ISBN, SELL_BY_PRODUCT, TICKET_NUMBER) 
VALUES (3, 5, 9703859304, 1234.50, 4); -- Crazy Rich Asians
INSERT INTO FACTS_TICKET (DAY_ID, WAREHOUSE_CODE, BOOKS_ISBN, SELL_BY_PRODUCT, TICKET_NUMBER) 
VALUES (3, 5, 9793937405, 23.60, 5);   -- Tell me you are mine
INSERT INTO FACTS_TICKET (DAY_ID, WAREHOUSE_CODE, BOOKS_ISBN, SELL_BY_PRODUCT, TICKET_NUMBER) 
VALUES (3, 5, 9729473006, 200.00, 6);  -- The spy and the traitor

-- Ticket 4: Single book in south america, Jan 4
INSERT INTO FACTS_TICKET (DAY_ID, WAREHOUSE_CODE, BOOKS_ISBN, SELL_BY_PRODUCT, TICKET_NUMBER) 
VALUES (4, 7, 9703746207, 145.00, 7); -- Paradaise Sky by Jose Lansdale

-- Ticket 5: Multi-book (2 books) in north of europe, Jan 5
INSERT INTO FACTS_TICKET (DAY_ID, WAREHOUSE_CODE, BOOKS_ISBN, SELL_BY_PRODUCT, TICKET_NUMBER) 
VALUES (5, 9, 9739482708, 165.99, 8); -- Meet Camaro from the Nigh Charter
INSERT INTO FACTS_TICKET (DAY_ID, WAREHOUSE_CODE, BOOKS_ISBN, SELL_BY_PRODUCT, TICKET_NUMBER) 
VALUES (5, 9, 9706279309, 344.00, 9); -- Book 9 for A Gentleman

-- Ticket 6: Single book in western europe, Jan 6
INSERT INTO FACTS_TICKET (DAY_ID, WAREHOUSE_CODE, BOOKS_ISBN, SELL_BY_PRODUCT, TICKET_NUMBER) 
VALUES (6, 10, 9728304810, 84.50, 10); -- Book 10 for Texas










-- Query 1: Total Daily Sales by Book and Subregion

SELECT 
    sr.SUBREGION_NAME AS SUBREGION,
    d.NUM_DAY AS DAY,
    b.ISBN AS ISBN,
    b.TITLE AS TITLE,
    SUM(ft.SELL_BY_PRODUCT) AS TOTAL_SALES
FROM FACTS_TICKET ft
JOIN BOOKS b ON ft.BOOKS_ISBN = b.ISBN
JOIN WAREHOUSE w ON ft.WAREHOUSE_CODE = w.CODE
JOIN SUBREGION sr ON w.SUBREGION_ID = sr.SUBREGION_ID
JOIN DAY d ON ft.DAY_ID = d.DAY_ID
GROUP BY sr.SUBREGION_NAME, d.NUM_DAY, b.ISBN, b.TITLE
ORDER BY sr.SUBREGION_NAME, d.NUM_DAY, b.ISBN;


-- Query 2: Annual Sales per Book
SELECT 
    y.YEAR AS YEAR,
    b.ISBN AS ISBN,
    b.TITLE AS TITLE,
    SUM(ft.SELL_BY_PRODUCT) AS SALES_AMOUNT
FROM FACTS_TICKET ft
JOIN BOOKS b ON ft.BOOKS_ISBN = b.ISBN
JOIN DAY d ON ft.DAY_ID = d.DAY_ID
JOIN MONTH m ON d.MONTH_ID = m.MONTH_ID
JOIN YEAR y ON m.YEAR_ID = y.YEAR_ID
GROUP BY y.YEAR, b.ISBN, b.TITLE
ORDER BY y.YEAR, b.ISBN;


-- Query 3: Region with the Most Transactions in January 2017 (Adjusted from October)
SELECT 
    r.REGION_NAME AS REGIONNAME,
    COUNT(ft.TICKET_NUMBER) AS NO_TRANSACTIONS
FROM FACTS_TICKET ft
JOIN WAREHOUSE w ON ft.WAREHOUSE_CODE = w.CODE
JOIN SUBREGION sr ON w.SUBREGION_ID = sr.SUBREGION_ID
JOIN REGION r ON sr.REGION_ID = r.REGION_ID
JOIN DAY d ON ft.DAY_ID = d.DAY_ID
JOIN MONTH m ON d.MONTH_ID = m.MONTH_ID
JOIN YEAR y ON m.YEAR_ID = y.YEAR_ID
WHERE m.MONTH = 'january' AND y.YEAR = '2017'
GROUP BY r.REGION_NAME
ORDER BY NO_TRANSACTIONS DESC
LIMIT 1;


-- Query 4: Average Sale per Ticket per Month in Each Region

SELECT 
    r.REGION_NAME AS REGIONAME,
    m.MONTH AS MONTH,
    AVG(ft.SELL_BY_PRODUCT) AS AVG_INVOICE
FROM FACTS_TICKET ft
JOIN WAREHOUSE w ON ft.WAREHOUSE_CODE = w.CODE
JOIN SUBREGION sr ON w.SUBREGION_ID = sr.SUBREGION_ID
JOIN REGION r ON sr.REGION_ID = r.REGION_ID
JOIN DAY d ON ft.DAY_ID = d.DAY_ID
JOIN MONTH m ON d.MONTH_ID = m.MONTH_ID
GROUP BY r.REGION_NAME, m.MONTH
ORDER BY r.REGION_NAME, m.MONTH;




-- Query 5: Books Bought Together More Frequently

SELECT 
    b1.ISBN AS ISBN1,
    b2.ISBN AS ISBN2,
    COUNT(DISTINCT ft1.TICKET_NUMBER) AS CO_PURCHASE_COUNT
FROM FACTS_TICKET ft1
JOIN FACTS_TICKET ft2 ON ft1.TICKET_NUMBER = ft2.TICKET_NUMBER AND ft1.BOOKS_ISBN < ft2.BOOKS_ISBN
JOIN BOOKS b1 ON ft1.BOOKS_ISBN = b1.ISBN
JOIN BOOKS b2 ON ft2.BOOKS_ISBN = b2.ISBN
GROUP BY b1.ISBN, b2.ISBN
ORDER BY CO_PURCHASE_COUNT DESC
LIMIT 5;





-- Dice: Total sales for books priced between $100 and $500 in america and europa, January 2017

SELECT 
    r.REGION_NAME,
    m.MONTH,
    b.TITLE,
    SUM(ft.SELL_BY_PRODUCT) AS TOTAL_SALES
FROM FACTS_TICKET ft
JOIN BOOKS b ON ft.BOOKS_ISBN = b.ISBN
JOIN WAREHOUSE w ON ft.WAREHOUSE_CODE = w.CODE
JOIN SUBREGION sr ON w.SUBREGION_ID = sr.SUBREGION_ID
JOIN REGION r ON sr.REGION_ID = r.REGION_ID
JOIN DAY d ON ft.DAY_ID = d.DAY_ID
JOIN MONTH m ON d.MONTH_ID = m.MONTH_ID
JOIN YEAR y ON m.YEAR_ID = y.YEAR_ID
WHERE b.PRICE BETWEEN 100 AND 500
AND r.REGION_NAME IN ('america', 'europa')
AND m.MONTH = 'january'
AND y.YEAR = '2017'
GROUP BY r.REGION_NAME, m.MONTH, b.TITLE
ORDER BY r.REGION_NAME, m.MONTH, TOTAL_SALES DESC;



Rollup: Total sales by region with grand total
SELECT 
    COALESCE(r.REGION_NAME, 'All Regions') AS REGION_NAME,
    SUM(ft.SELL_BY_PRODUCT) AS TOTAL_SALES
FROM FACTS_TICKET ft
JOIN WAREHOUSE w ON ft.WAREHOUSE_CODE = w.CODE
JOIN SUBREGION sr ON w.SUBREGION_ID = sr.SUBREGION_ID
JOIN REGION r ON sr.REGION_ID = r.REGION_ID
GROUP BY ROLLUP(r.REGION_NAME)
ORDER BY TOTAL_SALES DESC;


Pivot: Total sales by region and month for 2017
SELECT 
    r.REGION_NAME,
    SUM(CASE WHEN m.MONTH = 'january' THEN ft.SELL_BY_PRODUCT ELSE 0 END) AS January
FROM FACTS_TICKET ft
JOIN WAREHOUSE w ON ft.WAREHOUSE_CODE = w.CODE
JOIN SUBREGION sr ON w.SUBREGION_ID = sr.SUBREGION_ID
JOIN REGION r ON sr.REGION_ID = r.REGION_ID
JOIN DAY d ON ft.DAY_ID = d.DAY_ID
JOIN MONTH m ON d.MONTH_ID = m.MONTH_ID
JOIN YEAR y ON m.YEAR_ID = y.YEAR_ID
WHERE y.YEAR = '2017'
GROUP BY r.REGION_NAME
ORDER BY r.REGION_NAME;



Slice: Total sales by book and subregion for January 2017
SELECT 
    sr.SUBREGION_NAME,
    b.TITLE,
    SUM(ft.SELL_BY_PRODUCT) AS TOTAL_SALES
FROM FACTS_TICKET ft
JOIN BOOKS b ON ft.BOOKS_ISBN = b.ISBN
JOIN WAREHOUSE w ON ft.WAREHOUSE_CODE = w.CODE
JOIN SUBREGION sr ON w.SUBREGION_ID = sr.SUBREGION_ID
JOIN DAY d ON ft.DAY_ID = d.DAY_ID
JOIN MONTH m ON d.MONTH_ID = m.MONTH_ID
JOIN YEAR y ON m.YEAR_ID = y.YEAR_ID
WHERE m.MONTH = 'january' AND y.YEAR = '2017'
GROUP BY sr.SUBREGION_NAME, b.TITLE
ORDER BY sr.SUBREGION_NAME, TOTAL_SALES DESC;

