import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Show current folder and files
print("Current Folder:", os.getcwd())
print("Files in Folder:", os.listdir())

# Load data
df = pd.read_excel("Dataset for Data Analytics (3).xlsx")

print(df.head())
print(df.info())

# Data cleaning
df = df.drop_duplicates()
df = df.dropna()

print(df.isnull().sum())

# Select numerical columns
data = df.select_dtypes(include=['int64', 'float64'])

# Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Elbow Method
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(pca_data)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Clusters")
plt.ylabel("WCSS")
plt.savefig("outputs/elbow.png")
plt.show()

# KMeans Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(pca_data)

df["Cluster"] = clusters

# Silhouette Score
score = silhouette_score(pca_data, clusters)
print("Silhouette Score:", score)

# PCA Visualization
plt.figure(figsize=(8, 5))

sns.scatterplot(
    x=pca_data[:, 0],
    y=pca_data[:, 1],
    hue=clusters,
    palette="viridis"
)

plt.title("Customer Segmentation")
plt.savefig("outputs/customer_clusters.png")
plt.show()

# Save Result
df.to_excel("outputs/customer_segmented_data.xlsx", index=False)

print("Project Completed Successfully!")