
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.spatial.distance import minkowski as scipy_minkowski

# ============================
# Load Dataset
# ============================
file_path = "Lab Session Data.xlsx"
sheet_name = "marketing_campaign"

df = pd.read_excel(file_path, sheet_name=sheet_name)

# ============================================================
# A1. Identify Datatypes
# ============================================================
print("\n========== A1 ==========")
feature_types = {
    "ID":"Nominal",
    "Year_Birth":"Interval",
    "Education":"Ordinal",
    "Marital_Status":"Nominal",
    "Income":"Ratio",
    "Kidhome":"Ratio",
    "Teenhome":"Ratio",
    "Dt_Customer":"Interval",
    "Recency":"Ratio",
    "MntWines":"Ratio",
    "MntFruits":"Ratio",
    "MntMeatProducts":"Ratio",
    "MntFishProducts":"Ratio",
    "MntSweetProducts":"Ratio",
    "MntGoldProds":"Ratio",
    "NumDealsPurchases":"Ratio",
    "NumWebPurchases":"Ratio",
    "NumCatalogPurchases":"Ratio",
    "NumStorePurchases":"Ratio",
    "NumWebVisitsMonth":"Ratio",
    "AcceptedCmp3":"Nominal",
    "AcceptedCmp4":"Nominal",
    "AcceptedCmp5":"Nominal",
    "AcceptedCmp1":"Nominal",
    "AcceptedCmp2":"Nominal",
    "Complain":"Nominal",
    "Z_CostContact":"Ratio",
    "Z_Revenue":"Ratio",
    "Response":"Nominal"
}
for k,v in feature_types.items():
    print(f"{k:25} {v}")

# ============================================================
# A2. Encoding Functions
# ============================================================
print("\n========== A2 ==========")

def label_encode(series):
    vals=sorted(series.dropna().unique())
    mapping={v:i for i,v in enumerate(vals)}
    return series.map(mapping),mapping

def one_hot_encode(df,columns):
    out=df.copy()
    for col in columns:
        vals=sorted(out[col].dropna().unique())
        for v in vals:
            out[f"{col}_{v}"]=(out[col]==v).astype(int)
        out.drop(columns=col,inplace=True)
    return out

# ============================================================
# A3. Apply Encoding
# ============================================================
print("\n========== A3 ==========")

categorical=["Education","Marital_Status"]

for c in categorical:
    enc,m=label_encode(df[c])
    print(c,m)

encoded_df=one_hot_encode(df,categorical)

print("Original Shape :",df.shape)
print("Encoded Shape  :",encoded_df.shape)

# ============================================================
# Prepare Numeric Matrix
# ============================================================
numeric_df=encoded_df.select_dtypes(include=np.number).copy()

for c in ["ID"]:
    if c in numeric_df.columns:
        numeric_df.drop(columns=c,inplace=True)

numeric_df=numeric_df.fillna(numeric_df.mean(numeric_only=True))
X=numeric_df.values.astype(float)

# ============================================================
# A4. Minkowski Distance
# ============================================================
print("\n========== A4 ==========")

def minkowski_distance(a,b,p=2):
    return np.sum(np.abs(a-b)**p)**(1/p)

# ============================================================
# A5. Distance Plot
# ============================================================
print("\n========== A5 ==========")

v1=X[0]
v2=X[1]

plist=list(range(1,11))
dist=[]

for p in plist:
    d=minkowski_distance(v1,v2,p)
    dist.append(d)
    print(f"p={p} Distance={d:.4f}")

plt.figure(figsize=(6,4))
plt.plot(plist,dist,marker="o")
plt.grid(True)
plt.xlabel("p")
plt.ylabel("Distance")
plt.title("Minkowski Distance")
plt.savefig("A5_Minkowski.png")
plt.close()

# ============================================================
# A6. Compare with scipy
# ============================================================
print("\n========== A6 ==========")

for p in plist:
    d1=minkowski_distance(v1,v2,p)
    d2=scipy_minkowski(v1,v2,p)
    print(f"p={p} Custom={d1:.6f}  Scipy={d2:.6f}")

# ============================================================
# A7. Dot Product & Norm
# ============================================================
print("\n========== A7 ==========")

def dot_product(a,b):
    return np.sum(a*b)

def euclidean_norm(a):
    return np.sqrt(np.sum(a*a))

print("Custom Dot :",dot_product(v1,v2))
print("NumPy Dot  :",np.dot(v1,v2))

print("Custom Norm:",euclidean_norm(v1))
print("NumPy Norm :",np.linalg.norm(v1))

# ============================================================
# A8. Mean Variance Std
# ============================================================
print("\n========== A8 ==========")

def my_mean(x):
    return np.sum(x)/len(x)

def my_variance(x):
    m=my_mean(x)
    return np.sum((x-m)**2)/len(x)

def my_std(x):
    return np.sqrt(my_variance(x))

def matrix_statistics(matrix):
    means=[]
    variances=[]
    stds=[]
    for i in range(matrix.shape[1]):
        col=matrix[:,i]
        means.append(my_mean(col))
        variances.append(my_variance(col))
        stds.append(my_std(col))
    return np.array(means),np.array(variances),np.array(stds)

mean_custom,var_custom,std_custom=matrix_statistics(X)

# ============================================================
# A9. Compare with NumPy
# ============================================================
print("\n========== A9 ==========")

mean_np=np.mean(X,axis=0)
std_np=np.std(X,axis=0)

print("First Five Mean Values")
print(mean_custom[:5])
print(mean_np[:5])

print("First Five Std Values")
print(std_custom[:5])
print(std_np[:5])

# ============================================================
# A10. Histogram
# ============================================================
print("\n========== A10 ==========")

feature=X[:,0]

print("Mean :",my_mean(feature))
print("Variance :",my_variance(feature))

plt.figure(figsize=(6,4))
plt.hist(feature,bins=10,edgecolor="black")
plt.xlabel(numeric_df.columns[0])
plt.ylabel("Frequency")
plt.title("Histogram")
plt.savefig("A10_Histogram.png")
plt.close()

# ============================================================
# A11. K-Means
# ============================================================
print("\n========== A11 ==========")

def initialize_centroids(X,k):
    idx=random.sample(range(len(X)),k)
    return X[idx]

def assign_clusters(X,centroids):
    labels=[]
    for row in X:
        d=[minkowski_distance(row,c,2) for c in centroids]
        labels.append(np.argmin(d))
    return np.array(labels)

def recompute_centroids(X,labels,k):
    centroids=[]
    for i in range(k):
        pts=X[labels==i]
        if len(pts)==0:
            centroids.append(X[random.randint(0,len(X)-1)])
        else:
            centroid=[]
            for j in range(X.shape[1]):
                centroid.append(my_mean(pts[:,j]))
            centroids.append(centroid)
    return np.array(centroids)

def kmeans(X,k,max_iter=100):
    centroids=initialize_centroids(X,k)
    for _ in range(max_iter):
        labels=assign_clusters(X,centroids)
        new_centroids=recompute_centroids(X,labels,k)
        if np.allclose(centroids,new_centroids):
            break
        centroids=new_centroids
    return centroids,labels

centroids,labels=kmeans(X,3)

numeric_df["Cluster"]=labels
numeric_df.to_csv("marketing_campaign_clusters.csv",index=False)

print("Cluster Counts")
print(pd.Series(labels).value_counts().sort_index())
print("Completed Successfully.")
