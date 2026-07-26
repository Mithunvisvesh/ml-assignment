import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from numpy.linalg import matrix_rank
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# A1. Purchase Data Analysis
# ==========================================
df_purchase = pd.read_excel("Lab Session Data.xlsx", sheet_name="Purchase data")
print("--- Purchase Data Head ---")
print(df_purchase.head())

X = df_purchase[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
y = df_purchase[['Payment (Rs)']].values

print("\n[A1 Results]")
print("Dimensionality :", X.shape[1])
print("No. of vectors in vector space :", X.shape[0])

rank_A = matrix_rank(X)
print("Rank of matrix A :", rank_A)

# Pseudo-Inverse to find product cost
B = np.linalg.pinv(X)
cost = B @ y
print("Cost of Candy :", cost[0][0])
print("Cost of Mango :", cost[1][0])
print("Cost of Milk Packet :", cost[2][0])


# ==========================================
# A3. IRCTC Stock Price Analysis
# ==========================================
df_stock = pd.read_excel("Lab Session Data.xlsx", sheet_name="IRCTC Stock Price")
print("\n--- IRCTC Stock Data Head ---")
print(df_stock.head())

price = df_stock["Price"]
mean_price = np.mean(price)
variance_price = np.var(price)

def my_mean(data):
    total = 0
    for value in data:
        total += value
    return total / len(data)

def my_variance(data):
    mean = my_mean(data)
    total = 0
    for value in data:
        total += (value - mean) ** 2
    return total / len(data)

print("\nPopulation Mean (Numpy vs Custom):", np.mean(price), "|", my_mean(price))
print("Population Variance (Numpy vs Custom):", np.var(price), "|", my_variance(price))

# Timing comparison over 10 runs for Mean
mean_times = []
for i in range(10):
    start = time.perf_counter()
    np.mean(price)
    end = time.perf_counter()
    mean_times.append(end - start)
print("Average Time for Numpy Mean (10 runs):", sum(mean_times)/10)

custom_mean_times = []
for i in range(10):
    start = time.perf_counter()
    my_mean(price)
    end = time.perf_counter()
    custom_mean_times.append(end - start)
print("Average Time for Custom Mean (10 runs):", sum(custom_mean_times)/10)

# a) Wednesday sample mean vs population mean
wednesday = df_stock[df_stock["Day"] == "Wed"]
wednesday_mean = np.mean(wednesday["Price"])
print("\nPopulation Mean:", mean_price)
print("Wednesday Sample Mean:", wednesday_mean)

# b) April sample mean vs population mean
df_stock["Date"] = pd.to_datetime(df_stock["Date"])
april = df_stock[df_stock["Date"].dt.month == 4]
april_mean = np.mean(april["Price"])
print("April Sample Mean:", april_mean)

# c) Probability of making a loss
if df_stock["Chg%"].dtype == 'O':
    df_stock["Chg%"] = df_stock["Chg%"].str.replace('%', '').astype(float)
loss = list(filter(lambda x: x < 0, df_stock["Chg%"]))
prob_loss = len(loss) / len(df_stock)
print("Probability of making a loss:", prob_loss)

# d) Probability of profit on Wednesday
profit_wed = wednesday[wednesday["Chg%"] > 0]
prob_profit_wed = len(profit_wed) / len(df_stock)
print("Probability of profit on Wednesday:", prob_profit_wed)

# e) Conditional probability of profit given Wednesday
conditional_prob = len(profit_wed) / len(wednesday) if len(wednesday) > 0 else 0
print("Conditional Probability P(Profit | Wednesday):", conditional_prob)

# f) Scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(df_stock["Day"], df_stock["Chg%"])
plt.xlabel("Day")
plt.ylabel("Chg %")
plt.title("Chg% vs Day of Week")
plt.show()


# ==========================================
# A4, A8 & A9. Thyroid Data Exploration, Imputation & Normalization
# ==========================================
df_thyroid = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")
df_thyroid.replace('?', np.nan, inplace=True)

numeric_cols = df_thyroid.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_thyroid.select_dtypes(exclude=[np.number]).columns.tolist()

print("\n--- Thyroid Data Exploratory Statistics ---")
print("Missing Values Count:\n", df_thyroid.isnull().sum()[df_thyroid.isnull().sum() > 0])

print("\nNumeric Attributes Mean & Variance:")
for col in numeric_cols:
    print(f"{col} -> Mean: {df_thyroid[col].mean():.2f}, Var: {df_thyroid[col].var():.2f}")

# A8. Data Imputation
df_imputed = df_thyroid.copy()
for col in numeric_cols:
    Q1 = df_imputed[col].quantile(0.25)
    Q3 = df_imputed[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df_imputed[(df_imputed[col] < (Q1 - 1.5 * IQR)) | (df_imputed[col] > (Q3 + 1.5 * IQR))]
    if len(outliers) > 0:
        df_imputed[col].fillna(df_imputed[col].median(), inplace=True)
    else:
        df_imputed[col].fillna(df_imputed[col].mean(), inplace=True)

for col in categorical_cols:
    if df_imputed[col].isnull().sum() > 0:
        df_imputed[col].fillna(df_imputed[col].mode()[0], inplace=True)

print("\nMissing values after imputation:", df_imputed.isnull().sum().sum())

# A9. Normalization / Scaling
scaler = MinMaxScaler()
df_normalized = df_imputed.copy()
df_normalized[numeric_cols] = scaler.fit_transform(df_normalized[numeric_cols])
print("Data Normalization complete using MinMaxScaler.")


# ==========================================
# A5 & A6. Similarity Measures (Jaccard, SMC, Cosine)
# ==========================================
binary_cols = [col for col in df_thyroid.columns if df_thyroid[col].nunique() == 2]
bin_df = df_thyroid[binary_cols].copy()
for col in binary_cols:
    if bin_df[col].dtype == 'O':
        uniq = bin_df[col].dropna().unique()
        if len(uniq) == 2:
            bin_df[col] = bin_df[col].map({uniq[0]: 0, uniq[1]: 1})
bin_df = bin_df.fillna(0).values

vec1_bin = bin_df[0]
vec2_bin = bin_df[1]

f11 = np.sum((vec1_bin == 1) & (vec2_bin == 1))
f00 = np.sum((vec1_bin == 0) & (vec2_bin == 0))
f01 = np.sum((vec1_bin == 0) & (vec2_bin == 1))
f10 = np.sum((vec1_bin == 1) & (vec2_bin == 0))

jc = f11 / (f01 + f10 + f11) if (f01 + f10 + f11) != 0 else 0
smc = (f11 + f00) / (f00 + f01 + f10 + f11) if (f00 + f01 + f10 + f11) != 0 else 0

print("\n--- Similarity Measures (Observation 1 & 2) ---")
print("Jaccard Coefficient (JC):", jc)
print("Simple Matching Coefficient (SMC):", smc)

# A6. Cosine Similarity using numeric features of observation vectors 1 & 2
num_filled = df_thyroid[numeric_cols].fillna(0).values
vec1_num = num_filled[0]
vec2_num = num_filled[1]
cosine_sim = np.dot(vec1_num, vec2_num) / (np.linalg.norm(vec1_num) * np.linalg.norm(vec2_num))
print("Cosine Similarity:", cosine_sim)


# ==========================================
# A7. Heatmap Plot for First 20 Observations
# ==========================================
n_obs = 20
sub_bin = bin_df[:n_obs]
sub_num = num_filled[:n_obs]

jc_mat = np.zeros((n_obs, n_obs))
smc_mat = np.zeros((n_obs, n_obs))
cos_mat = np.zeros((n_obs, n_obs))

for i in range(n_obs):
    for j in range(n_obs):
        v1_b, v2_b = sub_bin[i], sub_bin[j]
        _f11 = np.sum((v1_b == 1) & (v2_b == 1))
        _f00 = np.sum((v1_b == 0) & (v2_b == 0))
        _f01 = np.sum((v1_b == 0) & (v2_b == 1))
        _f10 = np.sum((v1_b == 1) & (v2_b == 0))
        
        jc_mat[i, j] = _f11 / (_f01 + _f10 + _f11) if (_f01 + _f10 + _f11) != 0 else 0
        smc_mat[i, j] = (_f11 + _f00) / (_f00 + _f01 + _f10 + _f11) if (_f00 + _f01 + _f10 + _f11) != 0 else 0
        
        n1, n2 = sub_num[i], sub_num[j]
        norm_prod = np.linalg.norm(n1) * np.linalg.norm(n2)
        cos_mat[i, j] = np.dot(n1, n2) / norm_prod if norm_prod != 0 else 0

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.heatmap(jc_mat, annot=True, ax=axes[0], cmap="Blues").set_title("Jaccard Coefficient Heatmap")
sns.heatmap(smc_mat, annot=True, ax=axes[1], cmap="Greens").set_title("SMC Heatmap")
sns.heatmap(cos_mat, annot=True, ax=axes[2], cmap="Oranges").set_title("Cosine Similarity Heatmap")
plt.tight_layout()
plt.show()