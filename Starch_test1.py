import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
import pandas as pd
from sklearn.decomposition import PCA

# Load your data
df = pd.read_excel("f:/Project_Data/StarchData(2).xlsx")

# Create a DataFrame with only the mean columns
mean_columns = {
    'Mean_AC_1.time': df.iloc[:, 1:100].mean(axis=1),
    'Mean_AC_2.time': df.iloc[:, 101:199].mean(axis=1),
    'Mean_AC_3.time': df.iloc[:, 200:298].mean(axis=1),
    'Mean_AC_4.time': df.iloc[:, 299:397].mean(axis=1),
    'Mean_AC_5.time': df.iloc[:, 398:496].mean(axis=1),
    'Mean_AG_1.time': df.iloc[:, 497:595].mean(axis=1),
    'Mean_AG_2.time': df.iloc[:, 596:694].mean(axis=1),
    'Mean_AG_3.time': df.iloc[:, 695:793].mean(axis=1),
    'Mean_AG_4.time': df.iloc[:, 794:892].mean(axis=1),
    'Mean_AG_5.time': df.iloc[:, 893:991].mean(axis=1),
    'Mean_AH_1.time': df.iloc[:, 992:1090].mean(axis=1),
    'Mean_AH_2.time': df.iloc[:, 1091:1189].mean(axis=1),
    'Mean_AH_3.time': df.iloc[:, 1190:1288].mean(axis=1),
    'Mean_AH_4.time': df.iloc[:, 1289:1387].mean(axis=1),
    'Mean_AH_5.time': df.iloc[:, 1388:1486].mean(axis=1),
    'Mean_AM_1.time': df.iloc[:, 1487:1585].mean(axis=1),
    'Mean_AM_2.time': df.iloc[:, 1586:1684].mean(axis=1),
    'Mean_AM_3.time': df.iloc[:, 1685:1783].mean(axis=1),
    'Mean_AM_4.time': df.iloc[:, 1784:1882].mean(axis=1),
    'Mean_AM_5.time': df.iloc[:, 1883:1981].mean(axis=1),
    'Mean_KA_1.time': df.iloc[:, 1982:2080].mean(axis=1),
    'Mean_KA_2.time': df.iloc[:, 2081:2179].mean(axis=1),
    'Mean_KA_3.time': df.iloc[:, 2180:2278].mean(axis=1),
    'Mean_KA_4.time': df.iloc[:, 2279:2377].mean(axis=1),
    'Mean_KA_5.time': df.iloc[:, 2378:2476].mean(axis=1),
    'Mean_KP_1.time': df.iloc[:, 2477:2575].mean(axis=1),
    'Mean_KP_2.time': df.iloc[:, 2576:2674].mean(axis=1),
    'Mean_KP_3.time': df.iloc[:, 2675:2773].mean(axis=1),
    'Mean_KP_4.time': df.iloc[:, 2774:2872].mean(axis=1),
    'Mean_KP_5.time': df.iloc[:, 2873:2971].mean(axis=1),
    'Mean_MR_1.time': df.iloc[:, 2972:3070].mean(axis=1),
    'Mean_MR_2.time': df.iloc[:, 3071:3169].mean(axis=1),
    'Mean_MR_3.time': df.iloc[:, 3170:3268].mean(axis=1),
    'Mean_MR_4.time': df.iloc[:, 3269:3367].mean(axis=1),
    'Mean_MR_5.time': df.iloc[:, 3368:3466].mean(axis=1),
    'Mean_NK_1.time': df.iloc[:, 3467:3565].mean(axis=1),
    'Mean_NK_2.time': df.iloc[:, 3566:3664].mean(axis=1),
    'Mean_NK_3.time': df.iloc[:, 3665:3763].mean(axis=1),
    'Mean_NK_4.time': df.iloc[:, 3764:3862].mean(axis=1),
    'Mean_NK_5.time': df.iloc[:, 3863:3961].mean(axis=1),
    'Mean_OB_1.time': df.iloc[:, 3962:4060].mean(axis=1),
    'Mean_OB_2.time': df.iloc[:, 4061:4159].mean(axis=1),
    'Mean_OB_3.time': df.iloc[:, 4160:4258].mean(axis=1),
    'Mean_OB_4.time': df.iloc[:, 4258:4357].mean(axis=1),
    'Mean_OB_5.time': df.iloc[:, 4358:4456].mean(axis=1),
    'Mean_OF_1.time': df.iloc[:, 4457:4555].mean(axis=1),
    'Mean_OF_2.time': df.iloc[:, 4556:4654].mean(axis=1),
    'Mean_OF_3.time': df.iloc[:, 4655:4753].mean(axis=1),
    'Mean_OF_4.time': df.iloc[:, 4754:4852].mean(axis=1),
    'Mean_OF_5.time': df.iloc[:, 4853:4951].mean(axis=1),
    'Mean_CV_1.time': df.iloc[:, 4952:5050].mean(axis=1),
    'Mean_CV_2.time': df.iloc[:, 5051:5149].mean(axis=1),
    'Mean_CV_3.time': df.iloc[:, 5150:5248].mean(axis=1),
    'Mean_CV_4.time': df.iloc[:, 5249:5347].mean(axis=1),
    'Mean_CV_5.time': df.iloc[:, 5348:5446].mean(axis=1),
    'Mean_PR_1.time': df.iloc[:, 5447:5545].mean(axis=1),
    'Mean_PR_2.time': df.iloc[:, 5546:5644].mean(axis=1),
    'Mean_PR_3.time': df.iloc[:, 5645:5743].mean(axis=1),
    'Mean_PR_4.time': df.iloc[:, 5744:5842].mean(axis=1),
    'Mean_PR_5.time': df.iloc[:, 5843:5941].mean(axis=1),
    'Mean_SR_1.time': df.iloc[:, 5942:6040].mean(axis=1),
    'Mean_SR_2.time': df.iloc[:, 6041:6139].mean(axis=1),
    'Mean_SR_3.time': df.iloc[:, 6140:6238].mean(axis=1),
    'Mean_SR_4.time': df.iloc[:, 6239:6337].mean(axis=1),
    'Mean_SR_5.time': df.iloc[:, 6338:6436].mean(axis=1),
    'Mean_TD_1.time': df.iloc[:, 6437:6535].mean(axis=1),
    'Mean_TD_2.time': df.iloc[:, 6536:6634].mean(axis=1),
    'Mean_TD_3.time': df.iloc[:, 6635:6733].mean(axis=1),
    'Mean_TD_4.time': df.iloc[:, 6734:6832].mean(axis=1),
    'Mean_TD_5.time': df.iloc[:, 6833:6931].mean(axis=1),
    'Mean_WATER.time': df.iloc[:, 6932:7030].mean(axis=1),
    'Mean_WN_1.time': df.iloc[:, 7031:7129].mean(axis=1),
    'Mean_WN_2.time': df.iloc[:, 7130:7228].mean(axis=1),
    'Mean_WN_3.time': df.iloc[:, 7229:7327].mean(axis=1),
    'Mean_WN_4.time': df.iloc[:, 7328:7426].mean(axis=1),
    'Mean_WN_5.time': df.iloc[:, 7427:7525].mean(axis=1),
    'Mean_WP_1.time': df.iloc[:, 7526:7624].mean(axis=1),
    'Mean_WP_2.time': df.iloc[:, 7625:7723].mean(axis=1),
    'Mean_WP_3.time': df.iloc[:, 7724:7822].mean(axis=1),
    'Mean_WP_4.time': df.iloc[:, 7823:7921].mean(axis=1),
    'Mean_WP_5.time': df.iloc[:, 7922:8020].mean(axis=1),
    'Mean_YA003.time': df.iloc[:, 8021:8119].mean(axis=1),
    'Mean_YA007.time': df.iloc[:, 8120:8218].mean(axis=1),
    'Mean_YA011.time': df.iloc[:, 8219:8317].mean(axis=1),
    'Mean_YA015.time': df.iloc[:, 8318:8416].mean(axis=1),
    'Mean_YA019.time': df.iloc[:, 8417:8515].mean(axis=1),
}


data_frame = pd.DataFrame(mean_columns)
from sklearn.preprocessing import StandardScaler
# Normalize the data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data_frame)
data_normalized = pd.DataFrame(data_normalized, columns=data_frame.columns)

# Fill NaNs with column means
data_normalized = data_normalized.fillna(data_normalized.mean())

# Perform Mean Shift clustering
cluster_model = MeanShift(bandwidth=1.0)  # Adjust bandwidth if clustering is slow or not meaningful
data_frame['Cluster'] = cluster_model.fit_predict(data_normalized)
print("Unique clusters found:", data_frame['Cluster'].unique())

# Dimensionality reduction for visualization
pca = PCA(n_components=2)
data_reduced = pca.fit_transform(data_normalized)

# Plot the clusters
plt.figure(figsize=(10, 7))
plt.scatter(data_reduced[:, 0], data_reduced[:, 1], marker='o', c=data_frame['Cluster'], cmap='viridis')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Mean Shift Clustering Results')
plt.colorbar(label='Cluster')
plt.show()