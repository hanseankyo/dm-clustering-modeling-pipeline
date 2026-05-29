#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = BASE_DIR / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from config import DATA_DIR, OUTPUT_DIR, resolve_input

COHORT = Path(__file__).parent.name
SCRIPT_NAME = Path(__file__).stem
METHOD = SCRIPT_NAME.replace('_Kclustering', '').replace('_Hclustering', '')
ALGO = 'Kmeans' if 'Kclustering' in SCRIPT_NAME else 'Hclust'
SAVE_DIR = OUTPUT_DIR / 'Clustering' / COHORT / METHOD / ALGO
SAVE_DIR.mkdir(parents=True, exist_ok=True)

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_score,calinski_harabasz_score
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os

raw_data = pd.read_csv(resolve_input("SHAP_total_SNUH.csv"))

save_path = SAVE_DIR
save_path.mkdir(parents=True, exist_ok=True)

clustering_lst = ['bmi', 'FBS', 'Hba1c', 'MBP', 'Tg', 'ALT','exercise','drink','smoke']
drop_variables_lst = ['area']

np.random.seed(42)
clustering_data = raw_data[clustering_lst].copy()


pca = PCA(n_components=3, svd_solver='randomized', random_state=123)
clustering_data = pca.fit_transform(clustering_data)

silhouette_scores = []
calinski_scores = []
ssd_scores = []
cluster_range = range(3, 6)

for n_clusters in cluster_range:
    print(n_clusters,end='\r')
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(clustering_data)
    
    silhouette_avg = silhouette_score(clustering_data, cluster_labels)
    silhouette_scores.append(silhouette_avg)
        
    calinski_score = calinski_harabasz_score(clustering_data, cluster_labels)
    calinski_scores.append(calinski_score)
    
    ssd_scores.append(kmeans.inertia_)

    raw_data['cluster'] = cluster_labels
    raw_data.to_csv(save_path / f'K_cluster_{n_clusters}_raw_data.csv',index=False)
            

sihouette_pd = pd.DataFrame({
    'K': list(cluster_range),
    'silhouette_scores': silhouette_scores,
    'calinski_harabasz_scores': calinski_scores,
    'ssd_scores': ssd_scores
})
sihouette_pd.to_csv(save_path / f'K_silhouette_scores.csv',index=False)

# 최적의 클러스터 개수 찾기
best_cluster_num = cluster_range[silhouette_scores.index(max(silhouette_scores))]
print(f"최적의 클러스터 개수: {best_cluster_num}")
print(f"최적 실루엣 스코어: {round(max(silhouette_scores),4)}")

