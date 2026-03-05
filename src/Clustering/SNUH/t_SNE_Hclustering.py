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

from sklearn.manifold import TSNE

import os

raw_data = pd.read_csv(resolve_input("SHAP_total_SNUH.csv"))

save_path = SAVE_DIR
save_path.mkdir(parents=True, exist_ok=True)

clustering_lst = ['bmi', 'FBS', 'Hba1c', 'MBP', 'waist', 'Tg', 'ALT','exercise','drink','smoke']
drop_variables_lst = ['area']

np.random.seed(42)
clustering_data = raw_data[clustering_lst].copy()

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
clustering_data = tsne.fit_transform(clustering_data)

cluster_range = range(3, 6)

for linkage_str in ['ward','average','complete']:
    silhouette_scores = []
    calinski_scores = []
    ssd_scores = []
    print(linkage_str)
    for n_clusters in cluster_range:
        print(n_clusters, end='\r')
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_str) # ward
        cluster_labels = clustering.fit_predict(clustering_data)
        
        silhouette_avg = silhouette_score(clustering_data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        
        calinski_score = calinski_harabasz_score(clustering_data, cluster_labels)
        calinski_scores.append(calinski_score)
        
        
        # 각 클러스터 내의 거리 합 계산
        ssd = 0
        for label in set(cluster_labels):
            cluster_points = clustering_data[cluster_labels == label]
            if len(cluster_points) > 1:
                distances = pairwise_distances(cluster_points)
                ssd += distances.sum() / 2  # 거리 합의 절반(상삼각 행렬의 합)으로 계산
        ssd_scores.append(ssd)
        raw_data['cluster'] = cluster_labels
        raw_data.to_csv(save_path / f'{linkage_str}_cluster_{n_clusters}_raw_data.csv',index=False)
            

    sihouette_pd = pd.DataFrame({
        'K': list(cluster_range),
        'silhouette_scores': silhouette_scores,
        'calinski_harabasz_scores': calinski_scores,
        'ssd_scores': ssd_scores
    })
    sihouette_pd.to_csv(save_path / f'{linkage_str}_silhouette_scores.csv',index=False)

    # 최적의 클러스터 개수 찾기
    
    best_cluster_num = cluster_range[silhouette_scores.index(max(silhouette_scores))]
    print(f"최적의 클러스터 개수: {best_cluster_num}")
    print(f"최적 실루엣 스코어: {round(max(silhouette_scores),4)}")

