import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import seaborn as sns

# Load dataset
df = pd.read_csv('result/data.csv', low_memory=False)
df['예약 휴대폰번호'] = df['예약 휴대폰번호'].str.replace(' ', '')
print(df[:5])

# Categorical Date encoding - You may choose one of the options
# option1: Integer Encoding
df['요일'] = df['요일'].apply(lambda x:
                              0 if x == '월' else
                              1 if x == '화' else
                              2 if x == '수' else
                              3 if x == '목' else
                              4 if x == '금' else
                              5 if x == '토' else
                              6)

# option2: Binary Encoding
# df['요일'] = df['요일'].apply(lambda x:
#                               0 if x == '월' else
#                               0 if x == '화' else
#                               0 if x == '수' else
#                               0 if x == '목' else
#                               0 if x == '금' else
#                               1 if x == '토' else
#                               1)


# 고객별로 주중/주말 방문 비율과 시간대별 평균 방문 시간 계산
# Option1: 24시 기준 mapping
df['예약 시간'] = df['예약 시간'].str.replace('시', '').astype(int)
customer_features = df.groupby('차량번호').agg({
    '요일': 'mean',
    '예약 시간': 'mean'
}).reset_index()

# Option2: 오전/오후 binary mapping
# df['오전/오후'] = df['예약 시간'].apply(lambda x: 0 if x < 12 else 1)
# customer_features = df.groupby('차량번호').agg({
#     '요일': 'mean',
#     '오전/오후': 'mean'
# }).reset_index()

# K-means 클러스터링
k = 9
kmeans = KMeans(n_clusters=k, random_state=42)
customer_features['cluster'] = kmeans.fit_predict(customer_features[['요일', '오전/오후']])

# 시각화 - 선택한 option에 따라서 Y값 변경 가능
plt.figure(figsize=(8, 6))
sns.scatterplot(x='요일', y='오전/오후', hue='cluster', data=customer_features, palette='viridis', alpha=0.7)
plt.title('Customer Clustering based on Weekday/Weekend and Time of Day')
plt.xlabel('Weekday/Weekend')
plt.ylabel('Morning/Afternoon')
plt.yticks([0, 1], ['Morning', 'Afternoon'])
plt.grid(True)
plt.show()

exit()
#--------3차원 시각화---------#
# 클러스터링 진행
kmeans = KMeans(n_clusters=6, random_state=42)
customer_features['cluster'] = kmeans.fit_predict(customer_features[['평일/주말', '예약시간']])

# 클러스터링 결과 3차원 시각화 (요일, 시간, 해당 클러스터)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 데이터 포인트 플로팅
scatter = ax.scatter(customer_features['평일/주말'], customer_features['예약시간'], customer_features['cluster'],
                     c=customer_features['cluster'], cmap='viridis', depthshade=False)

# 클러스터 센터 플로팅
centers = kmeans.cluster_centers_
ax.scatter(centers[:, 0], centers[:, 1], np.arange(len(centers)), c='red', s=200, alpha=0.5, marker='x')

# 라벨링
ax.set_title('3D Visualization of Customer Clustering')
ax.set_xlabel('Weekday/Weekend Ratio')
ax.set_ylabel('Average Visit Time of Day')
ax.set_zlabel('Cluster')

# 범례 추가
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)

plt.show()