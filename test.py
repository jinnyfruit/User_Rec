import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import seaborn as sns

# 예제 데이터 로드 (가정)
df = pd.read_csv('test_data.csv')

# 카테고리 데이터 원-핫 인코딩 (주중: 0, 주말: 100)
df['평일/주말'] = df['평일/주말'].apply(lambda x: 0 if x == 'Weekday' else 1)

# '예약시간'을 datetime으로 변환 후 시간(hour) 추출
df['예약시간'] = pd.to_datetime(df['예약시간'])
df['예약시간'] = df['예약시간'].dt.hour

# 고객별로 주중/주말 방문 비율과 시간대별 평균 방문 시간 계산
customer_features = df.groupby('고객명').agg({
    '평일/주말': 'mean',  # 주중/주말 방문 비율
    '예약시간': 'mean'  # 시간대별 평균 방문 시간
}).reset_index()

# 오전/오후 변수 추가 (오전: 0, 오후: 1)
df['오전/오후'] = df['예약시간'].apply(lambda x: 0 if x < 12 else 1)

# 고객별 평균 계산
customer_features = df.groupby('고객명').agg({
    '평일/주말': 'mean',
    '오전/오후': 'mean'
}).reset_index()

# 클러스터링
kmeans = KMeans(n_clusters=4, random_state=42)  # k 값은 예시로 4를 사용
customer_features['cluster'] = kmeans.fit_predict(customer_features[['평일/주말', '오전/오후']])

# 시각화
plt.figure(figsize=(8, 6))
sns.scatterplot(x='평일/주말', y='오전/오후', hue='cluster', data=customer_features, palette='viridis', alpha=0.7)
plt.title('Customer Clustering based on Weekday/Weekend and Time of Day')
plt.xlabel('Weekday/Weekend')
plt.ylabel('Morning/Afternoon')
plt.yticks([0, 1], ['Morning', 'Afternoon'])
plt.grid(True)
plt.show()

exit()

# 클러스터링 진행
kmeans = KMeans(n_clusters=6, random_state=42)
customer_features['cluster'] = kmeans.fit_predict(customer_features[['평일/주말', '예약시간']])

# 클러스터링 결과 3차원 시각화
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
