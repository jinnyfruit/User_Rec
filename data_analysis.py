import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting을 위한 모듈
from sklearn.metrics import silhouette_score

df = pd.read_csv('test_data.csv')
# '예약시간'에서 시간대(hour) 추출

# categorical data one-hot encoding
df['평일/주말'] = df['평일/주말'].apply(lambda x: 0 if x == 'Weekday' else 1)

df['예약시간'] = pd.to_datetime(df['예약시간'])
# print(df['예약시간'])


# '예약시간'에서 시간대(hour) 추출
df['예약시간'] = df['예약시간'].dt.hour
print(df[:5])

# 고객별로 주중/주말 방문 비율과 시간대별 평균 방문 시간 계산 (주중 1, 주말 0)
# 주중/주말 칼럼이 이미 0과 1로 인코딩되었다고 가정
customer_features = df.groupby('고객명').agg({
    '평일/주말': 'mean',  # 주중/주말 방문 비율
    '예약시간': 'mean'  # 시간대별 평균 방문 시간
}).reset_index()

# 데이터 스케일링 필요 없음 (이미 0~1과 0~24 범위 내에 있음)

# Elbow 방법을 사용하여 최적의 K값 찾기
# 주중/주말 방문 비율과 시간대별 평균 방문 시간을 기반으로 한 데이터셋 준비
features = customer_features[['평일/주말', '예약시간']]

# Elbow 방법에 필요한 SSE(Sum of Squared Errors) 값 저장을 위한 리스트
sse = []
# Silhouette 점수 저장을 위한 리스트
silhouette_scores = []

# K값의 범위 설정 (1에서 10까지 시도)
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features)
    sse.append(kmeans.inertia_)  # SSE 값 추가

    # 클러스터링이 유효한 경우에만 Silhouette 점수 계산 (클러스터가 1개인 경우 제외)
    if k > 1:
        silhouette_score_val = silhouette_score(features, kmeans.labels_)
        silhouette_scores.append(silhouette_score_val)

# Elbow 그래프 시각화
plt.figure(figsize=(10, 5))
plt.plot(k_range, sse, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# Silhouette 점수 그래프 시각화
plt.figure(figsize=(10, 5))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Score')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.xticks(range(2, 11))
plt.grid(True)
plt.show()

# 최적의 K값을 제안
optimal_k_silhouette = range(2, 11)[silhouette_scores.index(max(silhouette_scores))]
print(f"Silhouette 점수가 최대인 최적의 K값: {optimal_k_silhouette}")

# K-Means 클러스터링 - 최적 K 값으로 클러스터링
kmeans = KMeans(n_clusters=optimal_k_silhouette, random_state=42)
customer_features['cluster'] = kmeans.fit_predict(customer_features[['평일/주말', '예약시간']])

# 클러스터링 결과 시각화
plt.figure(figsize=(10, 8))
plt.scatter(customer_features['평일/주말'], customer_features['예약시간'],
            c=customer_features['cluster'], cmap='viridis', marker='o', alpha=0.7)
plt.title('Customer Clustering based on Weekday/Weekend and Time of Day')
plt.xlabel('Weekday/Weekend Ratio')
plt.ylabel('Average Visit Time of Day')
plt.colorbar(label='Cluster')
plt.xlim(0, 1)
plt.ylim(0, 24)
plt.grid(True)
plt.show()
