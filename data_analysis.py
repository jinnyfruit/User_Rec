import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting을 위한 모듈
from sklearn.metrics import silhouette_score

# Read dataset
df = pd.read_csv('data.csv')
print(df)
df['요일'] = df['요일'].apply(lambda x: 0 if x in ['월', '화', '수', '목', '금'] else 1 if x in ['토','일'] else x)
#print(df['요일'])

# Data Preprocessing
# Categorical data one-hot encoding (범주형 데이터 인코딩)
# df['평일/주말'] = df['평일/주말'].apply(lambda x: 0 if x == 'Weekday' else 1)
# df['예약 시간'] = pd.to_datetime(df['예약 시간']) # 시간형 데이터로 변환
# df['예약 시간'] = df['예약 시간'].dt.hour # 예약 시간대 추출
df['예약 시간'] = df['예약 시간'].str.replace('시', '').astype(int)
print(df['예약 시간'])

# # Data visualization
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))  # 2x2 그리드로 변경
#
# # 전체 데이터 기준 평일/주말 정비횟수 비율
# weekend_weekday = df['요일'].value_counts()
# weekend_weekday.plot(kind='pie', ax=axes[0, 0], autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff'])
# axes[0, 0].set_title('Weekday vs. Weekend Visits')
# axes[0, 0].set_ylabel('')
#
# # 전체 데이터 기준 정비센터 시간대별 방문 분포
# hour_distribution = df['예약 시간'].value_counts().sort_index()
# sns.lineplot(x=hour_distribution.index, y=hour_distribution.values, ax=axes[0, 1], marker='o', color='green')
# axes[0, 1].set_title('Time of Day Visit Distribution')
# axes[0, 1].set_xlabel('Hour of Day')
# axes[0, 1].set_ylabel('Visit Frequency')
#
# # 고객별 누적 정비 기록 수 분포
# customer_visits_total = df['차량번호'].value_counts()
# sns.histplot(customer_visits_total, bins=30, kde=False, color='skyblue', ax=axes[1, 0])
# axes[1, 0].set_title('Accumulated Customer Visits Distribution')
# axes[1, 0].set_xlabel('Number of Visits')
# axes[1, 0].set_ylabel('Number of Customers')
#
# axes[1, 1].axis('off')
#
# plt.tight_layout()
# plt.show()

# 고객별로 주중/주말 방문 비율과 시간대별 평균 방문 시간 계산 (주중 1, 주말 0)
customer_features = df.groupby('예약 휴대폰번호').agg({
    '요일': 'mean',  # 주중/주말 방문 비율
    '예약 시간': 'mean'  # 시간대별 평균 방문 시간
}).reset_index()
#
# # Elbow 방법을 사용하여 최적의 K값 찾기
# features = customer_features[['요일', '예약 시간']]
#
# # Elbow 방법에 필요한 SSE(Sum of Squared Errors) 값 저장을 위한 리스트
# sse = []
# # Silhouette 점수 저장을 위한 리스트
# silhouette_scores = []

# # K값의 범위 설정 (2에서 10까지 시도)
# k_range = range(2, 11)
#
# for k in k_range:
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(features)
#     sse.append(kmeans.inertia_)  # SSE 값 추가
#
#     # 클러스터링이 유효한 경우에만 Silhouette 점수 계산 (클러스터가 1개인 경우 제외)
#     if k > 1:
#         silhouette_score_val = silhouette_score(features, kmeans.labels_)
#         silhouette_scores.append(silhouette_score_val)
#
# # Elbow 그래프 시각화
# plt.figure(figsize=(10, 5))
# plt.plot(k_range, sse, marker='o')
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('SSE')
# plt.xticks(k_range)
# plt.grid(True)
# plt.show()
#
# # Silhouette 점수 그래프 시각화
# plt.figure(figsize=(10, 5))
# plt.plot(range(2, 11), silhouette_scores, marker='o')
# plt.title('Silhouette Score')
# plt.xlabel('Number of clusters')
# plt.ylabel('Silhouette Score')
# plt.xticks(range(2, 11))
# plt.grid(True)
# plt.show()
#
# # 최적의 K값을 제안
# optimal_k_silhouette = range(2, 11)[silhouette_scores.index(max(silhouette_scores))]
# print(f"Silhouette 점수가  최대인 최적의 K값: {optimal_k_silhouette}")

# K-Means 클러스터링 - 최적 K 값으로 클러스터링 진행
kmeans = KMeans(n_clusters=5, random_state=42)
customer_features['cluster'] = kmeans.fit_predict(customer_features[['요일', '예약 시간']])

# 클러스터링 결과 시각화
plt.figure(figsize=(10, 8))
plt.scatter(customer_features['요일'], customer_features['예약 시간'],
            c=customer_features['cluster'], cmap='viridis', marker='o', alpha=0.7)
plt.title('Customer Clustering based on Weekday/Weekend and Time of Day')
plt.xlabel('Weekday/Weekend Ratio')
plt.ylabel('Average Visit Time of Day')
plt.colorbar(label='Cluster')
plt.xlim(0, 1)
plt.ylim(0, 24)
plt.grid(True)
plt.show()
