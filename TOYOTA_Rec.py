import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 데이터셋 읽기
df = pd.read_csv('test_data.csv')
print(df[:5])

# categorical data one-hot encoding
df['평일/주말'] = df['평일/주말'].apply(lambda x: 0 if x == 'Weekday' else 1)

# # 0과 1의 비율 계산
# ratio = df['평일/주말'].value_counts(normalize=True) * 100
# print(df[['평일/주말']])
# print("\n0(평일)과 1(주말)의 비율:")
# print(ratio)
# exit()

# 데이터 분석
# '예약시간' 열을 datetime 형식으로 변환
df['예약시간'] = pd.to_datetime(df['예약시간'])
# print(df['예약시간'])


# '예약시간'에서 시간대(hour) 추출
df['예약시간대'] = df['예약시간'].dt.hour
print(df[:5])




# 고객별 특성 추출: 주중/주말 방문 비율, 시간대별 평균 방문 시간
df['weekday_visit'] = df['예약시간'].apply(lambda x: 1 if x.weekday() < 5 else 0)
customer_features = df.groupby('고객명').agg({'weekday_visit': 'mean', '예약시간대': 'mean'}).reset_index()

# 특성 스케일링
scaler = StandardScaler()
scaled_features = scaler.fit_transform(customer_features[['weekday_visit', '예약시간대']])

# K-Means 클러스터링
kmeans = KMeans(n_clusters=4, random_state=42)
customer_features['cluster'] = kmeans.fit_predict(scaled_features)

# PCA를 사용한 차원 축소
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_features)

# 클러스터링 결과 시각화
plt.figure(figsize=(10, 8))
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=customer_features['cluster'], cmap='viridis', marker='o', alpha=0.7)
plt.title('Customer Clustering based on Visit Characteristics')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()