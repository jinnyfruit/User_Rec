import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import seaborn as sns

# Load dataset
df = pd.read_csv('data.csv',low_memory=False)
# df['예약 휴대폰번호'] = df['예약 휴대폰번호'].str.replace(' ', '')
# #print(df[:5])
#
# # 전화번호별 차량 수 계산
# vehicles_per_phone = df.groupby('예약 휴대폰번호')['차량번호'].nunique()
#
# # 차량이 1 초과인 전화번호와 한 대인 전화번호의 비율 계산
# more_than_one_vehicle = vehicles_per_phone[vehicles_per_phone > 1].count()
# one_vehicle = vehicles_per_phone[vehicles_per_phone == 1].count()
# print(more_than_one_vehicle)
# print(one_vehicle)
#
# # 차량이 1 초과인 전화번호를 csv 파일로 저장
# more_than_one_vehicle_phones = vehicles_per_phone[vehicles_per_phone > 1]
# more_than_one_vehicle_phones.to_csv('phones_with_more_than_one_vehicle.csv', header=True)
#
# # 그래프로 시각화
# plt.figure(figsize=(10, 6))
# vehicles_per_phone.plot(kind='bar')
# plt.title('전화번호별 차량 수')
# plt.xlabel('예약 휴대폰번호')
# plt.ylabel('차량 수')
# plt.xticks(rotation=45)
# plt.show()
# exit()

# 카테고리 데이터 원-핫 인코딩 (주중: 0, 주말: 1)
df['요일'] = df['요일'].apply(lambda x:
                              0 if x == '월' else
                              1 if x == '화' else
                              2 if x == '수' else
                              3 if x == '목' else
                              4 if x == '금' else
                              5 if x == '토' else
                              6)

df['예약 시간'] = df['예약 시간'].str.replace('시', '').astype(int)

# 고객별로 주중/주말 방문 비율과 시간대별 평균 방문 시간 계산
customer_features = df.groupby('차량번호').agg({
    '요일': 'mean',  # 주중/주말 방문 비율
    '예약 시간': 'mean'  # 시간대별 평균 방문 시간
}).reset_index()

# 오전/오후 변수 추가 (오전: 0, 오후: 1)
#df['오전/오후'] = df['예약 시간'].apply(lambda x: 0 if x < 12 else 1)

# 고객별 평균 계산
customer_features = df.groupby('차량번호').agg({
    '요일': 'mean',
    '예약 시간': 'mean'
}).reset_index()

# 클러스터링
kmeans = KMeans(n_clusters=8, random_state=42)  # k 값은 우선 4로 설정
customer_features['cluster'] = kmeans.fit_predict(customer_features[['요일', '예약 시간']])

# 시각화
plt.figure(figsize=(8, 6))
sns.scatterplot(x='요일', y='예약 시간', hue='cluster', data=customer_features, palette='viridis', alpha=0.7)
plt.title('Customer Clustering based on Weekday/Weekend and Time of Day')
plt.xlabel('Weekday/Weekend')
plt.ylabel('Morning/Afternoon')
plt.yticks([0, 24], ['0시', '24시'])
plt.grid(True)
plt.show()

exit()

#-------------------------------------------------------------------------------------------------------------#
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
