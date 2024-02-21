import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터셋 읽기
df = pd.read_csv('test_data.csv')
print(df[:5])

# 데이터 분석
# '예약시간' 열을 datetime 형식으로 변환
df['예약시간'] = pd.to_datetime(df['예약시간'])

# '예약시간'에서 시간대(hour) 추출
df['예약시간대'] = df['예약시간'].dt.hour
print(df['예약시간대'])

# 시각화
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))  # 2x2 그리드로 변경

# 평일 대비 주말 정비 횟수
weekend_weekday = df['평일/주말'].value_counts()
weekend_weekday.plot(kind='pie', ax=axes[0, 0], autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff'])
axes[0, 0].set_title('Weekday vs. Weekend Visits')
axes[0, 0].set_ylabel('')

# 시간대별 방문 분포
hour_distribution = df['예약시간대'].value_counts().sort_index()
sns.lineplot(x=hour_distribution.index, y=hour_distribution.values, ax=axes[0, 1], marker='o', color='green')
axes[0, 1].set_title('Time of Day Visit Distribution')
axes[0, 1].set_xlabel('Hour of Day')
axes[0, 1].set_ylabel('Visit Frequency')

# 누적 정비 기록 수 분포
customer_visits_total = df['고객명'].value_counts()
sns.histplot(customer_visits_total, bins=30, kde=False, color='skyblue', ax=axes[1, 0])
axes[1, 0].set_title('Accumulated Customer Visits Distribution')
axes[1, 0].set_xlabel('Number of Visits')
axes[1, 0].set_ylabel('Number of Customers')

axes[1, 1].axis('off')

plt.tight_layout()
plt.show()
