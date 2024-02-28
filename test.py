import pandas as pd
from sklearn.utils import shuffle
from datetime import datetime
import numpy as np

# 데이터 로드 및 가공 예시 (실제 'data.csv' 파일이 필요함)
df = pd.read_csv('data.csv')

# 차량번호별로 그룹화하고 각 그룹의 크기가 2 이상인 것만 필터링
filtered_df = df.groupby('차량번호').filter(lambda x: len(x) > 1)

# 데이터 전처리
filtered_df['예약 시간'] = filtered_df['예약 시간'].str.replace('시', '').astype(int)
요일_매핑 = {'월': 0, '화': 1, '수': 2, '목': 3, '금': 4, '토': 5, '일': 6}
filtered_df['요일'] = filtered_df['요일'].map(요일_매핑)

# 무작위로 10개의 차량번호 선택
selected_vehicles = shuffle(filtered_df['차량번호'].unique())[:20]

# 최신 정비 이력 추출 및 나머지 이력의 평균 계산
avg_records = []
latest_records = []
for vehicle in selected_vehicles:
    vehicle_records = filtered_df[filtered_df['차량번호'] == vehicle]
    latest_record = vehicle_records.sort_values(by='예약일자', ascending=False).iloc[0]
    latest_records.append(latest_record)

    remaining_records = vehicle_records.drop(latest_record.name)
    if not remaining_records.empty:
        avg_time = remaining_records['예약 시간'].mean()
        avg_day = remaining_records['요일'].mean()
        avg_records.append({'차량번호': vehicle, '평균 예약 시간': avg_time, '평균 요일': avg_day})

# 최신 정비 이력 DataFrame 생성
latest_df = pd.DataFrame(latest_records)


# 거리 계산 및 가장 가까운 두 개의 이력 추출
def calculate_distance(row1, row2):
    return np.sqrt((row1['예약 시간'] - row2['평균 예약 시간']) ** 2 + (row1['요일'] - row2['평균 요일']) ** 2)


correct_matches = 0
for index, latest_record in latest_df.iterrows():
    distances = []
    for avg_record in avg_records:
        distance = calculate_distance(latest_record, avg_record)
        distances.append((avg_record['차량번호'], distance))
    closest_two = sorted(distances, key=lambda x: x[1])[:2]

    # 정답 확인
    if latest_record['차량번호'] in [ct[0] for ct in closest_two]:
        correct_matches += 1

# 정답률 계산
accuracy = correct_matches / len(latest_df)
print(f"정답률: {accuracy:.2f}")
