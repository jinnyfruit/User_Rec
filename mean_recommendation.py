import pandas as pd
import numpy as np
from scipy import stats
from sklearn.utils import shuffle

# 데이터 로드 및 가공
df = pd.read_csv('result/data.csv', low_memory=False)

# 차량번호별로 그룹화하고 정비이력 기록이 6 이상인 차량만 필터링한 뒤, 100개만 무작위 선택
filtered_df = df.groupby('차량번호').filter(lambda x: len(x) >= 6)
selected_vehicles = shuffle(filtered_df['차량번호'].unique(), random_state=0)[:100]
filtered_df = filtered_df[filtered_df['차량번호'].isin(selected_vehicles)]

# 데이터 전처리
filtered_df['예약 시간'] = filtered_df['예약 시간'].str.replace('시', '').astype(int)
요일_매핑 = {'월': 0, '화': 1, '수': 2, '목': 3, '금': 4, '토': 5, '일': 6}
filtered_df['요일'] = filtered_df['요일'].map(요일_매핑)

# 결과 저장을 위한 리스트
avg_results = []
service_center_results = []

# 선택된 차량번호에 대해 반복
for vehicle in selected_vehicles:
    vehicle_records = filtered_df[filtered_df['차량번호'] == vehicle]
    latest_record = vehicle_records.sort_values(by='예약일자', ascending=False).iloc[0]
    remaining_records = vehicle_records.drop(latest_record.name)

    # 평균 및 최빈값 계산
    avg_time = remaining_records['예약 시간'].mean()
    avg_day = remaining_records['요일'].mean()

    # 서비스 센터명 추천 (가장 빈번한 서비스 센터)
    most_frequent_service_center = remaining_records['서비스센터명'].mode().iloc[0]

    # 정답 후보군
    answer_candidates = filtered_df.groupby('차량번호').apply(
        lambda x: x.sort_values(by='예약일자', ascending=False).iloc[0]).reset_index(drop=True)

    # 평균 기반 거리 계산 및 가장 가까운 두 기록 찾기
    answer_candidates['평균_거리'] = answer_candidates.apply(
        lambda x: np.sqrt((avg_time - x['예약 시간']) ** 2 + (avg_day - x['요일']) ** 2), axis=1)
    closest_avg_records = answer_candidates.nsmallest(2, '평균_거리')
    avg_correct_answer = vehicle in closest_avg_records['차량번호'].values

    # 결과 로깅 및 저장
    print()
    print(f"차량번호: {vehicle}")
    print('정비기록')
    print(remaining_records)
    print("평균 기반 추천:")
    for idx, row in closest_avg_records.iterrows():
        print(
            f"차량번호: {row['차량번호']}, 예약 시간: {row['예약 시간']}시, 요일: {list(요일_매핑.keys())[list(요일_매핑.values()).index(row['요일'])]}요일")
    print(f"추천 서비스 센터명: {most_frequent_service_center}")

    avg_results.append({'차량번호': vehicle, '정답': avg_correct_answer})
    service_center_results.append({'차량번호': vehicle, '추천 서비스센터명': most_frequent_service_center})

# 결과 DataFrame으로 변환 및 정답률 계산
avg_results_df = pd.DataFrame(avg_results)
service_center_results_df = pd.DataFrame(service_center_results)

avg_accuracy = avg_results_df['정답'].mean()

print(f"평균 기반 정답률: {avg_accuracy:.2f}")
print("평균 기반 결과와 추천 서비스센터명이 각각 'avg_results.csv'와 'service_center_results.csv'에 저장되었습니다.")

# 결과 저장
avg_results_df.to_csv('avg_results.csv', index=False)
service_center_results_df.to_csv('service_center_results.csv', index=False)
