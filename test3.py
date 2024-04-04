import pandas as pd
df = pd.read_csv('./data/TOYOTA_data.csv')
unique_names_per_phone = df.groupby('예약 휴대폰번호')['예약 고객명'].nunique()

# 고유한 이름이 1개를 초과하는 전화번호를 필터링
phones_with_multiple_names = unique_names_per_phone[unique_names_per_phone > 1].index

# 해당 전화번호를 가진 모든 레코드를 필터링
multiple_names_df = df[df['예약 휴대폰번호'].isin(phones_with_multiple_names)]

# 전화번호 한개당 이름이 두개 이상 조회되는 모든 번호들을 따로 저장
multiple_names_df.to_csv('TOYOTA_multiple_names_per_phone.csv', index=False)

total_phones = df['예약 휴대폰번호'].nunique()
phones_with_multiple_names_count = len(phones_with_multiple_names)
percentage = (phones_with_multiple_names_count / total_phones) * 100

print(f"전체 전화번호 중에서 이름이 여러 개인 경우는 {percentage:.2f}% 입니다.")
print("TOYOTA_multiple_names_per_phone.csv' 파일에 결과가 저장되었습니다.")
