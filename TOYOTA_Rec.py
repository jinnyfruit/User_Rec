import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
df = pd.read_csv('result/data.csv', low_memory=False)

# Create a copy of the dataframe to preserve original '요일' and '예약 시간' values
df_original = df.copy()

# Data preprocessing
df['예약 시간'] = df['예약 시간'].str.replace('시', '').astype(int)
요일_매핑 = {'월': 0, '화': 1, '수': 2, '목': 3, '금': 4, '토': 5, '일': 6}
df['요일'] = df['요일'].map(요일_매핑)

# Input vehicle number from terminal
selected_vehicle_number = input("차량번호를 입력해주세요: ")

if selected_vehicle_number in df_original['차량번호'].values:
    # Use the original dataframe to display the history with unprocessed '요일' and '예약 시간'
    vehicle_history = df_original[df_original['차량번호'] == selected_vehicle_number]
    print(vehicle_history)

    # Use the processed dataframe for analysis
    vehicle_history_processed = df[df['차량번호'] == selected_vehicle_number]

    # Calculate average reservation time and day of the week using the processed data
    avg_reservation_time = vehicle_history_processed['예약 시간'].mean()
    avg_day_of_week = vehicle_history_processed['요일'].mean()
    print(f"Average reservation time: {avg_reservation_time}시")
    print(f"Average day of week (0:월, 6:일): {avg_day_of_week}")
    # Prepare the DataFrame for available reservation times and days
    reservation_df = pd.DataFrame({
        '예약가능일자': [
            '2024-02-27', '2024-02-27',
            '2024-02-28', '2024-02-28',
            '2024-03-01', '2024-03-01',
            '2024-03-04', '2024-03-04',
            '2024-03-05', '2024-03-05',
            '2024-03-02', '2024-03-02',
        ],
        '요일': [
            '수', '금',
            '일', '목',
            '월', '수',
            '월', '일',
            '수', '화',
            '토', '토',
        ],
        '예약가능 시간': [
            '11시', '10시',
            '14시', '12시',
            '13시', '12시',
            '11시', '16시',
            '17시', '10시',
            '09시', '10시',
        ]
    }).assign(
        요일=lambda df: df['요일'].map(요일_매핑),
        예약가능_시간=lambda df: df['예약가능 시간'].str.replace('시', '').astype(int)
    )

    # Calculate distance from average for each available reservation
    reservation_df['distance'] = np.sqrt(
        (reservation_df['요일'] - avg_day_of_week) ** 2 +
        (reservation_df['예약가능_시간'] - avg_reservation_time) ** 2
    )

    # Find the two nearest available reservations
    nearest_reservations = reservation_df.nsmallest(2, 'distance').copy()

    # Convert numeric day of the week back to string representation
    요일_역매핑 = {v: k for k, v in 요일_매핑.items()}
    nearest_reservations['요일'] = nearest_reservations['요일'].map(요일_역매핑)
    nearest_reservations['예약가능_시간'] = nearest_reservations['예약가능_시간'].astype(str) + '시'

    # Show nearest available reservations
    print(nearest_reservations)

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(reservation_df['요일'], reservation_df['예약가능_시간'], color='blue', label='Available Reservations')
    plt.scatter(avg_day_of_week, avg_reservation_time, color='red', label=f'Average for {selected_vehicle_number}')
    plt.title(f'Average Reservation Time and Day of Week for Vehicle {selected_vehicle_number} and Available Reservations')
    plt.xlabel('Day of the Week')
    plt.ylabel('Reservation Time')
    plt.xticks(range(0, 7), ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'])
    plt.yticks(range(9, 19), [f'{hour}:00' for hour in range(9, 19)])
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print(f"Vehicle number {selected_vehicle_number} not found in the dataset.")