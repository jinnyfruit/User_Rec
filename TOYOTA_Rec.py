import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from datetime import datetime, timedelta
import random

# Load dataset
df = pd.read_csv('data.csv',low_memory=False)

# Data preprocessing
df['예약 시간'] = df['예약 시간'].str.replace('시', '').astype(int)
요일_매핑 = {'월': 0, '화': 1, '수': 2, '목': 3, '금': 4, '토': 5, '일': 6}
df['요일'] = df['요일'].map(요일_매핑)

# Select one vehicle's maintenance history randomly
unique_vehicles = df['차량번호'].unique()
selected_vehicle = np.random.choice(unique_vehicles)
vehicle_history = df[df['차량번호'] == selected_vehicle]
print(selected_vehicle)
print()
print(vehicle_history)
print()

# Calculate average reservation time and day of the week
avg_reservation_time = vehicle_history['예약 시간'].mean()
avg_day_of_week = vehicle_history['요일'].mean()
print(avg_reservation_time)
print()
print(avg_day_of_week)
print()

# Determine the most frequently visited service center
most_frequent_center = vehicle_history['서비스센터명'].mode()[0]
print(most_frequent_center)
print()

# Visualize the average reservation time and day of the week
plt.figure(figsize=(10, 6))
sns.barplot(x=['Average Reservation Time', 'Average Day of Week'], y=[avg_reservation_time, avg_day_of_week])
plt.ylabel('Average Value')
plt.title('Average Reservation Time and Day of Week for Vehicle ' + selected_vehicle)
plt.show()

# Generate 5 random available reservation dates
today = datetime.today()
available_dates = [today + timedelta(days=i) for i in range(1, 6)]
print(available_dates)
print()

# Convert avg_day_of_week to closest day in available_dates
avg_day_int = round(avg_day_of_week)
recommended_dates = []

for date in available_dates:
    if date.weekday() == avg_day_int:
        recommended_dates.append(date)
        if len(recommended_dates) == 2:
            break

# If less than 2 dates were found, fill in with closest available dates
while len(recommended_dates) < 2:
    closest_date = min(available_dates, key=lambda x: abs(x.weekday() - avg_day_int))
    recommended_dates.append(closest_date)
    available_dates.remove(closest_date)

# Print results
print(f"Most frequently visited service center: {most_frequent_center}")
print(f"추천 예약 시간: {round(avg_reservation_time)}:00")
print(f"추천 요일: {avg_day_of_week}")
print("가능한 추천일자:", recommended_dates)
