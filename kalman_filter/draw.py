import matplotlib.pyplot as plt

import numpy as np

# read raw data

file = open('/home/phoenix/Downloads/kalman_filter/homework_data_4.txt', 'r')
data = file.readlines()
file.close()
time=[]
data_raw=[]
data = [line.strip() for line in data]
for i in range(len(data)):
    print(data[i].split(' '))
    time.append(float(data[i].split(' ')[0]))
    data_raw.append(float(data[i].split(' ')[1]))

# read flitted data
file = open('/home/phoenix/Downloads/kalman_filter/build/kalman_output.txt', 'r')
flitted_data = file.readlines()
file.close()
flitted_data = [line.strip() for line in flitted_data]
data_flitted = []
for i in range(len(flitted_data)):
    data_flitted.append(float(flitted_data[i].split(' ')[1]))

# plot
plt.figure(figsize=(10, 5))
plt.plot(time, data_raw, label='raw data', color='blue', linewidth=1)
plt.plot(time, data_flitted, label='flitted data', color='red', linewidth=1)
plt.title('Kalman Filter')
plt.xlabel('time')
plt.ylabel('data')
plt.legend()
plt.grid()
plt.show()