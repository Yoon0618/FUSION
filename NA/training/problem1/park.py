
import numpy as np
import pymsis
from datetime import datetime

R_earth = 6378.0
pi = np.pi
r_start = R_earth
r_end = R_earth + 1000
n = 1000 # 모든 적분 변수들을 동일하게 n개로 나누셨습니다
phi_start = 0
phi_end = 2*pi
theta_start = 0
theta_end = pi

def density(alt, lon, lat):
    date = np.datetime64(datetime(2025, 1, 1))
    data = pymsis.calculate(date, lon, lat, alt, version = '2.1')
    rho = data[0, 0]
    return float(rho)

phi_list = np.linspace(phi_start, phi_end, n)
theta_list = np.linspace(theta_start, theta_end, n)
r_list = np.linspace(r_start, r_end, n)

d_phi = (phi_end - phi_start)/n
d_theta = (theta_end - theta_start)/n
d_r = (r_end - r_start)/n

M = 0
for r in r_list:
    for theta in theta_list:
        for phi in phi_list: # 삼중 적분도 for문을 중첩해서 잘 구현하셨습니다.
            d_value = density(r, np.rad2deg(theta), np.rad2deg(phi)) # 라디안 단위인 theta, phi도 60분법 단위로 잘 변환하셨습니다.
            # 그런데 density 함수는 alt를 변수로 받습니다. alt는 고도, 즉 지표면으로부터의 거리입니다.
            # 하지만 r은 지구 중심으로부터의 거리입니다. 따라서 r에서 지구 반지름을 빼줘야 합니다.
            # 또한 위도는 0-180도가 아닌 -90도에서 90도 사이의 값입니다.
            # 더해서, theta가 증가할 때 가리키는 점은 북극에서 남극입니다. 위도와는 반대 방향입니다.
            # 따라서 theta를 위도로 변환할 때는 
            # 위도 = 90 - np.rad2deg(theta)로 변환해야 합니다.
            # 이 변환이 맞는지 검증해봅시다. theta가 0이면 북극입니다. 위 변환식에 넣으면 위도는 90도이므로 역시 북극입니다. 
            # theta가 pi/2이면 적도입니다. 위 변환식에 넣으면 위도는 0도이므로 역시 적도입니다.
            m = d_value*(r**2)*np.sin(theta)*d_phi*d_theta*d_r # 미소 부피를 구하는 식도 잘 구현하셨습니다.
            # 다만 코드에서는 보이지 않는 부분에서 오류가 있습니다. 현재 거리의 단위로 km를 사용하고 있습니다.
            # 하지만 density 함수는 밀도를 kg/m^3 단위로 반환합니다.
            # 물리적인 단위가 맞지 않아 이대로는 계산할 수 없습니다.
            # 따라서 km나 m로 통일해야 합니다. 보통은 KMS 단위로 계산합니다.
            # 여기서는 m 단위로 통일하겠습니다. 방법은 간단합니다. 1을 곱합니다.

            # 1 = 1000 m / 1 km이므로, r(km) = r(m) * 1000 입니다. 따라서,
            # m = m * 1000**3
            # 이제 단위가 맞아 m의 단위가 kg이 됩니다.
            M = M + m

print(M)

###########################
# 계산 시간 추정하기

# 위의 삼중 적분 계산을 주석처리 해야 계산 시간 추정이 계산되용

# 먼저 계산은 for문이 3번 중첩되어 있으므로 총 n^3번 수행됩니다.
# 이중 계산이 얼마나 걸리는지 예측하기 위해서, n번 계산하는 시간을 측정합니다.
# 그러면 전체 계산 시간은 n번 계산 시간에 n^2을 곱한 값이 됩니다.
# 이때 시간을 측정하는 코드를 for문 중 어디에 위치해야 할지 잘 생각해야해요.

import time # 파이썬에서 시간을 다루기 위해서는 time 모듈이 필요합니다

# 계산 부분은 틀은 동일하게 유지합니다.
M = 0
for r in r_list:
    for theta in theta_list:

        time_start = time.time() # 계산 시작 시간을 
        
        for phi in phi_list:
            d_value = density(r, np.rad2deg(theta), np.rad2deg(phi))
            m = d_value*(r**2)*np.sin(theta)*d_phi*d_theta*d_r
            M = M + m

        time_end = time.time() # 계산 종료 시간을 기록합니다

        print(theta) # 계산이 어느정도 속도로 이루어 지는지 확인하기 위해서 theta를 출력합니다
        
        end = True
        break # n번 계산 시간만 측정하기 위해서 for문을 중단합니다
    break
n번_계산_시간 = time_end - time_start
예상_계산_시간 = n번_계산_시간 * n * n
print(f"Estimated time for all calculations: {예상_계산_시간:.2f} seconds")
예상_계산_일 = 예상_계산_시간 / (60 * 60 * 24) # 초 단위를 일 단위로 변환합니다
print(f"Estimated time for all calculations: {예상_계산_일:.2f} days")

# 계산해보면, 컴퓨터 성능에 따라 다르지만 약 3일이 걸리는 것을 확인할 수 있습니다.
# 해결 방법은 여러가지가 있습니다만, 가장 빠르게 해볼 수 있는 것은 n을 조정하는 것입니다.
# n을 막 줄이면 정확도가 떨어지므로, 물리적인 상황을 고려해서 적절한 값을 찾아야 합니다.
# 먼저 r에 대한 n은 줄이지 않겠습니다. 대기 밀도는 고도 0 근처에서 급격히 변하기 때문에 그대로 둘게요.
# 대신 theta와 phi에 대한 n을 줄여보겠습니다. 고도 변화에 비해선 위도와 경도 변화에 의한 밀도 변화는 크지 않을 것 같기 때문입니다.

n_r = 1000
n_theta = 180 # 위도는 -90도에서 90도까지 변화하므로, 180개로 나누면 1도 단위로 나눌 수 있습니다.
n_phi = 360 # 경도는 0도에서 360도까지 변화하므로, 360개로 나누면 1도 단위로 나눌 수 있습니다.

# 이러면, 이론상 계산 시간은 약 15분의 1로 줄어들게 됩니다.

# 더 개선할 수도 있습니다. 현재는 r을 균등한 간격으로 나누고 있습니다.
# 하지만 대기 밀도는 낮은 고도에서만 급격히 변하고, 높은 고도에서는 천천히 변합니다. 그래서 높은 고도에서는 간격이 커도 괜찮습니다.
# 따라서, 고도가 낮은 곳은 더 세밀하게 나누고, 고도가 높은 곳은 덜 세밀하게 나누는 것이 좋습니다.
# 이를 위해서는 np.geomspace 함수를 사용합니다. 이 함수는 linspace와 다르게 기하급수적으로 나누는 함수입니다. (geometric기하급수 vs linear선형)
# 예를 들어, np.geomspace(1, 1000, num=4) = [1, 10, 100, 1000]이 됩니다. 즉, 낮은 수쪽에 더 많은 점이 분포하게 됩니다.
# 이를 적용해봅시다.

n_r = 100
r_list = np.geomspace(0.1, 1000, num=n_r) # 0.1에서 1000까지 n_r개의 점을 기하급수적으로 나눕니다. geom은 0을 입력 할 수 없으므로 0.1로 시작합니다.
# geomspace(r_start, r_end, num=n_r)로 하지 않는 이유는, 지표면 기준으로 고도를 나누지, 지구 중심 기준으로 거리를 나누는 것이 아니기 때문입니다.
r_list = r_list + R_earth  # r_list를 지구 중심으로부터의 거리로 변환합니다
import matplotlib.pyplot as plt  # 시각화를 위해 matplotlib.pyplot을 import합니다
plt.scatter(np.linspace(0, 1, n_r), r_list, label='r_list')  # r_list의 변화를 시각화합니다
plt.show()
print(f"r_list: {r_list[:5]}...{r_list[-5:]}")  # r_list의 처음과 끝 5개 요소를 출력하여 확인합니다

# 이러면 이론상 계산은 150분의 1로 줄어듭니다.
# 이렇듯 수치해석은 물리적인 상황을 고려하여 적절한 수치를 선택하는 것이 중요합니다.