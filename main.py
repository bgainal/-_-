import numpy as np

ar = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(ar)
print(type(ar))

data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
answer = []
for di in data:
    answer.append(2 * di)
print(answer)

c = np.array([[10,20,30,40],[50,60,70,80]])
print(c)

m = np.array([[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14]])

print(m[1][2])# 이 행렬에서 값 7 을 인덱싱한다.
print(m[-1][-1])# 이 행렬에서 값 14 을 인덱싱한다.
print(m[1, 1:3])# 이 행렬에서 배열 [6, 7] 을 슬라이싱한다.
print(m[1:3,2])#이 행렬에서 배열 [7, 12] 을 슬라이싱한다.
print(m[0:2,3:])#이 행렬에서 배열 [[3, 4], [8, 9]] 을 슬라이싱한다.

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
             11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
#이 배열에서 3의 배수를 찾아라.
print(x[x%3==0])
#이 배열에서 4로 나누면 1이 남는 수를 찾아라.
print(x[x%4==1])
#이 배열에서 3으로 나누면 나누어지고 4로 나누면 1이 남는 수를 찾아라.
print(x[(x%3==0)&(x%4==1)])