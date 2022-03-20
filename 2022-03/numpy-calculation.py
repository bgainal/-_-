import numpy as np
print(np.exp(-np.inf))

x = np.arange(5)
x.reshape(5, 1)
x[:, np.newaxis]

c1 = np.arange(15)
c2 = np.arange(15)
c = np.dstack([c1,c2])
c1 = c1.reshape(3,-1)
c2 = c2.reshape(3,-1)
d1 = np.stack([c1,c2])
d2 = np.stack([c1,c2], axis=1)
d3 = np.stack([c1,c2], axis=2)

#배열 만들기
e1=np.zeros((3,3))#0으로 된 3x3
e2=np.ones((3,2))#1으로 된 3x2
e3 = np.hstack([e1,e2])#e1과 e2합침
e2 = np.arange(10,160,10, dtype="f")
e2 = e2.reshape(-1,5)
e4 = np.vstack([e3,e2])
e5 = np.tile(e4,(2,1))

x = np.arange(3)
y = np.arange(5)
X, Y = np.meshgrid(x, y)
f = [list(zip(x, y)) for x, y in zip(X,Y)]

x = np.vstack([range(7)[i:i + 3] for i in range(5)])

z = np.arange(1,31).reshape(5,6)
#전체의 최댓값
z1 = z.max()
#각 행의 합
z2 = z.sum(axis=1)
#각 행의 최댓값
z3 = z.max(axis=1)
#각 열의 평균
z4 = z.mean(axis=0)
#각 열의 최솟값
z5 = z.min(axis=0)

# 다음 배열은 첫번째 행(row)에 학번, 두번째 행에 영어 성적, 세번째 행에 수학 성적을 적은 배열이다. 영어 성적을 기준으로 각 열(column)을 재정렬하라.

g = np.array([[  1,    2,    3,    4],
       [ 46,   99,  100,   71],
       [ 81,   59,   90,  100]])

h = np.argsort(g[1])
g = g[:,h]

a = np.random.choice(5, 3, replace=False)

#동전을 10번 던져 앞면(숫자 1)과 뒷면(숫자 0)이 나오는 가상 실험을 파이썬으로 작성한다.
i1 = np.random.randint(2,size=10)
#주사위를 100번 던져서 나오는 숫자의 평균을 구하라.
i2 = np.random.randint(1,6, size=100).mean()
#가격이 10,000원인 주식이 있다. 이 주식의 일간 수익률(%)은 기댓값이 0%이고 표준편차가 1%인 표준 정규 분포를 따른다고 하자. 250일 동안의 주가를 무작위로 생성하라
i3 = np.random.randn(250)
i4 = np.zeros(250)
i4[0] = i3[0]+10000
for x in range(249):
  i4[x+1] = i4[x]+i3[x+1]