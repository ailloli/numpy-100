
# NumPy 예제 100개

* 스칼라 = 단일 값, 벡터 = 1차원 배열, 행렬 = 2차원 배열
* 배열의 '모양(shape)'이 (2, 3, 4)라는 건 2x3x4 배열임을 의미
* '랜덤'은 따로 언급하지 않는 이상 0과 1 사이의 무작위 실수를 의미

#### 1.  `np`로 NumPy 패키지 가져오기 (★☆☆)


```python
import numpy as np
```

#### 2.  NumPy 버전과 환경 설정 정보 출력하기 (★☆☆)


```python
print(np.__version__)
np.show_config()
```

#### 3.  크기 10의 영벡터 만들어 보기 (★☆☆)


```python
Z = np.zeros(10)
print(Z)
```

#### 4.  배열이 메모리에 저장되어 있는 크기 알아보기 (★☆☆)


```python
Z = np.zeros((10,10))
print("%d bytes" % (Z.size * Z.itemsize))
```

#### 5.  터미널에서 NumPy add 함수 설명 가져오기 (★☆☆)


```python
%run `python -c "import numpy; numpy.info(numpy.add)"`
```

#### 6.  5번째 값만 1이고 나머지는 다 0인 크기 10의 벡터 만들어 보기 (★☆☆)


```python
Z = np.zeros(10)
Z[4] = 1
print(Z)
```

#### 7.  10부터 49까지의 수가 연속적으로 들어가 있는 벡터 (\[10, 11, 12, ..., 48, 49\]) 만들기 (★☆☆)


```python
Z = np.arange(10,50)
print(Z)
```

#### 8.  벡터 뒤집기 (첫 원소가 마지막 원소가 됨) (★☆☆)


```python
Z = np.arange(50)
Z = Z[::-1]
print(Z)
```

#### 9.  0에서 8까지의 수가 연속적으로 들어 있는 3x3 행렬 만들기 (★☆☆)


```python
Z = np.arange(9).reshape(3,3)
print(Z)
```

#### 10. 배열 \[1,2,0,0,4,0\]에서 0이 아닌 원소들의 위치 찾기 (★☆☆)


```python
nz = np.nonzero([1,2,0,0,4,0])
print(nz)
```

#### 11. 3x3 단위행렬 만들기 (★☆☆)


```python
Z = np.eye(3)
print(Z)
```

#### 12. 0에서 1 사이의 랜덤한 실수로 채워진 3x3x3 배열 만들기 (★☆☆)


```python
Z = np.random.random((3,3,3))
print(Z)
```

#### 13. 10x10 랜덤 행렬을 만들고 최대, 최소값 구하기 (★☆☆)


```python
Z = np.random.random((10,10))
Zmin, Zmax = Z.min(), Z.max()
print(Zmin, Zmax)
```

#### 14. 크기 30의 랜덤 벡터를 만들고 평균 구하기 (★☆☆)


```python
Z = np.random.random(30)
m = Z.mean()
print(m)
```

#### 15. 안은 0으로, 테두리는 1로 채워진 행렬 만들기 (★☆☆)


```python
Z = np.ones((10,10))
Z[1:-1,1:-1] = 0
print(Z)
```

#### 16. 미리 존재하는 행렬에 0으로 채워진 테두리를 상하좌우 한 줄씩 추가하기 (★☆☆)


```python
Z = np.ones((5,5))
Z = np.pad(Z, pad_width=1, mode='constant', constant_values=0)
print(Z)
```

#### 17. 다음 코드의 실행 결과는? (★☆☆)


```python
print(0 * np.nan)
print(np.nan == np.nan)
print(np.inf > np.nan)
print(np.nan - np.nan)
print(np.nan in set([np.nan]))
print(0.3 == 3 * 0.1)
```

#### 18. 왼쪽 위에서 오른쪽 아래로 내려가는 대각선 바로 아래 값들이 1, 2, 3, 4로 채워진 5x5 행렬 만들기 (★☆☆)


```python
Z = np.diag(1+np.arange(4),k=-1)
print(Z)
```

#### 19. 8x8 체스판 행렬 만들기 (★☆☆)


```python
Z = np.zeros((8,8),dtype=int)
Z[1::2,::2] = 1
Z[::2,1::2] = 1
print(Z)
```

#### 20. 6x7x8 배열(혹은 모양이 (6, 7, 8)인 배열)에서 100번째 원소의 좌표 (x, y, z)는?


```python
print(np.unravel_index(99,(6,7,8)))
```

#### 21. tile 함수를 이용해 8x8 체스판 행렬 만들기 (★☆☆)


```python
Z = np.tile( np.array([[0,1],[1,0]]), (4,4))
print(Z)
```

#### 22. 5x5 랜덤 행렬 정규화하기 (★☆☆)


```python
Z = np.random.random((5,5))
Z = (Z - np.mean (Z)) / (np.std (Z))
print(Z)
```

#### 23. 4개의 unsigned byte로 색상을 표현하는 커스텀 타입 만들기 (빨강, 초록, 파랑, 투명도) (★☆☆)


```python
color = np.dtype([("r", np.ubyte, 1),
                  ("g", np.ubyte, 1),
                  ("b", np.ubyte, 1),
                  ("a", np.ubyte, 1)])
```

#### 24. 5x3 행렬과 3x2 행렬의 곱셈 (★☆☆)


```python
Z = np.dot(np.ones((5,3)), np.ones((3,2)))
print(Z)

# Alternative solution, in Python 3.5 and above
Z = np.ones((5,3)) @ np.ones((3,2))
print(Z)
```

#### 25. 1차원 배열에서, 3하고 8 사이의 값들에만 -1 곱하기 (★☆☆)


```python
# Author: Evgeni Burovski

Z = np.arange(11)
Z[(3 < Z) & (Z <= 8)] *= -1
print(Z)
```

#### 26. 다음 코드의 실행 결과는? (★☆☆)


```python
# Author: Jake VanderPlas

print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))
```

#### 27. 정수 벡터 Z에 대해 다음 중 올바른 식은? (★☆☆)


```python
Z**Z
2 << Z >> 2
Z <- Z
1j*Z
Z/1/1
Z<Z>Z
```

#### 28. 다음 코드의 실행 결과는?


```python
print(np.array(0) / np.array(0))
print(np.array(0) // np.array(0))
print(np.array([np.nan]).astype(int).astype(float))
```

#### 29. 실수 배열에서 모든 원소를 0에서 먼 쪽의 정수로 바꾸기 (★☆☆)


```python
# Author: Charles R Harris

Z = np.random.uniform(-10,+10,10)
print (np.copysign(np.ceil(np.abs(Z)), Z))
```

#### 30. 두 배열의 교집합 찾기 (★☆☆)


```python
Z1 = np.random.randint(0,10,10)
Z2 = np.random.randint(0,10,10)
print(np.intersect1d(Z1,Z2))
```

#### 31. NumPy의 모든 경고 무시하기 (추천하지 않음) (★☆☆)


```python
# Suicide mode on
defaults = np.seterr(all="ignore")
Z = np.ones(1) / 0

# Back to sanity
_ = np.seterr(**defaults)
```

Context manager를 사용한 다른 방법:

```python
with np.errstate(divide='ignore'):
    Z = np.ones(1) / 0
```

#### 32. 다음 코드의 실행 결과는? (★☆☆)


```python
np.sqrt(-1) == np.emath.sqrt(-1)
```

#### 33. 어제, 오늘, 내일 날짜 가져오기 (★☆☆)


```python
yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
today     = np.datetime64('today', 'D')
tomorrow  = np.datetime64('today', 'D') + np.timedelta64(1, 'D')
```

#### 34. 2016년 7월의 모든 날짜 가져오기 (★★☆)


```python
Z = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
print(Z)
```

#### 35. 새로운 값 선언이나 복사 없이 벡터 A, B에 대해 ((A+B)\*(-A/2)) 계산하기 (★★☆)


```python
A = np.ones(3)*1
B = np.ones(3)*2
C = np.ones(3)*3
np.add(A,B,out=B)
np.divide(A,2,out=A)
np.negative(A,out=A)
np.multiply(A,B,out=A)
```

#### 36. 다섯 가지 방법으로 랜덤 실수 배열의 정수 부분 가져오기 (★★☆)


```python
Z = np.random.uniform(0,10,10)

print (Z - Z%1)
print (np.floor(Z))
print (np.ceil(Z)-1)
print (Z.astype(int))
print (np.trunc(Z))
```

#### 37. 세로 값이 0, 1, 2, 3, 4로 채워진 5x5 행렬 만들기 (★★☆)


```python
Z = np.zeros((5,5))
Z += np.arange(5)
print(Z)
```

#### 38. 정수 10개를 생성해 배열을 만드는 제네레이터 함수 만들기 (★☆☆)


```python
def generate():
    for x in range(10):
        yield x
Z = np.fromiter(generate(),dtype=float,count=-1)
print(Z)
```

#### 39. 1/11부터 10/11까지의 값을 포함하는 크기 10의 벡터 만들기 (★★☆)


```python
Z = np.linspace(0,1,11,endpoint=False)[1:]
print(Z)
```

#### 40. 크기 10의 랜덤 벡터를  (★★☆)


```python
Z = np.random.random(10)
Z.sort()
print(Z)
```

#### 41. 작은 배열에서 np.sum보다 빠르게 총합 계산하기 (★★☆)


```python
# Author: Evgeni Burovski

Z = np.arange(10)
np.add.reduce(Z)
```

#### 42. 랜덤 배열 A, B가 같은지 확인하기 (★★☆)


```python
A = np.random.randint(0,2,5)
B = np.random.randint(0,2,5)

# Assuming identical shape of the arrays and a tolerance for the comparison of values
equal = np.allclose(A,B)
print(equal)

# Checking both the shape and the element values, no tolerance (values have to be exactly equal)
equal = np.array_equal(A,B)
print(equal)
```

#### 43. 배열을 immutable(수정 불가능)하게 만들기 (★★☆)


```python
Z = np.zeros(10)
Z.flags.writeable = False
Z[0] = 1
```

#### 44. 직교좌표계상의 점들의 좌표를 의미하는 랜덤 10x2 행렬을 만들고, 극좌표계로 변환하기 (★★☆)


```python
Z = np.random.random((10,2))
X,Y = Z[:,0], Z[:,1]
R = np.sqrt(X**2+Y**2)
T = np.arctan2(Y,X)
print(R)
print(T)
```

#### 45. 크기 10의 랜덤 벡터를 만들고 최댓값만 0으로 교체하기 (★★☆)


```python
Z = np.random.random(10)
Z[Z.argmax()] = 0
print(Z)
```

#### 46. \[0,1\]x\[0,1\] 영역을 16등분하는 점들의 좌표를 포함하는 structured array 만들기 (★★☆)


```python
Z = np.zeros((5,5), [('x',float),('y',float)])
Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,5),
                             np.linspace(0,1,5))
print(Z)
```

####  47. 배열 X와 Y에 대해 Cauchy 행렬 C (Cij =1/(xi - yj)) 계산하기 (★★☆)


```python
# Author: Evgeni Burovski

X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y)
print(np.linalg.det(C))
```

#### 48. NumPy 스칼라 자료형들의 최대, 최소값 출력하기 (★★☆)


```python
for dtype in [np.int8, np.int32, np.int64]:
   print(np.iinfo(dtype).min)
   print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
   print(np.finfo(dtype).min)
   print(np.finfo(dtype).max)
   print(np.finfo(dtype).eps)
```

#### 49. 배열의 모든 원소 출력하기 (★★☆)


```python
np.set_printoptions(threshold=np.nan)
Z = np.zeros((16,16))
print(Z)
```

#### 50. 벡터에서 어떤 스칼라에 가장 가까운 값  (★★☆)


```python
Z = np.arange(100)
v = np.random.uniform(0,100)
index = (np.abs(Z-v)).argmin()
print(Z[index])
```

#### 51. 위치 (x, y)와 색상 (r, g, b)를 포함하는 structured array 만들기 (★★☆)


```python
Z = np.zeros(10, [ ('position', [ ('x', float, 1),
                                  ('y', float, 1)]),
                   ('color',    [ ('r', float, 1),
                                  ('g', float, 1),
                                  ('b', float, 1)])])
print(Z)
```

#### 52. 좌표를 나타내는 모양이 (100, 2)인 랜덤 배열에서, 점과 점 사이의 거리 계산하기 (★★☆)


```python
Z = np.random.random((10,2))
X,Y = np.atleast_2d(Z[:,0], Z[:,1])
D = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)
print(D)

# Much faster with scipy
import scipy
# Thanks Gavin Heverly-Coulson (#issue 1)
import scipy.spatial

Z = np.random.random((10,2))
D = scipy.spatial.distance.cdist(Z,Z)
print(D)
```

#### 53. float (32비트) 배열을 integer (32비트) 배열로 변환하기 (★★☆)


```python
Z = np.arange(10, dtype=np.float32)
Z = Z.astype(np.int32, copy=False)
print(Z)
```

#### 54. 주어진 파일을 해석해 배열로 변환하기 (★★☆)


```python
from io import StringIO

# Fake file
s = StringIO("""1, 2, 3, 4, 5\n
                6,  ,  , 7, 8\n
                 ,  , 9,10,11\n""")
Z = np.genfromtxt(s, delimiter=",", dtype=np.int)
print(Z)
```

#### 55. NumPy 배열에서 enumerate와 같은 동작 (★★☆)


```python
Z = np.arange(9).reshape(3,3)
for index, value in np.ndenumerate(Z):
    print(index, value)
for index in np.ndindex(Z.shape):
    print(index, Z[index])
```

#### 56. 2차원 가우시안 분포 비슷한 배열 생성 (★★☆)


```python
X, Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
D = np.sqrt(X*X+Y*Y)
sigma, mu = 1.0, 0.0
G = np.exp(-( (D-mu)**2 / ( 2.0 * sigma**2 ) ) )
print(G)
```

#### 57. 2차원 배열에 원소 p개 랜덤하게 넣기 (★★☆)


```python
# Author: Divakar

n = 10
p = 3
Z = np.zeros((n,n))
np.put(Z, np.random.choice(range(n*n), p, replace=False),1)
print(Z)
```

#### 58. 행렬에서 매 열의 평균을 0으로 만들기 (★★☆)


```python
# Author: Warren Weckesser

X = np.random.rand(5, 10)

# Recent versions of numpy
Y = X - X.mean(axis=1, keepdims=True)

# Older versions of numpy
Y = X - X.mean(axis=1).reshape(-1, 1)

print(Y)
```

#### 59. n번째 행을 기준으로 배열 정렬하기 (★★☆)


```python
# Author: Steve Tjoa

Z = np.random.randint(0,10,(3,3))
print(Z)
print(Z[Z[:,1].argsort()])
```

#### 60. 어떤 행렬에서 모든 항목이 0인 행이 있는지 검사하기 (★★☆)


```python
# Author: Warren Weckesser

Z = np.random.randint(0,3,(3,10))
print((~Z.any(axis=0)).any())
```

#### 61. 배열에서 어떤 임의의 값에서 가장 가까운 값 찾기 (★★☆)


```python
Z = np.random.uniform(0,1,10)
z = 0.5
m = Z.flat[np.abs(Z - z).argmin()]
print(m)
```

#### 62. iterator를 이용해 (1, 3) 배열과 (3, 1) 배열의 합 계산 (★★☆)


```python
A = np.arange(3).reshape(3,1)
B = np.arange(3).reshape(1,3)
it = np.nditer([A,B,None])
for x,y,z in it: z[...] = x + y
print(it.operands[2])
```

#### 63. 이름 항목이 있는 배열 클래스 만들기 (★★☆)


```python
class NamedArray(np.ndarray):
    def __new__(cls, array, name="no name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'name', "no name")

Z = NamedArray(np.arange(10), "range_10")
print (Z.name)
```

#### 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? (★★★)


```python
# Author: Brett Olsen

Z = np.ones(10)
I = np.random.randint(0,len(Z),20)
Z += np.bincount(I, minlength=len(Z))
print(Z)

# Another solution
# Author: Bartosz Telenczuk
np.add.at(Z, I, 1)
print(Z)
```

#### 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★)


```python
# Author: Alan G Isaac

X = [1,2,3,4,5,6]
I = [1,3,9,3,4,1]
F = np.bincount(I,X)
print(F)
```

#### 66. 모양이 (w,h,3) 이고 데이터 타입이 ubyte 인 이미지에서 등장한 색상의 수를 구하기 (★★★)


```python
# Author: Nadav Horesh

w,h = 16,16
I = np.random.randint(0,2,(h,w,3)).astype(np.ubyte)
F = I[...,0]*256*256 + I[...,1]*256 +I[...,2]
n = len(np.unique(F))
print(np.unique(I))
```

#### 67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★)


```python
A = np.random.randint(0,10,(3,4,3,4))
# solution by passing a tuple of axes (introduced in numpy 1.7.0)
sum = A.sum(axis=(-2,-1))
print(sum)
# solution by flattening the last two dimensions into one
# (useful for functions that don't accept tuples for axis argument)
sum = A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1)
print(sum)
```

#### 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset  indices? (★★★)


```python
# Author: Jaime Fernández del Río

D = np.random.uniform(0,1,100)
S = np.random.randint(0,10,100)
D_sums = np.bincount(S, weights=D)
D_counts = np.bincount(S)
D_means = D_sums / D_counts
print(D_means)

# Pandas solution as a reference due to more intuitive code
import pandas as pd
print(pd.Series(D).groupby(S).mean())
```

#### 69. How to get the diagonal of a dot product? (★★★)


```python
# Author: Mathieu Blondel

A = np.random.uniform(0,1,(5,5))
B = np.random.uniform(0,1,(5,5))

# Slow version  
np.diag(np.dot(A, B))

# Fast version
np.sum(A * B.T, axis=1)

# Faster version
np.einsum("ij,ji->i", A, B)
```

#### 70. Consider the vector \[1, 2, 3, 4, 5\], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★)


```python
# Author: Warren Weckesser

Z = np.array([1,2,3,4,5])
nz = 3
Z0 = np.zeros(len(Z) + (len(Z)-1)*(nz))
Z0[::nz+1] = Z
print(Z0)
```

#### 71. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)? (★★★)


```python
A = np.ones((5,5,3))
B = 2*np.ones((5,5))
print(A * B[:,:,None])
```

#### 72. How to swap two rows of an array? (★★★)


```python
# Author: Eelco Hoogendoorn

A = np.arange(25).reshape(5,5)
A[[0,1]] = A[[1,0]]
print(A)
```

#### 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the  triangles (★★★)


```python
# Author: Nicolas P. Rougier

faces = np.random.randint(0,100,(10,3))
F = np.roll(faces.repeat(2,axis=1),-1,axis=1)
F = F.reshape(len(F)*3,2)
F = np.sort(F,axis=1)
G = F.view( dtype=[('p0',F.dtype),('p1',F.dtype)] )
G = np.unique(G)
print(G)
```

#### 74. Given an array C that is a bincount, how to produce an array A such that np.bincount(A) == C? (★★★)


```python
# Author: Jaime Fernández del Río

C = np.bincount([1,1,2,3,4,4,6])
A = np.repeat(np.arange(len(C)), C)
print(A)
```

#### 75. How to compute averages using a sliding window over an array? (★★★)


```python
# Author: Jaime Fernández del Río

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
Z = np.arange(20)
print(moving_average(Z, n=3))
```

#### 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z\[0\],Z\[1\],Z\[2\]) and each subsequent row is  shifted by 1 (last row should be (Z\[-3\],Z\[-2\],Z\[-1\]) (★★★)


```python
# Author: Joe Kington / Erik Rigtorp
from numpy.lib import stride_tricks

def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return stride_tricks.as_strided(a, shape=shape, strides=strides)
Z = rolling(np.arange(10), 3)
print(Z)
```

#### 77. How to negate a boolean, or to change the sign of a float inplace? (★★★)


```python
# Author: Nathaniel J. Smith

Z = np.random.randint(0,2,100)
np.logical_not(Z, out=Z)

Z = np.random.uniform(-1.0,1.0,100)
np.negative(Z, out=Z)
```

#### 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i  (P0\[i\],P1\[i\])? (★★★)


```python
def distance(P0, P1, p):
    T = P1 - P0
    L = (T**2).sum(axis=1)
    U = -((P0[:,0]-p[...,0])*T[:,0] + (P0[:,1]-p[...,1])*T[:,1]) / L
    U = U.reshape(len(U),1)
    D = P0 + U*T - p
    return np.sqrt((D**2).sum(axis=1))

P0 = np.random.uniform(-10,10,(10,2))
P1 = np.random.uniform(-10,10,(10,2))
p  = np.random.uniform(-10,10,( 1,2))
print(distance(P0, P1, p))
```

#### 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P\[j\]) to each line i (P0\[i\],P1\[i\])? (★★★)


```python
# Author: Italmassov Kuanysh

# based on distance function from previous question
P0 = np.random.uniform(-10, 10, (10,2))
P1 = np.random.uniform(-10,10,(10,2))
p = np.random.uniform(-10, 10, (10,2))
print(np.array([distance(P0,P1,p_i) for p_i in p]))
```

#### 80. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a `fill` value when necessary) (★★★)


```python
# Author: Nicolas Rougier

Z = np.random.randint(0,10,(10,10))
shape = (5,5)
fill  = 0
position = (1,1)

R = np.ones(shape, dtype=Z.dtype)*fill
P  = np.array(list(position)).astype(int)
Rs = np.array(list(R.shape)).astype(int)
Zs = np.array(list(Z.shape)).astype(int)

R_start = np.zeros((len(shape),)).astype(int)
R_stop  = np.array(list(shape)).astype(int)
Z_start = (P-Rs//2)
Z_stop  = (P+Rs//2)+Rs%2

R_start = (R_start - np.minimum(Z_start,0)).tolist()
Z_start = (np.maximum(Z_start,0)).tolist()
R_stop = np.maximum(R_start, (R_stop - np.maximum(Z_stop-Zs,0))).tolist()
Z_stop = (np.minimum(Z_stop,Zs)).tolist()

r = [slice(start,stop) for start,stop in zip(R_start,R_stop)]
z = [slice(start,stop) for start,stop in zip(Z_start,Z_stop)]
R[r] = Z[z]
print(Z)
print(R)
```

#### 81. Consider an array Z = \[1,2,3,4,5,6,7,8,9,10,11,12,13,14\], how to generate an array R = \[\[1,2,3,4\], \[2,3,4,5\], \[3,4,5,6\], ..., \[11,12,13,14\]\]? (★★★)


```python
# Author: Stefan van der Walt

Z = np.arange(1,15,dtype=np.uint32)
R = stride_tricks.as_strided(Z,(11,4),(4,4))
print(R)
```

#### 82. Compute a matrix rank (★★★)


```python
# Author: Stefan van der Walt

Z = np.random.uniform(0,1,(10,10))
U, S, V = np.linalg.svd(Z) # Singular Value Decomposition
rank = np.sum(S > 1e-10)
print(rank)
```

#### 83. How to find the most frequent value in an array?


```python
Z = np.random.randint(0,10,50)
print(np.bincount(Z).argmax())
```

#### 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★)


```python
# Author: Chris Barker

Z = np.random.randint(0,5,(10,10))
n = 3
i = 1 + (Z.shape[0]-3)
j = 1 + (Z.shape[1]-3)
C = stride_tricks.as_strided(Z, shape=(i, j, n, n), strides=Z.strides + Z.strides)
print(C)
```

#### 85. Create a 2D array subclass such that Z\[i,j\] == Z\[j,i\] (★★★)


```python
# Author: Eric O. Lebigot
# Note: only works for 2d array and value setting using indices

class Symetric(np.ndarray):
    def __setitem__(self, index, value):
        i,j = index
        super(Symetric, self).__setitem__((i,j), value)
        super(Symetric, self).__setitem__((j,i), value)

def symetric(Z):
    return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(Symetric)

S = symetric(np.random.randint(0,10,(5,5)))
S[2,3] = 42
print(S)
```

#### 86. Consider a set of p matrices wich shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1)) (★★★)


```python
# Author: Stefan van der Walt

p, n = 10, 20
M = np.ones((p,n,n))
V = np.ones((p,n,1))
S = np.tensordot(M, V, axes=[[0, 2], [0, 1]])
print(S)

# It works, because:
# M is (p,n,n)
# V is (p,n,1)
# Thus, summing over the paired axes 0 and 0 (of M and V independently),
# and 2 and 1, to remain with a (n,1) vector.
```

#### 87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★)


```python
# Author: Robert Kern

Z = np.ones((16,16))
k = 4
S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                       np.arange(0, Z.shape[1], k), axis=1)
print(S)
```

#### 88. How to implement the Game of Life using numpy arrays? (★★★)


```python
# Author: Nicolas Rougier

def iterate(Z):
    # Count neighbours
    N = (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
         Z[1:-1,0:-2]                + Z[1:-1,2:] +
         Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])

    # Apply rules
    birth = (N==3) & (Z[1:-1,1:-1]==0)
    survive = ((N==2) | (N==3)) & (Z[1:-1,1:-1]==1)
    Z[...] = 0
    Z[1:-1,1:-1][birth | survive] = 1
    return Z

Z = np.random.randint(0,2,(50,50))
for i in range(100): Z = iterate(Z)
print(Z)
```

#### 89. How to get the n largest values of an array (★★★)


```python
Z = np.arange(10000)
np.random.shuffle(Z)
n = 5

# Slow
print (Z[np.argsort(Z)[-n:]])

# Fast
print (Z[np.argpartition(-Z,n)[:n]])
```

#### 90. Given an arbitrary number of vectors, build the cartesian product (every combinations of every item) (★★★)


```python
# Author: Stefan Van der Walt

def cartesian(arrays):
    arrays = [np.asarray(a) for a in arrays]
    shape = (len(x) for x in arrays)

    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(arrays), -1).T

    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]

    return ix

print (cartesian(([1, 2, 3], [4, 5], [6, 7])))
```

#### 91. How to create a record array from a regular array? (★★★)


```python
Z = np.array([("Hello", 2.5, 3),
              ("World", 3.6, 2)])
R = np.core.records.fromarrays(Z.T,
                               names='col1, col2, col3',
                               formats = 'S8, f8, i8')
print(R)
```

#### 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★)


```python
# Author: Ryan G.

x = np.random.rand(5e7)

%timeit np.power(x,3)
%timeit x*x*x
%timeit np.einsum('i,i,i->i',x,x,x)
```

#### 93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (★★★)


```python
# Author: Gabe Schwartz

A = np.random.randint(0,5,(8,3))
B = np.random.randint(0,5,(2,2))

C = (A[..., np.newaxis, np.newaxis] == B)
rows = np.where(C.any((3,1)).all(1))[0]
print(rows)
```

#### 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. \[2,2,3\]) (★★★)


```python
# Author: Robert Kern

Z = np.random.randint(0,5,(10,3))
print(Z)
# solution for arrays of all dtypes (including string arrays and record arrays)
E = np.all(Z[:,1:] == Z[:,:-1], axis=1)
U = Z[~E]
print(U)
# soluiton for numerical arrays only, will work for any number of columns in Z
U = Z[Z.max(axis=1) != Z.min(axis=1),:]
print(U)
```

#### 95. Convert a vector of ints into a matrix binary representation (★★★)


```python
# Author: Warren Weckesser

I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128])
B = ((I.reshape(-1,1) & (2**np.arange(8))) != 0).astype(int)
print(B[:,::-1])

# Author: Daniel T. McDonald

I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128], dtype=np.uint8)
print(np.unpackbits(I[:, np.newaxis], axis=1))
```

#### 96. Given a two dimensional array, how to extract unique rows? (★★★)


```python
# Author: Jaime Fernández del Río

Z = np.random.randint(0,2,(6,3))
T = np.ascontiguousarray(Z).view(np.dtype((np.void, Z.dtype.itemsize * Z.shape[1])))
_, idx = np.unique(T, return_index=True)
uZ = Z[idx]
print(uZ)

# Author: Andreas Kouzelis
# NumPy >= 1.13
uZ = np.unique(Z, axis=0)
print(uZ)
```

#### 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★)


```python
# Author: Alex Riley
# Make sure to read: http://ajcr.net/Basic-guide-to-einsum/

A = np.random.uniform(0,1,10)
B = np.random.uniform(0,1,10)

np.einsum('i->', A)       # np.sum(A)
np.einsum('i,i->i', A, B) # A * B
np.einsum('i,i', A, B)    # np.inner(A, B)
np.einsum('i,j->ij', A, B)    # np.outer(A, B)
```

#### 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★)?


```python
# Author: Bas Swinckels

phi = np.arange(0, 10*np.pi, 0.1)
a = 1
x = a*phi*np.cos(phi)
y = a*phi*np.sin(phi)

dr = (np.diff(x)**2 + np.diff(y)**2)**.5 # segment lengths
r = np.zeros_like(x)
r[1:] = np.cumsum(dr)                # integrate path
r_int = np.linspace(0, r.max(), 200) # regular spaced path
x_int = np.interp(r_int, r, x)       # integrate path
y_int = np.interp(r_int, r, y)
```

#### 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n. (★★★)


```python
# Author: Evgeni Burovski

X = np.asarray([[1.0, 0.0, 3.0, 8.0],
                [2.0, 0.0, 1.0, 1.0],
                [1.5, 2.5, 1.0, 0.0]])
n = 4
M = np.logical_and.reduce(np.mod(X, 1) == 0, axis=-1)
M &= (X.sum(axis=-1) == n)
print(X[M])
```

#### 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★)


```python
# Author: Jessica B. Hamrick

X = np.random.randn(100) # random 1D array
N = 1000 # number of bootstrap samples
idx = np.random.randint(0, X.size, (N, X.size))
means = X[idx].mean(axis=1)
confint = np.percentile(means, [2.5, 97.5])
print(confint)
```
