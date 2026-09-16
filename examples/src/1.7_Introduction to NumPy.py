import numpy as np
np.zeros(10)

#Fill 10 zeros
print(np.zeros(10))
#Fill 10 ones
print(np.ones(10))
#Fill 10 specific values (2.5)
print(np.full(10, 2.5))

#Create array from a Python list
a = np.array([1, 2, 3, 5, 7, 12])
print(a)
print(a[2])

#Create array of range from (start, end exclusive)
b = np.arange(3, 10)
print(b)

b1 = np.array(range(3,10))
print(b1)

#Create evenly spaced numbers over a specified interval (start, end, intervals)
c = np.linspace(0, 100, 9)
print(c)

#Two dimensional array (row, column)
d =np.zeros((5,2))
print(d)

d1 = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(d1)
print(d1[1, 2]) #Accessing element at row 1, column 2
d1[1, 2] = 10 #Modifying element at row 1, column 2
print(d1)
print(d1[0]) #Accessing row 0

d1[2] = [1, 1, 1] #Modifying row 2
print(d1)

d1[:, 1] = [0, 0, 0] #Modifying column 1
print(d1)

#Random numbers
np.random.seed(2) #Setting seed for reproducibility
e = np.random.rand(5, 2) # 5x2 array of random numbers between 0 and 1
print(e)

e1 = 100 * e # random numbers between 0 and 100
print(e1)

e2 = np.random.randn(5, 2) # normal distribution (mean=0, std=1)
print(e2)

# random integers between 0 and 100, array size 5x2
e3 = np.random.randint(0, 100, (5, 2))
print(e3)

#element-wise operations
f = np.arange(1,6)
print(f)
f1 = f + 1
print(f1)
f2 = f * 2
print(f2)
f3 = (10 + (f * 2) ** 2) / 100
print(f3)
f4 = f + f3
print(f4)

#Comparison operations
g = a >= 2
print(g)
g1 = f > f3
print(g1)
g2 = f[f > f3] #use boolean condition for filtering
print(g2)

#Summarizing operations
h = a.min()
print(h)
h1 = a.std()
print(h1)