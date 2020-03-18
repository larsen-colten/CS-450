from nn import fun as nn
from svm import fun as svm
from nb import fun  as nb
from dt import fun as dt
from rf import fun as rf
from gbm import fun as gbm

x, y, z = nn()
print(x)
print(y)
print(z)
x, y, z = svm()
print(x)
print(y)
print(z)
x, y, z = nb()
print(x)
print(y)
print(z)
x, y, z = dt()
print(x)
print(y)
print(z)
x, y, z = rf()
print(x)
print(y)
print(z)
x, y, z = gbm()
print(x)
print(y)
print(z)

