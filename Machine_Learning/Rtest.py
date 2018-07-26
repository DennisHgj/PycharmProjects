from numpy import random

a = random.randint(0, 2, size=(100, 100))
print(a)
file=open("C://work//data2.txt",'w+')
file.write(str(a))
file.close()

