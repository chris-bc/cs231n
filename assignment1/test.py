import numpy as np

dists=np.zeros((5,2))
train=np.zeros((2,3))
test=np.zeros((5,3))

train[0,0]=1
train[0,1]=2
train[0,2]=3
train[1,0]=4
train[1,1]=5
train[1,2]=6
test[0,0]=6
test[0,1]=9
test[0,2]=12
test[1,0]=4
test[1,1]=5
test[1,2]=6
test[2,0]=7
test[2,1]=8
test[2,2]=9
test[3,0]=10
test[3,1]=11
test[3,2]=12
print train[0]
print test[0]
print np.sqrt(np.sum((train[0]-test[0])**2))
print np.sqrt(np.sum(np.square(train[0]-test[0]), axis=0))
print np.sqrt(np.sum(np.square(train[0]-test[0,:]),axis=0))
print train[0,:]
print train[0]
print np.sqrt(np.sum(np.square(train[0,:]-test[0,:])))

a = np.array([0,3,2,3,1,5,0,3,7,0,0,7,7,7,7,7,7,7,7,7,0,0,0,0,0,0])
unique,counts = np.unique(a, return_counts=True)
print np.argmax(counts)
print unique

b=np.argsort(a)
print b

print train.shape
print train.shape[0]
print len(train)
