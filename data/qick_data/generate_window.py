import numpy as np

LO = 0
HI = 770

SL = 25

X_train = np.load('{}_{}/X_train_{}_{}.npy'.format(LO, HI, LO, HI))
y_train = np.load('{}_{}/y_train_{}_{}.npy'.format(LO, HI, LO, HI))
X_test = np.load('{}_{}/X_test_{}_{}.npy'.format(LO, HI, LO, HI))
y_test = np.load('{}_{}/y_test_{}_{}.npy'.format(LO, HI, LO, HI))

assert(X_train.shape[1] == HI*2)
assert(X_test.shape[1] == HI*2)

print('X_train', X_train.shape)
print('y_train', y_train.shape)

X_train_slice_lo = SL*2
X_train_slice_hi = X_train.shape[1] - SL*2
X_train_sliced = X_train[:,X_train_slice_lo:X_train_slice_hi]

X_test_slice_lo = SL*2
X_test_slice_hi = X_test.shape[1] - SL*2
X_test_sliced = X_test[:,X_test_slice_lo:X_test_slice_hi]

print('X_train (sliced)', X_train_sliced.shape)
print('y_train (sliced)', X_test_sliced.shape)

np.save('{}_{}/X_train_{}_{}.npy'.format(LO+SL, HI-SL, LO+SL, HI-SL), X_train_sliced)
np.save('{}_{}/y_train_{}_{}.npy'.format(LO+SL, HI-SL, LO+SL, HI-SL), y_train)

np.save('{}_{}/X_test_{}_{}.npy'.format(LO+SL, HI-SL, LO+SL, HI-SL), X_test_sliced)
np.save('{}_{}/y_test_{}_{}.npy'.format(LO+SL, HI-SL, LO+SL, HI-SL), y_test)

