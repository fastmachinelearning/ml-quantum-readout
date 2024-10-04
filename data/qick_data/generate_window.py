import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--in_lo', type=int, default=0, help='Input low range (default: 0)')
parser.add_argument('--in_hi', type=int, default=770, help='Input high range (default: 770)')
parser.add_argument('--ou_lo', type=int, required=True, help='Output low range')
parser.add_argument('--ou_hi', type=int, required=True, help='Output high range')

args = parser.parse_args()

IN_LO = args.in_lo
IN_HI = args.in_hi

OU_LO = args.ou_lo
OU_HI = args.ou_hi

print('{:03d}_{:03d} --> {:03d}_{:03d}'.format(IN_LO, IN_HI, OU_LO, OU_HI))

X_train = np.load('{:03d}_{:03d}/X_train_{:03d}_{:03d}.npy'.format(IN_LO, IN_HI, IN_LO, IN_HI))
y_train = np.load('{:03d}_{:03d}/y_train_{:03d}_{:03d}.npy'.format(IN_LO, IN_HI, IN_LO, IN_HI))
X_test = np.load('{:03d}_{:03d}/X_test_{:03d}_{:03d}.npy'.format(IN_LO, IN_HI, IN_LO, IN_HI))
y_test = np.load('{:03d}_{:03d}/y_test_{:03d}_{:03d}.npy'.format(IN_LO, IN_HI, IN_LO, IN_HI))

assert(X_train.shape[1] == IN_HI*2)
assert(X_test.shape[1] == IN_HI*2)

print('X_train', X_train.shape)
print('y_train', y_train.shape)
print('X_test', X_test.shape)
print('y_test', y_test.shape)


X_train_slice_lo = OU_LO*2
X_train_slice_hi = OU_HI*2
X_train_sliced = X_train[:,X_train_slice_lo:X_train_slice_hi]

X_test_slice_lo = OU_LO*2
X_test_slice_hi = OU_HI*2
X_test_sliced = X_test[:,X_test_slice_lo:X_test_slice_hi]

print('X_train (sliced)', X_train_sliced.shape)
print('X_test (sliced)', X_test_sliced.shape)

os.makedirs('{:03d}_{:03d}'.format(OU_LO, OU_HI), exist_ok=True)

np.save('{:03d}_{:03d}/X_train_{:03d}_{:03d}.npy'.format(OU_LO, OU_HI, OU_LO, OU_HI), X_train_sliced)
np.save('{:03d}_{:03d}/y_train_{:03d}_{:03d}.npy'.format(OU_LO, OU_HI, OU_LO, OU_HI), y_train)

np.save('{:03d}_{:03d}/X_test_{:03d}_{:03d}.npy'.format(OU_LO, OU_HI, OU_LO, OU_HI), X_test_sliced)
np.save('{:03d}_{:03d}/y_test_{:03d}_{:03d}.npy'.format(OU_LO, OU_HI, OU_LO, OU_HI), y_test)

