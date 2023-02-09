import numpy as np


def evaluate_hls(hls_model, test_data):
    correct = 0
    total = 0

    for idx in range(1000):
        data, target = test_data[idx]

        data = np.ascontiguousarray(data.numpy())

        states = hls_model.predict(data)
        target = target.numpy()
        pred = np.argmax(states).astype(np.int32)

        if pred - target == 0:
            correct += 1
        total += 1
    
    return correct/total
