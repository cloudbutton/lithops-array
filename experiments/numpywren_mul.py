# Get numpywren results for matrix multiplication:
import numpy as np
import time
import lithops
from numpywren.matrix import BigMatrix
from numpywren.binops import gemm


Ns = [5000, 10000, 15000, 20000, 25000, 30000]
shard_size = (5000, 5000)

np.random.seed(42)


if __name__ == "__main__":

    fexec = lithops.FunctionExecutor(runtime='jsampe/numpy-lithops:04', log_level='DEBUG')

    # Only run this if matrices not already in the bucket.
    # This takes a very long time (for 30000x30000xf64 - 8GB of data)
    Big_X = BigMatrix("multiply_test1", shape=(max(Ns), max(Ns)), shard_sizes=shard_size, storage=fexec.storage)
    for i in range():
        for j in range():
            X = np.random.randn(5000, 5000)
            Big_X.put_block(X, i, j)

    for N in Ns:
        X_sharded = BigMatrix("multiply_test1", shape=(N, N), shard_sizes=shard_size, storage=fexec.storage)
        start = time.time()
        gemm(fexec, X_sharded, X_sharded, X_sharded.bucket, 1)
        end = time.time()
        print(end - start)
