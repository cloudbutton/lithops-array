from numpywren.matrix import BigMatrix
from numpywren import binops
from numpywren.matrix_init import shard_matrix
import numpy as np
import lithops
import unittest
import os


class GemmTestClass(unittest.TestCase):
    def test_single_shard_matrix_multiply(self):
        fexec = lithops.FunctionExecutor(runtime='jsampe/numpy-lithops:04', log_level='DEBUG')

        X = np.random.randn(16, 16)
        X_sharded = BigMatrix("gemm_test_0", shape=X.shape, shard_sizes=X.shape, storage=fexec.storage)
        shard_matrix(X_sharded, X)

        XX_sharded = binops.gemm(fexec, X_sharded, X_sharded.T, X_sharded.bucket, 1)

        XX_sharded_local = XX_sharded.numpy()
        XX = X.dot(X.T)
        X_sharded.free()
        XX_sharded.free()

        assert(np.all(np.isclose(XX, XX_sharded_local)))
        os.system("rm -rf /dev/shm/*")

    def test_multiple_shard_matrix_multiply(self):
        fexec = lithops.FunctionExecutor(runtime='jsampe/numpy-lithops:04', log_level='DEBUG')

        X = np.random.randn(16, 16)
        X_shard_sizes = tuple(map(int, np.array(X.shape)/2))
        X_sharded = BigMatrix("gemm_test_1", shape=X.shape, shard_sizes=X_shard_sizes, storage=fexec.storage)

        Y = np.random.randn(16, 16)
        Y_shard_sizes = tuple(map(int, np.array(Y.shape)/2))
        Y_sharded = BigMatrix("gemm_test_2", shape=Y.shape, shard_sizes=Y_shard_sizes, storage=fexec.storage)

        shard_matrix(X_sharded, X)
        shard_matrix(Y_sharded, Y)

        XY_sharded = binops.gemm(fexec, X_sharded, Y_sharded, X_sharded.bucket, 1)

        XY_sharded_local = XY_sharded.numpy()
        XY = X.dot(Y)
        X_sharded.free()
        Y_sharded.free()
        XY_sharded.free()
        assert(np.all(np.isclose(XY, XY_sharded_local)))
        os.system("rm -rf /dev/shm/*")


if __name__ == "__main__":
    tests = GemmTestClass()
    tests.test_single_shard_matrix_multiply()
    tests.test_multiple_shard_matrix_multiply()
