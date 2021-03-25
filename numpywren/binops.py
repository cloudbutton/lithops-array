import numpy as np
from .matrix import BigMatrix
from .matrix_utils import chunk, generate_key_name_binop
from . import matrix_utils
import concurrent.futures as fs
import os
import lithops

import time
from . import lambdapack as lp
from . import job_runner


def _gemm_remote_0(block_pairs, XY, X, Y, reduce_idxs=[0], dtype=np.float64, **kwargs):
    print('block_pairs: ', block_pairs)
    print('reduce_idxs: ', reduce_idxs)

    for bp in block_pairs:
        bidx_0, bidx_1 = bp
        XY_block = None
        X.dtype = dtype
        Y.dtype = dtype

        for r in reduce_idxs:
            block1 = X.get_block(bidx_0, r)
            block2 = Y.get_block(r, bidx_1)
            if XY_block is None:
                XY_block = block1.dot(block2)
            else:
                XY_block += block1.dot(block2)

        XY.put_block(XY_block, bidx_0, bidx_1)


def _gemm_remote_1(block_pairs, XY, X, Y, reduce_idxs=[0], dtype=np.float64, **kwargs):
    os.system("sudo mount -o remount,size=50g /dev/shm")
    X.dtype = dtype
    Y.dtype = dtype
    for bp in block_pairs:
        bidx_0, bidx_1 = bp
        block0 = matrix_utils.get_row(X, bidx_0, mmap_loc="/dev/shm/block_0")
        block1 = matrix_utils.get_col(Y, bidx_1, mmap_loc="/dev/shm/block_1")
        XY_block = block0.dot(block1)
        XY.put_block(XY_block, bidx_0, bidx_1)


def _gemm_remote_2(block_pairs, XY, X, Y, reduce_idxs=[0], dtype=np.float64, **kwargs):
    os.system("sudo mount -o remount,size=50g /dev/shm")
    X.dtype = dtype
    X.dtype = dtype
    Y.dtype = dtype
    block_chunk_size = kwargs.get("block_chunk_size")
    for bp in block_pairs:
        bidx_0, bidx_1 = bp
        result = gemm_with_prefetch(X, Y, bidx_0, bidx_1, block_chunk_size=block_chunk_size)
        XY.put_block(result, bidx_0, bidx_1)


_gemms = [_gemm_remote_0, _gemm_remote_1, _gemm_remote_2]


def gemm_with_prefetch(X, Y, bidx0, bidx1, block_chunk_size=16):
    # prefetch first 16 columns
    parity = 0
    executor = fs.ProcessPoolExecutor(32)
    block_chunk_size = min(block_chunk_size, len(X._block_idxs(1)))
    chunked_blocks = list(matrix_utils.chunk(X._block_idxs(1), block_chunk_size))
    assert(chunked_blocks[0] == list(range(block_chunk_size)))
    futures0 = matrix_utils.get_matrix_blocks_full_async(X, "/dev/shm/block0_{0}".format(parity), [bidx0], list(range(block_chunk_size)), big_axis=1, executor=executor)
    futures1 = matrix_utils.get_matrix_blocks_full_async(Y, "/dev/shm/block1_{0}".format(parity), list(range(block_chunk_size)), [bidx1], big_axis=0, executor=executor)
    assert X._block_idxs(1) == Y._block_idxs(0)
    chunked_blocks = chunked_blocks[1:]
    start_x, end_x = X._blocks(0)[bidx0]
    start_y, end_y = Y._blocks(1)[bidx1]
    result = np.zeros((end_x - start_x, end_y - start_y), dtype=X.dtype)
    for blocks in chunked_blocks:
        t = time.time()
        fs.wait(futures0)
        fs.wait(futures1)
        e = time.time()
        print("Block Download took effectively {0}".format(e - t))
        results = [f.result() for f in futures0]
        b1 = matrix_utils.load_mmap(*results[0])
        results = [f.result() for f in futures1]
        b2 = matrix_utils.load_mmap(*results[0])
        parity = (parity + 1) % 2
        futures0 = matrix_utils.get_matrix_blocks_full_async(X, "/dev/shm/block0_{0}".format(parity), [bidx0], blocks, big_axis=1, executor=executor)
        futures1 = matrix_utils.get_matrix_blocks_full_async(Y, "/dev/shm/block1_{0}".format(parity), blocks, [bidx1], big_axis=0, executor=executor)
        t = time.time()
        result += b1.dot(b2)
        e = time.time()
        print("Block Matmul took effectively {0}".format(e - t))
    t = time.time()
    fs.wait(futures0)
    fs.wait(futures1)
    e = time.time()
    print("Block Download took effectively {0}".format(e - t))
    results = [f.result() for f in futures0]
    b1 = matrix_utils.load_mmap(*results[0])
    results = [f.result() for f in futures1]
    b2 = matrix_utils.load_mmap(*results[0])
    t = time.time()
    result += b1.dot(b2)
    e = time.time()
    print("Block Matmul took effectively {0}".format(e - t))
    return result


def gemm(fexec, X, Y, out_bucket=None, tasks_per_job=1, local=False,
         dtype=np.float64, overwrite=True, gemm_impl=0, gemm_chunk_size=16):
    '''
        Compute XY return
        @param pwex - Execution context
        @param X - rhs matrix
        @param Y - lhs matrix
        @param tasks_per_job - number of tasks per job
        @param out_bucket - bucket job writes to
        @param num_jobs - how many lambdas to run
        @param local - run locally? #TODO remove once local lithops executor is provided
    '''
    reduce_idxs = Y._block_idxs(axis=0)
    if out_bucket is None:
        out_bucket = X.bucket

    root_key = generate_key_name_binop(X, Y, "gemm")
    if (Y.shard_sizes[0] != X.shard_sizes[1]):
        raise Exception("X dim 1 shard size must match Y dim 0 shard size")

    XY = BigMatrix(root_key, shape=(X.shape[0], Y.shape[1]),
                   bucket=out_bucket, shard_sizes=[X.shard_sizes[0], Y.shard_sizes[1]],
                   dtype=dtype, write_header=True, storage=X.storage)

    num_out_blocks = len(XY.blocks)
    if (tasks_per_job > num_out_blocks):
        tasks_per_job = 1
    num_jobs = int(num_out_blocks/float(tasks_per_job))

    print("Out Shape", XY.shape)
    print("Total number of output blocks", len(XY.block_idxs))
    print("Total number of output blocks that exist", len(XY.blocks_exist))

    if (overwrite):
        block_idxs_to_map = list(set(XY.block_idxs))
    else:
        block_idxs_to_map = list(set(XY.block_idxs_not_exist))

    print("block_idxs_to_map: ", block_idxs_to_map)
    print("Number of output blocks to generate ", len(block_idxs_to_map))

    print("Tasks per job: ", tasks_per_job)
    print("Num Jobs: ", num_jobs)

    print('GEMM impl: ', gemm_impl, _gemms[gemm_impl])

    chunked_blocks = list(chunk(block_idxs_to_map, tasks_per_job))
    chunked_blocks = [(cb, ) for cb in chunked_blocks]

    #if (not isinstance(fexec.invoker, fexec.queues.SQSInvoker) and gemm_impl > 0):
    #    raise Exception("GEMM IMPL > 0 only supported for standalone mode pywren")

    # Josep: Storage class is not pickable, so delete it before invoke Lithops
    saved_stroage = X.storage
    XY.storage = Y.storage = X.storage = None

    def lithops_run(block_pairs, storage):
        XY.storage = storage
        X.storage = storage
        Y.storage = storage
        return _gemms[gemm_impl](block_pairs, XY, X, Y, reduce_idxs=reduce_idxs,
                                 dtype=dtype, block_chunk_size=gemm_chunk_size)

    if (local):
        list(map(lithops_run, chunked_blocks))
        return XY
    else:
        fexec.map(lithops_run, chunked_blocks, include_modules=['numpywren'])
        fexec.wait()

    Y.storage = X.storage = saved_stroage

    return XY


# matrix vector multiply
# hard
def gemv(pwex, X, Y, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError


# symmetric rank k update
# hard
def syrk(pwex, X, Y, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError


# very hard
def posv(pwex, X, Y, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError


# easy
def add(pwex, X, Y, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError


# easy
def sub(pwex, X, Y, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError


# easy
def mul(pwex, X, Y, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError


# easy
def div(pwex, X, Y, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError


def logical_and(pwex, X, Y, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError


def logical_or(pwex, X, Y, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError


def xor(pwex, X, Y, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError


def elemwise_binop_func(pwex, X, Y, f, out_bucket=None, tasks_per_job=1, local=False):
    raise NotImplementedError


def trisolve(pwex, A, B, out_bucket=None, tasks_per_job=1, lower=False):
    if out_bucket is None:
        out_bucket = A.bucket

    root_key = generate_key_name_binop(A, B, "trisolve")
    instructions, X, scratch = lp._trisolve(A, B, out_bucket=out_bucket, lower=lower)
    config = pwex.config
    # if (isinstance(pwex.invoker, pywren.queues.SQSInvoker)):
    #     executor = pywren.standalone_executor
    # else:
    fexec = lithops.FunctionExecutor()
    program = lp.LambdaPackProgram(instructions, executor=fexec, pywren_config=config)
    print(program)
    #assert False
    program.start()
    job_runner.lambdapack_run(program)
    program.wait()
    if program.program_status() != lp.PS.SUCCESS:
        program.unwind()
        raise Exception("Lambdapack Exception : {0}".format(program.program_status()))
    program.free()

    # delete all intermediate information
    [M.free() for M in scratch] 
    return X
