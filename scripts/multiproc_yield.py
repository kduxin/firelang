
__all__ = ['yielder']

import time
import heapq
from multiprocessing import Process
from multiprocessing.queues import Empty
from faster_fifo import Queue

FOREVER = 1e10

class EndOfQueue:
    pass

def wrapped_put(que, item, timeout=FOREVER):
    for _ in range(max(1, int(timeout))):
        try:
            que.put(item, timeout=1)
        except:
            time.sleep(1)
            continue
        break

def wrapped_get(que, timeout=FOREVER):
    for _ in range(max(1, int(timeout))):
        try:
            item = que.get(timeout=1)
        except:
            time.sleep(1)
            continue
        break
    return item

def master_queue(generator, taskque: Queue, num_workers):
    for i, task in enumerate(generator):
        wrapped_put(taskque, (i, task))
    for i in range(num_workers):
        wrapped_put(taskque, EndOfQueue)

def slave_queue(func, taskque: Queue, resque: Queue, additional_kwds={}):
    stop = False
    while not stop:
        task = wrapped_get(taskque)
        if task is EndOfQueue:
            stop = True
            res = EndOfQueue
        else:
            jobid, task = task
            res = (jobid, func(task, **additional_kwds))
        wrapped_put(resque, res)

def yielder(generator, func, num_workers, additional_kwds={},
            verbose=True, print_interval:int=10000,
            max_size_bytes=1024*1024):
    taskque = Queue(max_size_bytes=max_size_bytes)
    resque = Queue(max_size_bytes=max_size_bytes)

    buff.clear()
    _master = Process(target=master_queue, args=(generator, taskque, num_workers))
    _master.start()
    for _ in range(num_workers):
        _slave = Process(target=slave_queue, args=(func, taskque, resque, additional_kwds))
        _slave.start()
    
    for i, x in enumerate(ordered_results(resque, num_workers)):
        yield x

        if verbose and i % print_interval == 0:
            print(f'{i}: taskque: {taskque.qsize()}, resque: {resque.qsize()}. heap buffsize: {len(buff)}')
    return StopIteration

buff = []   # for results to be outputed later
def ordered_results(resque: Queue, num_workers):
    ''' Online sort of the outputs in queue, according to their job id '''
    n = num_workers
    pos = 0
    while n > 0:  # run until any active process exists
        res = wrapped_get(resque)
        if res is EndOfQueue:
            n -= 1
        else:
            i, x = res
            heapq.heappush(buff, (i, x))
            while len(buff) and buff[0][0] == pos:
                i, x = heapq.heappop(buff)
                yield x
                pos += 1
    return StopIteration


if __name__ == '__main__':
    from time import time
    from tqdm import tqdm

    def job(x, margin):
        return max(x**3, margin)

    t0 = time()
    num_workers = 16
    generator = range(1000000)
    for x in tqdm(
        yielder(generator, func=job, num_workers=num_workers, 
                additional_kwds={'margin': 3}, max_size_bytes=1024*1024)):
        pass
    print(f'Elapsed: {time() - t0}')