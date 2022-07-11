from multiprocessing import Pool, Queue, Process, cpu_count

class UptownFunc:
    '''
    Source: https://www.benmather.info/post/2018-11-24-multiprocessing-in-python/
    '''

    def __init__(self):
        pass

    def _func_queue(self, func, q_in, q_out, *args, **kwargs):
        """ Retrive processes from the queue """
        while True:
            pos, var = q_in.get()
            if pos is None:
                break

            res = func(var, *args, **kwargs)
            q_out.put((pos, res))
        return

    def parallelise_function(self, var, func, *args, **kwargs):
        """ Split evaluations of func across processors """
        n = len(var)

        processes = []
        q_in = Queue(1)
        q_out = Queue()

        nprocs = cpu_count()

        for i in range(nprocs):
            pass_args = [func, q_in, q_out]
            # pass_args.extend(args)

            p = Process(target=self._func_queue,\
                        args=tuple(pass_args),\
                        kwargs=kwargs)

            processes.append(p)

        for p in processes:
            p.daemon = True
            p.start()

        # put items in the queue
        sent = [q_in.put((i, var[i])) for i in range(n)]
        [q_in.put((None, None)) for _ in range(nprocs)]

        # get the results
        results = [[] for i in range(n)]
        for i in range(len(sent)):
            index, res = q_out.get()
            results[index] = res

        # wait until each processor has finished
        [p.join() for p in processes]

        # reorder results
        return results
