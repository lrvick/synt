# -*- coding: utf-8 -*-
"""Tools to deal with multi-processing."""
import multiprocessing

def batch_job(producer, consumer, chunksize=10000, processes=None):
    """
    Call consumer on everything that is produced from producer, using a pool.

    Args:
    producer    -- produces the events that are fed to the consumer.
    consumer    -- the function called with values recieved from the producer

    Keyword Arguments:
    chunksize   -- how many values to request from the producer
    processes   -- how many processes should be created to handle jobs
    """

    if type(producer) in [list,tuple]:
        #replace the list or tuple with a dummy producer function
        
        def tmp(offset, length):
            """
            A wraper for lists to allow them to be used as producers.
            """
            return producer[offset:offset+length]
        producer = tmp


    if not processes:
        processes = multiprocessing.cpu_count()
    
    offset = 0
    
    finished = False
    
    pool = multiprocessing.Pool(processes)
   
    while not finished:
        
        for i in range(1, processes + 1):
           
            samples = producer(offset, chunksize)
            
            if not samples:
                finished = True
                break
            
            pool.apply_async(consumer, [samples])
            

            offset += len(samples)
    
    pool.close() 
    pool.join() #wait for workers to finish

if __name__=="__main__":
    #example usage
    
    def producer(offset, length):
        if offset >= 50:
            return []
        return range(offset, offset + length)

    queue = multiprocessing.Queue()
    def consumer(data):
        global queue
        
        for i in data:
            queue.put(i)

    batch_job(producer, consumer, 10)

    out = []

    while not queue.empty():
        out.append(queue.get())
    print out
