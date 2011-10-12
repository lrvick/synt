# -*- coding: utf-8 -*-
"""Tools to deal with multi-processing."""
import multiprocessing
from synt.logger import create_logger

logger = create_logger(__file__)

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

        logger.info("producer was a list, wrapping it in temporary producer.")

    if not processes:
        logger.info("processes was 0 or none, using cpu count.")
        processes = multiprocessing.cpu_count()
    
    offset = 0
    
    finished = False
    
    pool = multiprocessing.Pool(processes)
   
    while not finished:
        
        for i in range(1, processes + 1):
            logger.info("producing %r %r",offset,chunksize)
           
            samples = producer(offset,chunksize)
            
            if not samples:
                finished = True
                logger.info("Producer returned False/empty list. Quitting.")
                break
            
            pool.apply_async(consumer, [samples])
            

            offset += len(samples)
    
    logger.info("Waiting for pools to clear.")
    pool.close() 
    pool.join() #wait for workers to finish
    logger.info("Last pool finished.")

if __name__=="__main__":
    import time
    def producer(offset, length):
        logger.info("Producing %r %r",offset,length)
        
        if offset >= 100:
            return []
        return range(offset,offset+length)

    queue = multiprocessing.Queue()
    def consumer(data):
        global queue
        logger.info("Consuming: %r",len(data))
        for i in data:
            queue.put(i)

    logger.info("starting.")
    start = time.time()
    batch_job(producer,consumer,10)
    logger.info("Finished: %r",time.time()-start)
    
    i = 0
    msg = ""
    while not queue.empty():
        i = i + 1 
        msg += "%r "%queue.get()
        if i == 10:
            logger.info("%r",msg)
            msg = ""
            i=0
