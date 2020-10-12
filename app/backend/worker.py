import os

import redis
from loguru import logger
from rq import Worker, Queue, Connection

listen = ['high', 'default', 'low']

redis_url = 'redis://localhost:6379' # os.getenv('REDIS_URL')
logger.info('REDIS URL {}'.format(redis_url))
logger.info('In directory {}'.format(os.getcwd()))

conn = redis.from_url(redis_url)

if __name__ == '__main__':
  logger.info('__main__')
  with Connection(conn):
    worker = Worker(map(Queue, listen))
    logger.info(f'worker: {worker}')
    worker.work()