import os

from flask import (
    Flask, redirect, request, url_for, 
    json
)
from flask_cors import CORS
from loguru import logger

from rq import Queue
from rq.job import Job
from worker import conn

q = Queue(connection=conn)

app = Flask(__name__)
CORS(app)  # Cross-origin resource sharing (React app using different port than Flask app)


@app.route('/upload_video', methods=['POST'])
def upload_video():
    video            = request.files
    instructor       = video['instructor']
    student          = video['student']
    instructor_fname = instructor.filename
    student_fname    = student.filename

    logger.info('Received {} & {}'.format(instructor_fname, student_fname))
    logger.info('In directory {}'.format(os.getcwd()))

    with open('../../videos/input/' + instructor_fname, 'wb') as f:
        f.write(instructor.read())
    
    with open('../../videos/input/' + student_fname, 'wb') as f:
        f.write(student.read())

    one_week = 60 * 60 * 24 * 7

    q.enqueue(
        'util.pose_extraction.get_error',
        args=(instructor_fname, student_fname),
        job_timeout=one_week
    )
    
    return {
        'status': 200,
        'mimetype': 'application/json'
    }

if __name__ == '__main__':
    app.run()
