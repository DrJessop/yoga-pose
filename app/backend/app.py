from flask import (
    Flask, redirect, request, url_for
)
from flask_cors import CORS
from loguru import logger

from rq import Queue
from rq.job import Job
# q = Queue(connection=conn)

app = Flask(__name__)
CORS(app)  # Cross-origin resource sharing (React app using different port than Flask app)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    files = request.files
    instructor = files['instructor']
    student = files['student']

    logger.info(instructor)
    logger.info(student)

    return {
        'status': 200,
        'mimetype': 'application/json'
    }

if __name__ == '__main__':
    app.run()
