from flask import (
    Flask, redirect, request, url_for
)
# import loguru

app = Flask(__name__)

@app.route('/register_request')
def register():
    registration_data = request.form
    email = registration_data['email']
    password = registration_data['password']
    confirm_password = registration_data['confirm_password']
    print(email, password, confirm_password)
    return
