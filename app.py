import random
from twilio.rest import Client
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from sentiment_analysis.modeling import SentimentClassifier
from sentiment_analysis.dataloader import tokenizer, MAX_LEN
import torch
from flask_mail import Mail, Message
import secrets
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = ''  # Replace with your secret key
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Twilio configuration
TWILIO_ACCOUNT_SID = ''
TWILIO_AUTH_TOKEN = ''
TWILIO_PHONE_NUMBER = ''
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

current_model_path='sentiment_analysis/best_model_state1.bin'

# Initialize the model without loading state_dict
model = SentimentClassifier(5)
model = model.to(device)
model.eval()

# Function to load model state with error handling
def load_model_state(model_path=None):
    global current_model_path
    if model_path is None:
        model_path = current_model_path
    else:
        current_model_path = model_path
    try:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model state: {str(e)}")

# Call the function to load model state
load_model_state()

# Function to create a database connection
def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=('GET', 'POST'))
def register():
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'send_otp':
            username = request.form['username']
            email = request.form['email']
            password = request.form['password']
            phone = request.form['phone']  # Add this line to get the phone number
            hashed_password = generate_password_hash(password)

            conn = get_db_connection()
            existing_user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
            conn.close()

            if existing_user:
                return jsonify({'success': False, 'message': 'Email address already registered'})

            otp = ''.join([str(random.randint(0,9)) for _ in range(6)])

            session['registration_data'] = {
                'username': username,
                'email': email,
                'password': hashed_password,
                'phone': phone,
                'otp': otp
            }

            try:
                message = twilio_client.messages.create(
                    body=f'Your OTP for registration is: {otp}',
                    from_=TWILIO_PHONE_NUMBER,
                    to=phone
                )
                return jsonify({'success': True, 'message': 'OTP sent successfully'})
            except Exception as e:
                return jsonify({'success': False, 'message': f'Error sending OTP: {str(e)}'})

        elif action == 'verify_otp':
            user_otp = request.form['otp']
            if 'registration_data' not in session:
                return jsonify({'success': False, 'message': 'Registration data not found. Please start over.'})

            if user_otp == session['registration_data']['otp']:
                # OTP is correct, proceed with registration
                conn = get_db_connection()
                try:
                    conn.execute('INSERT INTO users (username, email, password, phone) VALUES (?, ?, ?, ?)',
                                 (session['registration_data']['username'],
                                  session['registration_data']['email'],
                                  session['registration_data']['password'],
                                  session['registration_data']['phone']))
                    conn.commit()
                except sqlite3.IntegrityError:
                    return jsonify({'success': False, 'message': 'Email address already registered'})
                finally:
                    conn.close()

                # Clear the registration data from session
                session.pop('registration_data')

                return jsonify({'success': True, 'message': 'Registration successful! Please log in.'})
            else:
                return jsonify({'success': False, 'message': 'Incorrect OTP. Please try again.'})

    return render_template('register.html')

@app.route('/login', methods=('GET', 'POST'))
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()

        if user is None:
            flash('No account found with that email address', 'error')
            return redirect(url_for('login'))
        elif not check_password_hash(user['password'], password):
            flash('Incorrect password, try again.', 'error')
            return redirect(url_for('login'))

        session.clear()
        session['user_id'] = user['id']
        session['user_name']=user['username']
        return redirect(url_for('user_dashboard'))

    return render_template('login.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()

        if user:
            token = secrets.token_urlsafe(32)
            conn = get_db_connection()
            conn.execute('UPDATE users SET reset_token = ? WHERE id = ?', (token, user['id']))
            conn.commit()
            conn.close()

            reset_url = url_for('reset_password', token=token, _external=True)
            msg = Message('Password Reset Request',
                          sender='noreply@yourdomain.com',
                          recipients=[email])
            #msg.body = f'To reset your password, visit the following link: {reset_url}'mail.send(msg)

            flash('A password reset link has been sent to your email.', 'info')
            return redirect(url_for('login'))
        else:
            flash('Email not found.', 'danger')
    return render_template('forgot_password.html')

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    if request.method == 'POST':
        new_password = request.form['password']
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE reset_token = ?', (token,)).fetchone()
        if user:
            hashed_password = generate_password_hash(new_password)
            conn.execute('UPDATE users SET password = ?, reset_token = NULL WHERE id = ?', (hashed_password, user['id']))
            conn.commit()
            flash('Your password has been updated. Please log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Invalid or expired reset token.', 'danger')
        conn.close()
    return render_template('reset_password.html', token=token)

@app.route('/admin_login', methods=('GET', 'POST'))
def admin_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        admin = conn.execute('SELECT * FROM admins WHERE email = ?', (email,)).fetchone()
        conn.close()

        if admin is None:
            flash('No admin account found with that email address', 'error')
            return redirect(url_for('admin_login'))
        elif not check_password_hash(admin['password'], password):
            flash('Incorrect password, try again.', 'error')
            return redirect(url_for('admin_login'))

        session.clear()
        session['admin_id'] = admin['id']
        session['password'] = admin['password']
        return redirect(url_for('admin_dashboard'))

    return render_template('admin_login.html')

@app.route('/admin_dashboard', methods=['GET', 'POST'])
def admin_dashboard():
    if 'admin_id' not in session:
        return redirect(url_for('admin_login'))

    conn = get_db_connection()

    # Fetch current admin details
    current_admin = conn.execute('SELECT * FROM admins WHERE id = ?', (session['admin_id'],)).fetchone()

    if request.method == 'POST':
        data = request.get_json()
        action = data.get('action')

        if action == 'add_user':
            username = data.get('username')
            email = data.get('email')
            password = data.get('password')

            if not username or not email or not password:
                return jsonify({'success': False, 'message': 'All fields are required'}), 400

            hashed_password = generate_password_hash(password)
            try:
                conn.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                             (username, email, hashed_password))
                conn.commit()
                return jsonify({'success': True}), 201
            except sqlite3.IntegrityError:
                return jsonify({'success': False, 'message': 'Email address already registered'}), 400

        elif action == 'modify_user':
            user_id = data.get('user_id')
            username = data.get('username')
            email = data.get('email')
            password = data.get('password')
            admin_password = data.get('admin_password')

            if not check_password_hash(session['password'], admin_password):
                return jsonify({'success': False, 'message': 'Incorrect admin password'}), 403

            if password:
                hashed_password = generate_password_hash(password)
                conn.execute('UPDATE users SET username = ?, email = ?, password = ? WHERE id = ?',
                             (username, email, hashed_password, user_id))
            else:
                conn.execute('UPDATE users SET username = ?, email = ? WHERE id = ?',
                             (username, email, user_id))
            conn.commit()
            return jsonify({'success': True}), 200

        elif action == 'remove_user':
            user_id = data.get('user_id')
            conn.execute('DELETE FROM users WHERE id = ?', (user_id,))
            conn.commit()
            return jsonify({'success': True}), 200

        return jsonify({'success': False, 'message': 'Invalid action'}), 400

    users = conn.execute('SELECT * FROM users').fetchall()
    conn.close()

    # Convert current_admin to a dictionary
    current_admin_dict = dict(current_admin) if current_admin else {}

    return render_template('admin_dashboard.html', users=users, current_admin=current_admin_dict)

@app.route('/user_dashboard', methods=['GET', 'POST'])
def user_dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    conn = get_db_connection()
    user=conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()

    current_time=datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    conn.execute('UPDATE users SET last_login = ? WHERE id = ?', (current_time, session['user_id']))
    conn.commit()

    user_details={
        'name':user['username'],
        'email':user['email'],
        'last_login':user['last_login'] if user['last_login'] else 'First login'
    }
    if request.method == 'POST':
        review_text = request.form['review_text']
        model_selection=request.form['model_selection']

        #Here we're loading the selected  model
        if model_selection == 'best_model_state1.bin':
            load_model_state('sentiment_analysis/best_model_state1.bin')
            model_used='best_model_state1.bin'
        elif model_selection == 'best_model_state.bin':
            load_model_state('sentiment_analysis/best_model_state.bin')
            model_used='best_model_state.bin'
        else:
            model_used='unknown'


        encoded_review = tokenizer.encode_plus(
            review_text,
            max_length=MAX_LEN,
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded_review['input_ids'].to(device)
        attention_mask = encoded_review['attention_mask'].to(device)
        output = model(input_ids, attention_mask)
        _, prediction = torch.max(output, dim=1)

        sentiment = int(prediction.item() + 1)
        confidence_score = torch.nn.functional.softmax(output, dim=1)[0][prediction].item()
        sentiment_class = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"][sentiment - 1]

        # In case of any error remove this part
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
        conn.execute('INSERT INTO user_logs (user_name, email_id, timestamp, text, confidence_score, sentiment_class,model_used) VALUES (?, ?, ?, ?, ?, ?, ?)',
                     (user['username'], user['email'], datetime.now(), review_text, confidence_score, sentiment_class,model_used))
        conn.commit()
        conn.close()
        ###########################################
        return jsonify({
            'sentiment': sentiment,
            'sentiment_class': sentiment_class,
            'confidence_score': confidence_score,
            'model_used': model_used
        })
    conn.close()
    return render_template('user_dashboard.html',user_details=user_details)

@app.route('/get_user_logs')
def get_user_logs():
    if 'user_id' not in session:
        return jsonify([])

    conn = get_db_connection()

    # Fetch all logs from the database, ordered by timestamp
    logs = conn.execute('SELECT * FROM user_logs ORDER BY timestamp DESC').fetchall()

    conn.close()

    log_list = [{
        'timestamp': log['timestamp'],
        'text': log['text'],
        'sentiment_class': log['sentiment_class'],
        'confidence_score': log['confidence_score'],
        'model_used': log['model_used'],
        'user_name': log['user_name']  # Include the username in the output
    } for log in logs]

    print(f"Number of Logs fetched from database: {len(log_list)}")
    return jsonify(log_list)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)

