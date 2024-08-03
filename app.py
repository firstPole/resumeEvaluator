import logging
from text_processing import *
from sentiment_analysis import analyze_job_description, perform_sentiment_analysis
import os
import uuid
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory
from flask_oauthlib.client import OAuth
from dotenv import load_dotenv
import mimetypes
import braintree
# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Use environment variables for client ID and secret
app.config['GOOGLE_ID'] = os.getenv('GOOGLE_CLIENT_ID')
app.config['GOOGLE_SECRET'] = os.getenv('GOOGLE_CLIENT_SECRET')
app.config['merchant_id']= os.getenv('merchant_id')
app.config['public_key']= os.getenv('public_key')
app.config['private_key']= os.getenv('private_key')

oauth = OAuth(app)
google = oauth.remote_app(
    'google',
    consumer_key=app.config['GOOGLE_ID'],
    consumer_secret=app.config['GOOGLE_SECRET'],
    request_token_params={
        'scope': 'openid email profile',
    },
    base_url='https://www.googleapis.com/oauth2/v1/',
    request_token_url=None,
    access_token_url='https://accounts.google.com/o/oauth2/token',
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    access_token_method='POST',
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Replace with your Braintree environment details

import braintree

gateway = braintree.BraintreeGateway(
  braintree.Configuration(
      braintree.Environment.Sandbox,
      merchant_id=app.config['merchant_id'],
        public_key=app.config['public_key'],
        private_key=app.config['private_key']
  )
)




@app.route('/')
def home():
    return render_template('login.html')

@app.route('/index')
def index():
    if 'google_token' in session:
        me = google.get('userinfo')
        if me.data and 'name' in me.data:
            user_info = {
                'name': me.data["name"],
                'profile_pic': me.data.get("picture")  # Get the profile picture URL
            }
            return render_template('index.html', user=user_info)
        else:
            return render_template('index.html', user={'name': 'User', 'profile_pic': ''})
    return redirect(url_for('home'))

@app.route('/login')
def login():
    return google.authorize(callback=url_for('google_callback', _external=True))

@app.route('/logout')
def logout():
    session.pop('google_token', None)
    session.pop('user', None)
    return redirect(url_for('home'))

@app.route('/google_callback')
def google_callback():
    if 'error' in request.args:
        return f'Error: {request.args["error"]}'
    
    response = google.authorized_response()
    
    if response is None or response.get('access_token') is None:
        return 'Access denied: reason={} error={}'.format(
            request.args.get('error_reason', 'unknown'),
            request.args.get('error_description', 'unknown')
        )
    
    session['google_token'] = (response['access_token'], '')
    return redirect(url_for('index'))

@google.tokengetter
def get_google_oauth_token():
    return session.get('google_token')

def validate_file_type(file):
    mime_type, _ = mimetypes.guess_type(file.filename)
    logging.debug(f'File MIME type: {mime_type}')
    return mime_type in ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']

def extract_text(file):
    file_ext = file.filename.rsplit('.', 1)[1].lower()
    logging.debug(f'File extension: {file_ext}')
    if file_ext == 'pdf':
        return extract_text_from_pdf(file.stream)
    elif file_ext == 'docx':
        return extract_text_from_docx(file.stream)
    else:
        return None

# Paypal payment routines

@app.route('/dropin')
def dropin():
    client_token = gateway.client_token.generate()
    return render_template('payment.html', client_token=client_token)

@app.route('/process_payment', methods=['POST'])
def process_payment():
    try:
        data = request.get_json()
        nonce = data['payment_method_nonce']
        amount = data['amount']

        # Validate amount here if needed (e.g., check if it's a number, within a range, etc.)

        result = gateway.transaction.sale({
            "amount": str(amount),
            "payment_method_nonce": nonce,
            "options": {
                "submit_for_settlement": True
            }
        })
        
        if result.is_success:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': result.message})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/thank_you')
def thank_you():
    return render_template('thank_you.html')


@app.route('/evaluate', methods=['POST'])
def evaluate_resume():
    resume_file = request.files.get('resume_file')
    job_description = request.form.get('job_description')

    if not resume_file or not resume_file.filename:
        logging.error('No resume file provided.')
        return jsonify({'error': 'No resume file provided.'}), 400

    if not validate_file_type(resume_file):
        logging.error('Unsupported file type.')
        return jsonify({'error': 'Unsupported file type. Please upload a PDF or DOCX file.'}), 400

    resume_text = extract_text(resume_file)
    if resume_text is None:
        logging.error('Failed to extract text from resume file.')
        return jsonify({'error': 'Failed to extract text from resume file.'}), 400

    resume_text = preprocess_text(resume_text)
    job_description = preprocess_text(job_description)

    # # Ensure the job description is valid
    # if not any(keyword in job_description.lower() for keyword in ["responsibilities", "requirements", "qualifications"]):
    #     logging.error('Invalid job description content.')
    #     return jsonify({'error': 'Invalid job description content.'}), 400

    sentiment_analysis = perform_sentiment_analysis(resume_text)
    match_score = calculate_matching_score(resume_text, job_description)

    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description)

    missing_skills = job_skills - resume_skills
    matched_skills = job_skills & resume_skills

    missing_skills = filter_relevant_skills(matched_skills, list(missing_skills))
    job_analysis = analyze_job_description(job_description)

    keyword_density = []
    matching_tag_occurrences = count_tag_occurrences(job_description, list(matched_skills))
    non_matching_tag_occurrences = count_tag_occurrences(job_description, list(missing_skills))

    for keyword, count in matching_tag_occurrences.items():
        if count > 0:
            keyword_density.append({"keyword": keyword, "density": count})

    for keyword, count in non_matching_tag_occurrences.items():
        if count > 0:
            keyword_density.append({"keyword": keyword, "density": count})

    email = extract_email(job_description)
    phone_number = extract_phone_number(job_description)
    visa_info = extract_visa_info(job_description)
    
    
    session['unique_id'] = str(uuid.uuid4())
    session['resume_text'] = resume_text
    session['missing_skills'] = list(missing_skills)

    return jsonify({
        'match_score': match_score,
        'sentiment': sentiment_analysis,
        'relevant_tags': list(matched_skills),
        'non_matching_tags': list(missing_skills),
        'job_analysis': {
            'language_tone': job_analysis['language_tone'],
            'emphasis': job_analysis['emphasis_teamwork'],
            'social_responsibility': job_analysis['social_responsibility'],
            'keyword_density': keyword_density
        },
        'email': email,
        'phone_number': phone_number,
        'visa_info': visa_info
    })


@app.route('/generate-updated-resume', methods=['POST'])
def generate_updated_resume():
    try:
        selected_tags = request.json.get('tags', [])
        resume_text = session.get('resume_text', '')

        if not resume_text.strip():
            return jsonify({'error': 'No resume text found.'}), 400

        updated_resume_text = update_resume_with_tags_using_llm(resume_text, selected_tags)

        output_file_path = f"static/updated_resume_{session['unique_id']}.docx"
        save_text_to_docx(updated_resume_text, output_file_path)

        return jsonify({'file_path': output_file_path})
    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({'error': 'An error occurred while generating the updated resume.'}), 500

@app.route('/static/<path:filename>')
def download_file(filename):
    return send_from_directory('static', filename)




if __name__ == '__main__':
    app.run(debug=True)
