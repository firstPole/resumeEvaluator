import logging
import os
import uuid
import mimetypes
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory, flash
from dotenv import load_dotenv
import braintree
from flask_mail import Mail, Message
from text_processing import *
from sentiment_analysis import *
import time 

# Load environment variables from .env file
# load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Use environment variables for Braintree and mail configurations
app.config['merchant_id'] = os.getenv('merchant_id')
app.config['public_key'] = os.getenv('public_key')
app.config['private_key'] = os.getenv('private_key')

mail = Mail(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Replace with your Braintree environment details
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
    return redirect(url_for('index'))

@app.route('/index')
def index():
    # Removed login check and user context
    user = None
    return render_template('index.html', user=user)

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

@app.route('/dropin')
def dropin():
    try:
        client_token = gateway.client_token.generate()
        return render_template('payment.html', client_token=client_token)
    except Exception as e:
        logging.error(f"Error generating client token: {e}")
        return render_template('error.html', message='Unable to generate payment token.')

@app.route('/process_payment', methods=['POST'])
def process_payment():
    try:
        data = request.get_json()
        nonce = data['payment_method_nonce']
        amount = data['amount']

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
        logging.error(f"Error processing payment: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/thank_you')
def thank_you():
    return render_template('thank_you.html')

@app.route('/evaluate', methods=['POST'])
def evaluate_resume():
    try:
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

        if not job_description.strip():
            logging.error('Invalid job description content.')
            return jsonify({'error': 'Invalid job description content.'}), 400

        sentiment_analysis = perform_sentiment_analysis(resume_text)
        match_score = calculate_matching_score(resume_text, job_description)
        match_score =  round(match_score, 2)

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
    except Exception as e:
        logging.error(f"Error evaluating resume: {e}")
        return jsonify({'error': 'An error occurred while evaluating the resume.'}), 500


@app.route('/feedback', methods=['POST'])
def feedback():
    feedback_category = request.form.get('feedback_category')
    feedback_text = request.form.get('feedback_text')
    feedback_email = request.form.get('feedback_email')

    if 'google_token' not in session:
        return redirect(url_for('login'))

    try:
        id_token = session['google_token'].get('id_token')
        if id_token:
            user_info = oauth.google.parse_id_token(id_token)
            if user_info:
                user_name = user_info.get('name')
                user_email = user_info.get('email')
            else:
                user_name = 'Anonymous'
                user_email = feedback_email
        else:
            user_name = 'Anonymous'
            user_email = feedback_email

        msg = Message('User Feedback', sender='noreply@example.com', recipients=['feedback@example.com'])
        msg.body = f'Category: {feedback_category}\n\nFeedback: {feedback_text}\n\nUser: {user_name}\nEmail: {user_email}'
        mail.send(msg)

        flash('Thank you for your feedback!', 'success')
        return redirect(url_for('index'))
    except Exception as e:
        logging.error(f"Error sending feedback: {e}")
        flash('An error occurred while sending your feedback. Please try again later.', 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    # Set debug=False for production
    #app.run(debug=False, host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
   app.run(debug=True)
