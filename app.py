from flask import Flask, request, render_template, jsonify,session,send_from_directory
from text_processing import count_tag_occurrences, update_resume_with_tags_using_llm ,save_text_to_docx,extract_text_from_docx, extract_text_from_pdf, preprocess_text, extract_skills, calculate_matching_score, filter_relevant_skills
from sentiment_analysis import analyze_job_description,perform_sentiment_analysis,get_keyword_density
import os
import secrets
import uuid

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY') or secrets.token_hex(16)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate_resume():
    resume_file = request.files.get('resume_file')
    job_description = request.form.get('job_description')
    
    if resume_file and resume_file.filename:
        file_ext = resume_file.filename.rsplit('.', 1)[1].lower()
        if file_ext == 'pdf':
            resume_text = extract_text_from_pdf(resume_file.stream)
        elif file_ext == 'docx':
            resume_text = extract_text_from_docx(resume_file.stream)
        else:
            return jsonify({'error': 'Unsupported file type. Please upload a PDF or DOCX file.'}), 400
    else:
        return jsonify({'error': 'No resume file provided.'}), 400

    resume_text = preprocess_text(resume_text)
    job_description = preprocess_text(job_description)
    sentiment_analysis = perform_sentiment_analysis(resume_text)
    match_score = calculate_matching_score(resume_text, job_description)

    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description)
    
    missing_skills = job_skills - resume_skills
    matched_skills = job_skills & resume_skills

    missing_skills = filter_relevant_skills(matched_skills, list(missing_skills))

    job_analysis = analyze_job_description(job_description)
    
    # Include keyword density in the response
    # keyword_density = job_analysis.get('keyword_density', [])
    keyword_density = []
    # Get occurrences of matching and non-matching tags
    matching_tag_occurrences = count_tag_occurrences(job_description, list(matched_skills))
    non_matching_tag_occurrences = count_tag_occurrences(job_description, list(missing_skills))

    # Convert occurrences to the required format and append to keyword_density
    for keyword, count in matching_tag_occurrences.items():
        if count > 0:
            keyword_density.append({"keyword": keyword, "density": count})
    
    for keyword, count in non_matching_tag_occurrences.items():
        if count > 0:
            keyword_density.append({"keyword": keyword, "density": count})

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
            'keyword_density': keyword_density  # Include keyword density here
            
        }
    })

@app.route('/generate-updated-resume', methods=['POST'])
def generate_updated_resume():
    try:
        print("inside generate")
        selected_tags = request.json.get('tags', [])
        resume_text = session.get('resume_text', '')
        
        if not resume_text.strip():
            return jsonify({'error': 'No resume text found.'}), 400
        
        updated_resume_text = update_resume_with_tags_using_llm(resume_text, selected_tags)
        
        output_file_path = f"static/updated_resume_{session['unique_id']}.docx"
        save_text_to_docx(updated_resume_text, output_file_path)
        
        return jsonify({'file_path': output_file_path})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred while generating the updated resume.'}), 500

@app.route('/static/<path:filename>')
def download_file(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
