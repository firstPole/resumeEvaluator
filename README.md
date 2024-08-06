# Resume Evaluator

## Overview

The Resume Evaluator is a Flask-based web application designed to analyze and match resumes against job descriptions. It uses NLP and AI techniques to evaluate resumes and job descriptions, providing insights into matching skills, keyword density, and overall compatibility. The application generates a visually appealing report with charts and tags to help users improve their resumes and better align them with job requirements.

## Features

- **Resume and Job Description Upload**: Allows users to upload resumes and job descriptions for evaluation.
- **Matching Score**: Provides a score indicating how well the resume matches the job description.
- **Tag Analysis**: Displays matching and non-matching skills and qualifications.
- **Keyword Density**: Shows the density of important keywords in the resume and job description.
- **Visual Reports**: Includes charts and graphs to visualize matching scores and keyword density.
- **Updated Resume Generation**: Offers an option to generate and download an updated resume.

## Installation

### Prerequisites

- Python 3.x
- Flask
- OpenAI API Key (for NLP and AI features)
- Other dependencies as listed in `requirements.txt`

### Clone the Repository

```bash
git clone https://github.com/firstPole/resumeEvaluator.git
cd resumeEvaluator

**### Install Dependencies**
Create a virtual environment and install the required packages:

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt

**### Configuration**
Obtain an OpenAI API key and add it to your environment variables or configuration file.
Ensure that you have the necessary permissions to read and write files if you're working with file uploads.

**### Usage**
Running the Application
To start the Flask server, use the following command:

python app.py

Open your web browser and navigate to http://127.0.0.1:5000/ to access the application.

**Uploading Resumes and Job Descriptions**

Upload your resume and job description using the provided forms.
Click the 'Evaluate' button to analyze the documents.
Review the matching score, tags, and keyword density.
Optionally, generate and download an updated resume using the 'Generate Updated Resume' button.

**File Structure**

app.py: Main Flask application file.
index.html: HTML file for the main application interface.
optimized_result.html: HTML file for displaying the evaluation results.
requirements.txt: List of Python dependencies.
README.md: Project documentation.
**Contributing**
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. Ensure that your changes are well-documented and tested.

**Contact**

Feel free to customize the contact details and any other sections to fit your project’s specifics.
