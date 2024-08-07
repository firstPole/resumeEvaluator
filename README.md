# Resume Evaluator

## Overview

The Resume Evaluator is a web application built with Flask that uses NLP and AI techniques to analyze and match resumes against job descriptions. This tool helps users assess how well their resumes align with job requirements, visualize matching skills, and improve their resumes based on detailed analysis.

## Features

- **Resume and Job Description Upload**: Users can upload their resumes and job descriptions in various formats.
- **Matching Score**: Calculates and displays a matching score indicating the compatibility between the resume and the job description.
- **Tag Analysis**: Shows relevant matching and non-matching skills and qualifications.
- **Keyword Density**: Analyzes and displays the frequency of important keywords.
- **Visual Reports**: Includes charts and graphs to visualize matching scores and keyword density.
- **Updated Resume Generation**: Provides an option to generate an updated resume based on the analysis.
- **Tooltip Information**: Includes tooltips for additional information on matching tags and keyword density.

## Installation

### Prerequisites

- Python 3.x
- Flask
- OpenAI API Key (for NLP and AI features)
- Required Python packages listed in `requirements.txt`

### Clone the Repository

First, clone the repository to your local machine:

    ```bash
    git clone https://github.com/firstPole/resumeEvaluator.git
    cd resumeEvaluator

## Set Up a Virtual Environment
- Create a virtual environment to manage dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

## Install Dependencies

### Install the required Python packages:
    ```bash
    pip install -r requirements.txt

## Configuration
1.	OpenAI API Key: Obtain an OpenAI API key and set it in your environment variables or configuration file.
2.	File Uploads: Ensure you have the necessary permissions to read and write files for handling uploads.
3. .env File: Create a .env file in the root directory of your project and add your OpenAI  API key and other environment variables. For example:
        ```bash 
        OPENAI_API_KEY=your_openai_api_key_here


## Usage
    Running the Application
    To start the Flask application, run:
        ```bash
        python app.py

    Open your web browser and go to http://127.0.0.1:5000/ to access the application.

## How to Use
    1.	Upload Documents: Use the upload forms to submit your resume and the job description you want to match it against.
    2.	Evaluate: Click the 'Evaluate' button to start the analysis.
    3.	Review Results: The application will display the matching score, relevant tags, and keyword density.
    4.	Generate Updated Resume: If desired, click the 'Generate Updated Resume' button to download an improved version of your resume based on the analysis.
## File Structure
    •	app.py: Main file containing the Flask application code.
    •	index.html: HTML file for the main interface of the application.
    •	optimized_result.html: HTML file for displaying the results of the evaluation.
    •	requirements.txt: Contains a list of dependencies required for the project.
    •	static/: Directory for static files such as CSS and JavaScript.
    •	templates/: Directory for HTML templates used in the application.
## Contributing
    Contributions to the Resume Evaluator are welcome! If you have suggestions or improvements, please fork the repository and submit a pull request.

## Acknowledgments
    •	Flask for the web framework.
    •	OpenAI for providing NLP and AI capabilities.
    •	Chart.js for the charting library used in visualizations.
    For any questions or issues, please open an issue on the GitHub repository.
    css

    This version includes all necessary setup instructions, usage guidelines, and additional information in a clear and structured format.

