# Resume Evaluation Application

This Flask application evaluates resumes by comparing them to job descriptions. It calculates a match score and identifies both matched and missing key phrases between the resume and the job description.

## Features

- Upload resumes in PDF or DOCX format.
- Input job description as plain text.
- Calculate match score based on cosine similarity.
- Identify and display matched and missing key phrases.

## Requirements

- Python 3.6 or higher
- Flask
- scikit-learn
- python-docx
- PyPDF2

## Installation

1. Clone this repository:

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Dependencies

Create a `requirements.txt` file with the following contents:

```plaintext
Flask
scikit-learn
python-docx
PyPDF2
