# Resume Resurrector

![Logo](./images/inverted_panda.png)

## Description

Resume Resurrector is a web application powered by LangChain and OpenAI's LLM model, designed to help users evaluate their resumes for job applications. It uses natural language processing to analyze resumes in the context of specific job descriptions and provides feedback on various criteria.

## Features

- Resume analysis based on job descriptions.
- Evaluation of education, technical skills, projects, experience, problem-solving abilities, and communication skills.
- User-friendly web interface built with Streamlit.
- Integration with LangChain and OpenAI.

## DEMO
http://resumeresurrector.streamlit.app/

## Getting Started

To get started with Resume Resurrector, follow these steps:

1. **Clone this repository:** Click on the "Clone or Download" button on the GitHub repository and select "Download ZIP" to download the project files to your computer.

2. **Install Dependencies:** Install the required Python dependencies by running the following command in your Python environment:

   ```python
   pip install -r requirements.txt

3. Set up OpenAI API Key: Create a .env file in the project directory and add your OpenAI API key as follows:
    OPENAI_API_KEY=your_api_key_here
4. Run the Application: Start the Streamlit app by running the following command:
    streamlit run 1_About.py
   
## Usage
Enter Job Role: Provide the job role you'd like to apply for.

Job Description: Paste or type the job description for the role.

Upload Resume: Upload your resume in PDF format using the provided file uploader.

Resume Analysis: The application will analyze your resume and provide feedback based on the provided job description.

## Contributing
Contributions to this project are welcome! To contribute:

Fork the repository.

Create a new branch for your feature or bug fix.

Make your changes and commit them.

Push your changes to your forked repository.

Create a pull request to merge your changes into the main repository.

## Acknowledgments
Thanks to Streamlit, LangChain, and OpenAI for their amazing tools and technologies. 
