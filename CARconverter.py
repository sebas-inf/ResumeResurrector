import streamlit as st
import openai
import pandas as pd
from tqdm import tqdm
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
import requests

def toCAR(experience):
    prompt = f"Reformat this experience using the CAR method {{{experience}}}"
    try:
            # Call the OpenAI API to generate a completion based on the prompt
            carMethod = openai.Completion.create(
                engine="davinci",  # Choose an engine, e.g., "davinci" or "text-davinci-003"
                prompt=prompt,
                max_tokens=1000  # Adjust as needed
            )

            # Extract and return the generated text from the API response
            return carMethod.choices[0].text.strip()

    except openai.error.OpenAIError as e:
        print(f"Error: {e}")
        return None