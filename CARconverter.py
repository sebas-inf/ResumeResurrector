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

def toCAR(experience, langchain_url, langchain_api_key):
    prompt = f"Reformat this experience using the CAR method {{{experience}}}"
    # Set up the request headers for Langchain
    langchain_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {langchain_api_key}"
    }

    # Define the Langchain data payload with the prompt
    langchain_data = {
        "prompt": prompt
    }

    try:
        # Send the request to Langchain
        langchain_response = requests.post(langchain_url, json=langchain_data, headers=langchain_headers)

        # Check if the Langchain request was successful
        if langchain_response.status_code == 200:
            langchain_data = langchain_response.json()

            # Extract the response from Langchain
            car = langchain_data.get('response')

            # Return the Langchain response (which should be the OpenAI response)
            return car

        else:
            print(f"Error: Langchain request failed with status code {langchain_response.status_code}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None