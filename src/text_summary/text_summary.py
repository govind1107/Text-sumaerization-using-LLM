import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv

from src.text_summary.logger import logging

from langchain.chat_models import ChatOpenAI

from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI 

import re
from nltk.tokenize import sent_tokenize
from transformers import pipeline


# loading the enviorment variables
load_dotenv()
# get the key
KEY = os.getenv("OPEN_API_KEY")

# create the openai llm client
llm = ChatOpenAI(openai_api_key = KEY, model_name="gpt-3.5-turbo",
                 temperature=0.3)


prompt_template = """Write a concise summary of the following topic in {number} words only:
"{text}"
CONCISE SUMMARY:"""

summary_generation_prompt = PromptTemplate(
    input_variables=['number','text'],
    template=prompt_template
)


llm_chain = LLMChain(llm=llm, prompt=summary_generation_prompt, output_key='summary',verbose=True)


def preprocess_text(text):
    # Remove citations
    text = re.sub(r'\[\d+\]', '', text)
    return text



def generate_summary(text):
    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    # Concatenate sentences into one string for summarization
    text = ' '.join(sentences)
    # Initialize pipeline for summarization
    summarization_pipeline = pipeline("summarization")
    # Generate summary
    summary = summarization_pipeline(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']


def pipeline(text):
    preprocessed_text = preprocess_text(text)

    summary_text = generate_summary(preprocessed_text)

    return summary_text



