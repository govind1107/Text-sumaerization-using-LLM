import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.text_summary.utils import read_file
import streamlit as st
from langchain.callbacks import get_openai_callback

from src.text_summary.logger import logging
from src.text_summary.text_summary import llm_chain

from src.text_summary.text_summary import pipeline


st.title("Summary generator app with LANGCHAIN")


with st.form("user_inputs"):
    uploaded_file = st.file_uploader("upload file or text file")
    word_count = st.number_input("No of word",min_value=20,max_value=200)

    button = st.form_submit_button("Generate Summary")

    
    if button and uploaded_file is not None and word_count:
        st.spinner("loading.....")
        try:
            text = read_file(uploaded_file)
            with get_openai_callback() as cb:
                response = llm_chain(
                    {

                        "number":word_count,
                        "text":text,

                    }
                )
        except Exception as e:
            traceback.print_exception(type(e),e,e.__traceback__)
            st.error("Error")

        else:
            print(f"Total Token: {cb.total_tokens}")
            print(f"Prompt tokens : {cb.prompt_tokens}")
            print(f"Completion tokens : {cb.completion_tokens}")

            print(f"Total cost : {cb.total_cost}")

        if isinstance(response,dict):
            summary = response.get("summary",None)

            if summary is not None:
                st.write(summary)

            else:
                st.error("Error in table data")

        else:
            st.write(response)


st.title("Summary generator app with Transformers")






# with st.form("user_inputs for transformers"):
#     uploaded_file = st.file_uploader("upload file or text file")


#     button = st.form_submit_button("Generate Summary using transformers")

    
#     if button and uploaded_file is not None:
#         st.spinner("loading.....")
#         try:
#             text = read_file(uploaded_file)

#             summary = pipeline(text)

#             if summary:
#                 st.write(summary)
#             else:
#                 st.error("Error")

           
#         except Exception as e:
#             traceback.print_exception(type(e),e,e.__traceback__)
#             st.error("Error")

       



