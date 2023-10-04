import openai
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

openai.api_key = st.secrets["OPENAI_API_KEY"]
llm_model = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0.0, max_tokens = 300)

daily_objectives_schema = ResponseSchema(name = "Daily Learning Objectives",
                                         description = "Daily tasks and outcomes at least 50 words")

other_schema = ResponseSchema(name= "Other",
                              description = "other daily pharmacy jobs at least 50 words")

response_schema = [daily_objectives_schema,
                   other_schema]

output_parser = StructuredOutputParser.from_response_schemas(response_schema)
format_instructions = output_parser.get_format_instructions()

def daily_internship_diary(day_of_internship, todays_topic):
    prompt_template = """
    You are a internship diary writer of a pharmacy student.
    Your task is the write a daily internship diary.  
    
    Take the today's topic below delimited by triple backticks and write a internship diary mentioning about it.

    today's topic:  ```{todays_topic}```
    
    day of internship: {day_of_internship}
    
    minimum 50, maximum 60 words of text;
    
    {format_instructions} 
    """

    input_variables = ["day_of_internship", "todays_topic"]
    prompt = ChatPromptTemplate(messages = [HumanMessagePromptTemplate.from_template(prompt_template)],
                                input_variables = input_variables,
                                template = prompt_template,
                                partial_variables = {"format_instructions": format_instructions})

    diary_llm = LLMChain(llm = llm_model, prompt = prompt, output_parser = output_parser, verbose = True)

    dairy = diary_llm.predict_and_parse(day_of_internship = [day_of_internship], todays_topic = [todays_topic])

    return dairy


st.title("Pharmacy Internship Diary Generator for MY LOVE")

day_of_internship = st.text_input("Day of Internship:")
todays_topic = st.text_area("Today's Topic:")

if st.button("Generate Diary Entry"):
    if not day_of_internship or not todays_topic:
        st.warning("Please provide both the day of internship and today's topic.")
    else:
        result = daily_internship_diary(day_of_internship, todays_topic)
        if 'Daily Learning Objectives' in result:
            st.subheader("Daily Learning Objectives:")
            st.write(result['Daily Learning Objectives'])

        if 'Other' in result:
            st.subheader("Other:")
            st.write(result['Other'])

if __name__ == "__main__":
    st.write("Input your details and generate your internship diary!")
