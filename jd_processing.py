from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.output_parsers import OutputFixingParser
from langchain.schema.output_parser import OutputParserException
from json import JSONDecodeError
import pandas as pd
from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field, validator
from langchain_community.chat_models import ChatOpenAI
import json, os
import pyarrow as pa
import getpass
import json
from functools import reduce
import ast
from dotenv import load_dotenv
from resume_processing import *

# function for extraction of details from jd using openAi
def extract_information_using_openAI_jd(uploaded_jd):
    
    #os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY") 
    model_name = "gpt-3.5-turbo" 
    temperature = 0.0
    openai_model = OpenAI(model_name=model_name, temperature=temperature,
               api_key=openai_api_key)
    
   # Defining class for structuring
    class JDParser(BaseModel):
        #hard_techical_skills: List[str] = Field(description="Hard technical skills if present")
        #soft_skills: List[str] = Field(description="Soft skills as a list")
        job_qualifications: List[str] = Field(description = "Job qualification if present")
        experience: List[str] = Field(description = "Experience if present")
        salary: List[str] = Field(description = "CTC or Salary if present")
        Sponsor_Visa: List[str] = Field(description="Sponsor Visa ")
    
    JD_text = uploaded_jd
    
    if not isinstance(JD_text, str):
        JD_text = str(JD_text)
        
    JD_parser_query = f"""
    Can you parse the below Job description text and answer the questions that follow the passage? \n
    {JD_text} \n
    1. What are the job qualifications and soft skills if present as an array?\n
    2. What are the years of expereince if present as an array?\n
    3. What is the salary if present?\n
    4. Respond YES if job sponsors Visa otherwise repond NO\n
    """
    
    parser = PydanticOutputParser(pydantic_object=JDParser)
    prompt = PromptTemplate(
        template="""Answer the user query. \n{format_instructions}\n{query}\n""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    JD_parsed_output_new = None
    
    try:
        _input = prompt.format_prompt(query=JD_parser_query)
        JD_parsed_output = openai_model(_input.to_string())
    except OutputParserException:
        print("Exception with parsing output")
        new_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"]))
        JD_parsed_output = new_parser.parse(JD_parsed_output)
        
    if JD_parsed_output is not None:
        JD_parsed_output_json = json.dumps(JD_parsed_output, default=lambda x: x.__dict__)
        #return JD_parsed_output
        returned_value = json_output_to_dictionary_jd(JD_parsed_output_json)
        print("\nthe function \'json_output_to_dictionary_jd\' is called \n")
        
        if json_output_to_dictionary_jd(JD_parsed_output_json):
            return returned_value
        else:
            return False
            
    else:
        return JD_parsed_output
    


# function for converting json to dict
def  json_output_to_dictionary_jd(jsonString):

    print("\n function \'json_output_to_dictionary_jd()\' is currently running...........")

    raw_string = jsonString
    raw_string = raw_string.replace("\\n", "")
    raw_string = raw_string.replace("\\", "")
    raw_string = raw_string.replace("json","")
    raw_string = raw_string.replace("`","")

    # Remove whitespaces and new-line characters
    cleaned_string = raw_string.replace('\n', '')

    modified_string = " " + cleaned_string[1:-1] + " "

    try:
        print("\nThe final version of clean and modified json is processing.........")
        python_dict = json.loads(modified_string)

        if isinstance(python_dict, dict):
            # return python_dict
            print("\ncalling function \'sentences_of_skills_from_jd()\'")
            returned_value = sentences_of_skills_from_jd(python_dict)

            if sentences_of_skills_from_jd(python_dict):
                return returned_value
            else:
                return False
        else:
            return None
            
    except json.JSONDecodeError as e:
        print("\nError decoding JSON:", e)

    
# function for converting dict to sentences
def sentences_of_skills_from_jd(dictionary):

    try:
        

        print("\nfunction \'sentences_of_skills_from_jd()\' is currently running...........\n")

        print(f"Given input string of is of type: ", type(dictionary))

        to_str = str(dictionary)
        print(f"Given input string of is of type after performing str() ", type(to_str))

        to_dict = ast.literal_eval(to_str)
        
        print("\nFirst conversion from str to : ", type(to_dict))

        skills_from_dict = to_dict['job_qualifications']

        print("\n the qualifications take to form sentences are: ",skills_from_dict)

        if skills_from_dict is not None:
            str_of_skills_jd = reduce(lambda a, b : a + " " + str(b), skills_from_dict)
            print(f"\nThe job qualifications are successfully converted into a sentence as: \n", str_of_skills_jd)
            return str_of_skills_jd

        else:
            str_of_skills_jd = ""
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    
