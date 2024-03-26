from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.output_parsers import OutputFixingParser
from langchain.schema.output_parser import OutputParserException
from json import JSONDecodeError
import pandas as pd
from langchain.llms import OpenAI
import getpass
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field, validator
from langchain.chat_models import ChatOpenAI
import json, os
import ast
from functools import reduce
import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import pdfplumber
from io import BytesIO
from dotenv import load_dotenv
import streamlit as st


#FIRST STEP IS TO EXTRACT THE TEXT FROM THE MULTIPLE RESUMES UPLOADED ON THE APPLICATION INTO A EXCEL FILE FOR FURTHER PROCESSING

def read_uploaded_pdfs(uploaded_files, output_excel_file="extracted_resumes.xlsx"):
    # Create an empty list to store extracted text
    extracted_texts = []

    # Iterate through each uploaded file
    for index, file in enumerate(uploaded_files):
        try:
            # Open the PDF file from BytesIO
            with pdfplumber.open(BytesIO(file.read())) as pdf:
                # Iterate through each page of the PDF
                text = ""
                for page in pdf.pages:
                    # Extract text from the current page
                    text += page.extract_text()

                # Append the extracted text to the list
                extracted_texts.append((index, text))
        except Exception as e:
            st.write(f"Error processing file {index + 1}: {e}")

    # Create a DataFrame from the extracted text
    df = pd.DataFrame(extracted_texts, columns=["Resume_Index", "Extracted_Resume_Text"])
    
    # Get the current directory
    current_directory = os.getcwd()
    
    # Set the path for the output Excel file
    output_excel_path = os.path.join(current_directory, output_excel_file)

    # Write DataFrame to an Excel file
    df.to_excel(output_excel_path, index=False)

    #st.success(f"Data written to: {output_excel_path}")

    return output_excel_path



#SECOND STEP IS TO EXTRACT INFORMATION FROM THE RESUME FILE (that's created in previous function) USING OPENAI
def extract_information_using_openAI_resume(input_file_path):
    # os.environ["OpenAI_API_KEY"] = getpass.getpass("OpenAI API Key:") 
    print("The function \'extract_information_using_openAI_resume()\' is currently running......")
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")   
    model_name = "gpt-3.5-turbo"
    temperature = 0.0
    openai_model = OpenAI(model_name=model_name, temperature=temperature, api_key=openai_api_key)

    class ResumeParser(BaseModel):
        name: str = Field(description="Full name of the candidate in the resume text")
        emails: List[str] = Field(description="Emails of the candidate in the resume text if present")
        phones: List[str] = Field(description="Phone numbers of the candidate if present")
        job_titles: List[str] = Field(description="Job titles if present")
        companies: List[str] = Field(description="Companies that the candidate worked in the past")
        schools_attended: List[str] = Field(description="list of schools the candidate attended")
        skills: List[str] = Field(description="Skills as a list")
        certifications: List[str] = Field(description="Certifications as a list")
        urls: List[str] = Field(description="urls")
    
    resume_file = pd.read_excel(input_file_path)
    resume_file["Extracted_Data_in_JSON"] = ""

    for index, row in resume_file.iterrows():
        resume_text = row["Extracted_Resume_Text"]
            
        resume_parser_query = f"""
            Can you parse the below resume text and answer the questions that follow the passage? \n
            {resume_text} \n
                1. What is the name of the person ? \n
                2. What are the emails if present as an array? \n
                3. What are the phone numbers if present as an array? \n
                4. What are the job titles if present as an array? \n
                5. What are the companies if present as an array? \n
                6. Can you list the schools attended if present as an array \n
                7. Can you list the skills if present in the order of their frequency of occurrence? \n
                8. Can you list the development tools if present ?
                9. Can you list the operating systems if mentioned ?
                10. Can you list any certifications if present in the resume as an array? \n
                11. Can you list links & hyperlinks in a JSON format if present in the resume as an array?
            """
            
        parser = PydanticOutputParser(pydantic_object=ResumeParser)
            
        prompt = PromptTemplate(
                template="""Answer the user query. \n{format_instructions}\n{query}\n""",
                input_variables=["query"],
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )
            
        resume_parsed_output_new = None

        try:
                _input = prompt.format_prompt(query=resume_parser_query)
                resume_parsed_output = openai_model(_input.to_string())
        except OutputParserException:
                print("Exception with parsing output")
                new_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"]))
                resume_parsed_output = new_parser.parse(resume_parsed_output)

        if resume_parsed_output is not None:
                resume_parsed_json = json.dumps(resume_parsed_output, default=lambda x: x.__dict__)
                resume_file.at[index, "Extracted_Data_in_JSON"] = resume_parsed_json
                
        else:
                resume_file.at[index, "Extracted_Data_in_JSON"] = "No Data Was Extracted"

    output_file = "extracted_resume_data.xlsx"
    output_excel_path = os.path.join(os.getcwd(), output_file)
    resume_file.to_excel(output_excel_path, index=False)
    print("Data written to:", output_excel_path)


    print("Calling of next function \'json_output_to_dictionary_resume()\'")
    if json_output_to_dictionary_resume(output_excel_path):
        return True
    else:
        return False
    
    

#THE BELOW FUNCTION IS FOR CLEANING OF THE OBTAINED JSON
def further_clean_and_check(json_string):

    try:
        # Remove backticks from the JSON string
        cleaned_string = json_string.replace('`', '')
        print("\nAfter removing the backtick (`):\n")

        # Check if the cleaned string contains the substring "json"
        if "json" in cleaned_string:
            # If it does, perform the replacement
            cleaned_string = cleaned_string.replace("json", "")
            print("After replacing 'json':\n")
            
            # Return the cleaned string on successful cleaning
            return cleaned_string
        else:
            # If it doesn't contain "json," return None for unsuccessful cleaning
            print("No 'json' found. No further cleaning needed.")
            return None

    except json.JSONDecodeError as e:
        # Handle any exceptions and return None for unsuccessful cleaning
        print("Error during cleaning:", e)
        return None


#FUNCTION FOR CONVERTING OF THE JSON INTO WORKABLE DICTIONARY  

def json_output_to_dictionary_resume(input_file_path):
    print("The function 'json_output_to_dictionary_resume()' is currently running...")
    # Read the Excel file into a DataFrame
    uploaded_files = pd.read_excel(input_file_path)
    
    uploaded_files["JSON_TO_DICT"] = ""

    for index, row in uploaded_files.iterrows():
        raw_string = uploaded_files.loc[index, "Extracted_Data_in_JSON"]
        
        # Check if the cell is empty
        # if pd.isnull(raw_string):
        #     print(f"The cell at index '{index}' is empty. Stopping iteration.")
        #     break
        
        raw_string = raw_string.replace(" ", "")
        print(f"The index '{index}'s data is modified with no whitespace")
            
        raw_string = raw_string.replace("\\n", "")
        print(f"The index '{index}'s data is modified with no new-line")
            
        raw_string = raw_string.replace("\\", "")
        print(f"The index '{index}'s data is modified with no escape sequence")  

        # Remove whitespaces and new-line characters
        cleaned_string = raw_string.replace(" ", "").replace('\n', '') 
           
        modified_string = " " + cleaned_string[1:-1] + " "
        
        try:
            print("\nThe final version of clean and modified json is processing...")
            # Ensure that property names are enclosed in double quotes
            # modified_string = ensure_double_quotes(modified_string)
            python_dict = json.loads(modified_string)

            if isinstance(python_dict, dict):
                uploaded_files.at[index, "JSON_TO_DICT"] = python_dict
                print(f"The index '{index}'s extracted json is successfully converted into a python dictionary")
            else:
                print(f"The index '{index}'s failed to convert into a python dictionary")

        except json.JSONDecodeError as e:
            # Handle JSON decoding errors
            modified_string = further_clean_and_check(modified_string)
            if modified_string is not None:
                python_dict = json.loads(modified_string)

                if isinstance(python_dict, dict):
                    uploaded_files.at[index, "JSON_TO_DICT"] = python_dict
                    print(f"\nThe index '{index}'s extracted json is successfully converted into a python dictionary after further cleaning")

                else:
                    print(f"\nThe index '{index}'s failed to convert into a python dictionary after further cleaning")
                
            else:
                print("\nFurther cleaning unsuccessful. Error decoding JSON:", e)

        print("----------------------------------------------------------------------------------------------")
    
    print("All the extracted JSON Output is converted into dictionary")
    print("Calling the next function 'sentences_of_skills_from_resume()'")
    if sentences_of_skills_from_resume(uploaded_files, input_file_path):
        return True
    else:
        return False


# def ensure_double_quotes(json_str):
#     # Ensure that property names are enclosed in double quotes
#     return json_str.replace("'", '"')


# FUNCTION WRITTEN TO TAKE THE SKILLS FROM THE RESUME TO FORM THE SENTENCES BEFORE STORING IT INTO VECTOR DATABASES
def sentences_of_skills_from_resume(uploaded_files, input_file_path):
    print("The function 'sentences_of_skills_from_resume()' is currently running...")
    
    uploaded_files["sentences_formed_using_skills"] = ""

    print(uploaded_files.head())

    for index, row in uploaded_files.iterrows():
        try:
            input_text = str(row["JSON_TO_DICT"])  # Convert dictionary object to string
            print(f"Given input string of {index} is of type:", type(input_text))

            to_dict = ast.literal_eval(input_text)
            print("\nFirst conversion from str to : ", type(to_dict))

            skills_from_dict = to_dict["skills"]

            if skills_from_dict is not None:
                sentence_formed_skills = reduce(lambda a, b : a + " " + str(b), skills_from_dict)
                uploaded_files.at[index, "sentences_formed_using_skills"] = sentence_formed_skills
                print(f"\nThe row: {index}'s is successfully converted into a sentence! \n")

        except Exception as e:
            print(f"An error occurred: {e}")

        print("--------------------------------------------------")

    uploaded_files.to_excel(input_file_path, index=False)

    print("All the values of skills are taken to form sentences")
    print("Calling the next function 'vectorization_storing_of_resumes_in_chromadb()'")
    
    if vectorization_storing_of_resumes_in_chromadb(input_file_path):
        return True
    else:
        return False




#FUNCTION WRITTEN TO CREATE OR GET CHROMA DB COLLECTION AS A PERSISTENT CLIENT
def chromadb_collection(embedding_model_name, texts, collection_name, persist_directory, distance_function="l2"):
    if os.path.isdir(persist_directory):
        persistent_client = chromadb.PersistentClient(path=persist_directory)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model_name)
        mycollection = persistent_client.get_or_create_collection(name=collection_name, embedding_function=ef)
        return mycollection
    else:
        persistent_client = chromadb.PersistentClient(path=persist_directory)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model_name)
        # We can change distance function when creating collection
        mycollection = persistent_client.get_or_create_collection(collection_name, embedding_function=ef, metadata={"hnsw:space": distance_function})
        mycollection.add(ids=[str(i) for i in range(0, len(texts))], documents=texts)
        return mycollection


#FUNCTION WRITTEN TO STORE AND VECTORIZE THE SENTENCES FORMED USING SKILLS FROM THE RESUME
def vectorization_storing_of_resumes_in_chromadb(input_file_path):
    print("The function \'vectorization_storing_of_resumes_in_chromadb()\' is currently running......")
    input_file = pd.read_excel(input_file_path)

    input_file.dropna(subset=["sentences_formed_using_skills"], inplace=True)

    embedding_model_name = "all-mpnet-base-v2"

    database_path = os.path.join(os.getcwd())
    client = chromadb.PersistentClient(path=database_path)

    list_job_hard_skills = input_file["sentences_formed_using_skills"].to_list()
    print(list_job_hard_skills)

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model_name)
    resume_collection = chromadb_collection("all-mpnet-base-v2", list_job_hard_skills, "resume_collection", "./chromadb_folder", "cosine")

    if resume_collection is not None:
        print("Resume collection is successfully created.")
        print(resume_collection.peek())
    else:
        print("Failed to create resume collection.")


#QUERYING THE DATABASE AGAINST THE JOB DESCRIPTION 
def querying_database_against_jobdescription(skills_from_jd, n_results):
    
    n_results = int(n_results)

    persist_directory = "./chromadb_folder"
    embedding_model_name = "all-mpnet-base-v2"
    collection_name = "resume_collection"

    print("skills from jd:", skills_from_jd)

    persistent_client = chromadb.PersistentClient(path=persist_directory)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model_name)
    mycollection = persistent_client.get_or_create_collection(name=collection_name, embedding_function=ef)

    query_result = mycollection.query(query_texts=[skills_from_jd], n_results=n_results)
    print("\nThe matched resumes are: ",query_result)
    return query_result

