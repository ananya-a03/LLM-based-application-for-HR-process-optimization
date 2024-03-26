import streamlit as st
from resume_processing import *
from jd_processing import *
import time
from dotenv import load_dotenv


class SimilarityMatchingError(Exception):
    """Exception raised for failed similarity matching in Chroma DB."""
    def __init__(self, message="Failed to find similar documents in the database."):
        self.message = message
        super().__init__(self.message)

def load_data(query_result, resume_file):
    input_file = pd.read_excel(resume_file)
    resumes_jobs_similarity = pd.DataFrame(columns=["Name", "Email", "Phone"])
    ids = query_result["ids"][0]  # Extracting ids from the query result

    for resume_id in ids:
        string = input_file.loc[int(resume_id), 'JSON_TO_DICT']  # Convert id to int and locate JSON data
        string = string.replace("'", '"')
        dictionary = json.loads(string)
        resumes_jobs_similarity = resumes_jobs_similarity.append({
            "Name": dictionary.get('name', ''),
            "Email": dictionary.get('emails', ''),
            "Phone": dictionary.get('phones', '')
        }, ignore_index=True)

    return resumes_jobs_similarity


def main():

    load_dotenv()

    st.set_page_config(page_title="Resume Parser")
    st.title("Ranking of Resumes")
    st.subheader("I can help you in resume sorting")

    pdf_files = st.file_uploader("Upload resumes here (PDF only)", type=["pdf"], accept_multiple_files=True)
    

    if pdf_files:
        number_of_files = len(pdf_files)
        st.write(f"You uploaded {number_of_files} resumes.")
        st.success("Resumes uploaded successfully!")
        
        try:
            resumes_read_successfully = read_uploaded_pdfs(pdf_files)
            
            if resumes_read_successfully:
                st.success("The Data from the resumes are extracted successfully!!!")
                st.info('The Resumes are under further processing ......', icon="ℹ️")
                # extract_information_using_openAI_resume_successfully = extract_information_using_openAI_resume(resumes_read_successfully)
                
                try:
                    extract_information_using_openAI_resume(resumes_read_successfully)
                    st.success("Resumes provided are processed and stored in the database!")          
                    with st.container():
                        uploaded_jd = st.text_area("Job Description", height= 100, key= "query_text")                 
                        
                        if uploaded_jd:             
                            top_candidates = st.number_input("Enter the number of required top candidates", value=None, placeholder="Type a number...")                                    
                            if top_candidates is not None:
                                submit = st.button("Submit")
                            else:
                                submit = False

                            if submit:
                                st.info("The given job description is being processed........", icon="ℹ️")
                                try:

                                    skills_returned_value = extract_information_using_openAI_jd(uploaded_jd)
                                    # st.write(f"the returned value of string is {skills_returned_value}")

                                    st.success("The qualifications required for the job position is successfully extracted for the matching process!")

                                    
                                    try:
                                        st.info("The matching process of resumes and job description has started...", icon="ℹ️")
                                        query_result = querying_database_against_jobdescription(skills_returned_value, top_candidates)

                                        if querying_database_against_jobdescription(skills_returned_value, top_candidates):
                                            st.success("The Matching process is completed!")

                                        

                                        st.subheader("Query Result")
                                        if query_result:
                                                df_resumes_jobs_similarity = load_data(query_result,"extracted_resume_data.xlsx")
                                                csv = df_resumes_jobs_similarity.to_csv(index=False)
                                                st.write(df_resumes_jobs_similarity)
                                                download = st.download_button(label="Download CSV", data=csv, file_name="resumes_data.csv", mime="text/csv")
                                                if download:
                                                    st.success("The file is downloaded successfully!")

                                        else:
                                             st.write("No results found.")                                          

                                    except SimilarityMatchingError as e:
                                        st.error(f"Exception occurred as : {e}")

                                    except Exception as e:
                                        st.error(f"Exception occurred: {e}")
                                        

                                except Exception as e:
                                        st.error(f"Exception occurred: {e}")
                            

                        else:                
                            st.warning("Please enter the job description before submitting.")            
                except Exception as e:
                     st.error(f"Exception occured: {e}")
            else:
                st.error("Resumes cannot be read")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
