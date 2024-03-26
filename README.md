# LLM-based-application-for-HR-process-optimization
## Overview

The LLM-based application simplifies and improves HR processes by using Language, Logic, and Machine Learning. It leverages Natural Language Processing, machine learning, and analytics to enhance efficiency, effectiveness, and decision-making in HR operations.

This user-friendly, scalable, and customizable application integrates with existing HR systems and tools, ensuring smooth data flow and compatibility. Its ultimate goal is to optimize HR processes, minimize manual efforts, enhance decision-making, and improve the employee experience.


## Tasks Performed
- **Text Extraction and Text Processing:**
  1. Resumes and job descriptions are processed using "GPT-3.5 Turbo".
  2. Resumes are analyzed to extract details like name, email, URLs, skills, previous job titles, and experiences, stored in JSON format.
  3. Similar processing is applied to job descriptions to extract job titles, required skills, salary, and sponsorship details as JSON.
  4. Extracted details are stored in .xlsx files, but may require further cleaning to ensure accuracy.

- **Tokenization and Vectorization:**
  1. Skills are extracted from resumes and converted into embeddings.
  2. The "all-mpnet-base-v2" model is used for this process.
  3. Embeddings are stored in a vector database, such as Chroma DB.

- **Querying of the Database:**
  1. Skills from the job description are used to query the database.
  2. This query retrieves the top "N" results from the stored resumes.
  3. The results facilitate efficient candidate matching and selection.

Through these systematic steps, LLM recruiters streamline the recruitment process, ensuring optimal matches between candidates and job opportunities.

## Software Requirements
1. OpenAi gpt 3.5-trubo
2. all-mpnet-base-v2 (Sentence Tranformers)
3. Chroma Db (Vector Database)
4. LangChain
