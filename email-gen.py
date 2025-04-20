import pandas as pd
import uuid
import chromadb
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Step 1: Prompt the user for inputs
job_url = input("Enter the job URL: ")
csv_file = input("Enter the path to your CSV file: ")
api_key = input("Enter your API key: ")

print("\nGeneration in progess....\n")

# Step 1: Initialize LangChain model
llm = ChatGroq(
    temperature=0, 
    groq_api_key=api_key,  
    model_name="llama-3.3-70b-versatile"
)

# Step 2: Web scraping
loader = WebBaseLoader(job_url)
page_data = loader.load().pop().page_content

# Step 3: Job Data Extraction using Langain
prompt_extract = PromptTemplate.from_template(
        """
        ### SCRAPED TEXT FROM WEBSITE:
        {page_data}
        ### INSTRUCTION:
        The scraped text is from the career's page of a website.
        Your job is to extract the job postings and return them in JSON format containing the 
        following keys: `role`, `experience`, `skills` and `description`.
        Only return the valid JSON.
        ### VALID JSON (NO PREAMBLE):    
        """
)
# getting prompt then pass it to model 
chain_extract = prompt_extract | llm 
res = chain_extract.invoke(input={'page_data':page_data})
#print(res.content) #this is str

# convert text to json
json_parser = JsonOutputParser()
json_res = json_parser.parse(res.content)
job = json_res # for step 6

# Step 4: Load User Data from CSV
df = pd.read_csv(csv_file)
student = chromadb.PersistentClient('vectorstore') # PersistentClient will create db on disk physically 
collection = student.get_or_create_collection(name="portfolio")

if not collection.count():
    for _, row in df.iterrows():
        collection.add(documents=row["Techstack"],
                       metadatas={"links": row["Links"]},
                       ids=[str(uuid.uuid4())])


# Assuming 'skills' in job JSON and user portfolio have similar fields
# Step 5: Matching User Portfolio Links with Job Skills
links = collection.query(query_texts=job['skills'], n_results=2).get('metadatas', []) # for step 6


# Step 6: Email Generator
prompt_email = PromptTemplate.from_template(
        """
        ### JOB DESCRIPTION:
        {job_description}

        ### INSTRUCTION:
        You are a student who is pursuing a double degree, Engineer's degree and Master's degree, in Data Science at Paris-Saclay University.
        Throughout your studies, you have acquired different skills and have done numerous projects related to machine learning, deep learning, and so on.
        Right now, you are looking for an internship. 
        So, your job is to write a cold email to the recruiter regarding the internship mentioned above describing your capability in fulfilling their needs or roles.
        Also add the most relevant ones from the following links to showcase your portfolio: {link_list}
        Remember you are a student applying for internship. 
        Do not provide a preamble.
        Also, your name is Bob
        ### EMAIL (NO PREAMBLE):

        """
        )

# throw prompt to LLM
chain_email = prompt_email | llm
res = chain_email.invoke({"job_description": str(job), "link_list": links})
# Print the generated email
print("Generated email:\n")
print(res.content)
print("\n")