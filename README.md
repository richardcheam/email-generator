# Job Portfolio Matcher

This project uses vector search to match job descriptions with relevant portfolio items based on skills and technologies. It scrapes job listings, extracts key details (role, skills, experience), and stores portfolio items as vector embeddings using ChromaDB. By leveraging semantic search, it identifies the most relevant projects from a user's portfolio, which are then used to generate personalized cold emails for internship applications.

## Key Features:
* Job Description Scraping: Extracts job requirements from job postings.
* Portfolio Matching: Uses vector search to find relevant portfolio items based on job skills.
* Personalized Email Generation: Creates tailored emails with links to the most relevant portfolio items.
* This system improves job application efficiency by leveraging machine learning and semantic search techniques.

## To get started: 

create a new environment, then
run "pip install -r requirements.txt"

API_KEY of groqcloud is needed, generate one here: https://console.groq.com/login
