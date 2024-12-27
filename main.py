from dotenv import load_dotenv
from crewai import Crew, Agent, Task, Process
import asyncio
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai_tools import (
    ScrapeWebsiteTool,
    WebsiteSearchTool,
    YoutubeVideoSearchTool
)   
import streamlit as st
def get_or_create_event_loop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        return asyncio.new_event_loop()

loop = get_or_create_event_loop()
asyncio.set_event_loop(loop)

load_dotenv()

#LLM
llm = ChatGoogleGenerativeAI(
    model='gemini-1.5-flash',
    verbose=True,
    temperature=0.5,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

#Agents
researcher = Agent(
    role="Senior Researcher",
    goal='Uncover and extract information from the URL',
    verbose=True,
    memory=True,
    backstory="""You are curious and want to extract data from YouTube videos and websites. You want to get all the key information about the particular topic.""",
    tools=[],
    llm=llm,
    allow_delegation=True
)

writer = Agent(
    role='Writer',
    goal="Write information about the provided URL in quick way",
    verbose=True,
    memory=True,
    backstory="""You are an expert and a professional agent in simplifying complex information and narrating them in a very meaningful and mannered way in a quick and fast process.""",
    tools=[],
    llm=llm,
    allow_delegation=False
)

st.title("URL Information Extractor")
user_input = st.text_input("Enter URL")

def is_youtube_url(url):
    return "youtube.com" in url.lower() or "youtu.be" in url.lower()
    return False

if st.button("Enter"):
    url = user_input.strip()
    
    if is_youtube_url(url):
        ytscraper = YoutubeVideoSearchTool(
            config=dict(
                llm=dict(
                    provider="google",
                    config=dict(
                        model="gemini-1.5-flash",
                    ),
                ),
                embedder=dict(
                    provider="google",
                    config=dict(
                        model="models/embedding-001",
                        task_type="retrieval_document",
                    ),
                ),
            ), youtube_video_url=url
        )
        researcher_task = Task(
            description=f'Identify and extract information from {url}',
            expected_output=f'A long 3 paragraph research blog on {url} but do it quickly',
            agent=researcher,
            tools=[ytscraper]
        )

        writer_task = Task(
            description=f'Compose an easy-to-understand article on {url}, focusing on trends impacting the industry',
            expected_output=f'A 3 paragraph article on {url}',
            agent=writer,
            async_execution=False,
            tools=[ytscraper],
            output_file='example.mb'
        )

    else:
        webscrape1 = ScrapeWebsiteTool(website_url=url)
        webscrape2 = WebsiteSearchTool(
            config=dict(
                llm=dict(
                    provider="google",
                    config=dict(
                        model="gemini-1.5-flash",
                    ),
                ),
                embedder=dict(
                    provider="google",
                    config=dict(
                        model="models/embedding-001",
                        task_type="retrieval_document",
                    ),
                ),
            ), website=url
        )
        researcher_task = Task(
            description=f'Identify and extract information from {url}',
            expected_output=f'A long 3 paragraph research blog on {url}',
            agent=researcher,
            tools=[webscrape1, webscrape2]
        )

        writer_task = Task(
            description=f'Compose an easy-to-understand article on {url}, focusing on trends impacting the industry',
            expected_output=f'A 3 paragraph article on {url}',
            agent=writer,
            async_execution=False,
            tools=[webscrape1, webscrape2],
            output_file='example.mb'
        )

    researcher.tools = researcher_task.tools
    writer.tools = writer_task.tools

    my_crew = Crew(
        agents=[researcher, writer],
        tasks=[researcher_task, writer_task],
        process=Process.sequential,
        verbose=True,
    )

    result = my_crew.kickoff()
    st.write(result)
