from crewai import Agent, Crew, Process, Task
from langchain_community.tools import DuckDuckGoSearchRun
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
import os


llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key="AIzaSyCApXbTlB1M2k-L9S-LLWzp5xXD3VZleJM")

search_tool = DuckDuckGoSearchRun()

st.title('Research & Write')

topic = st.text_input('Topic')

if st.button('Research and Write!'):
    researcher = Agent(
        role='Senior Student Researcher', 
        goal=f"Uncover groundbreaking technologies around {topic}",
        backstory="Driven by curiosity, you're at the forefront of innovation",
        verbose=True,
        llm=llm
    )

    writer = Agent(
        role="Writer", 
        goal=f"Narrate compelling tech stories about {topic}",
        backstory="With a flair for simplifying complex topics, you craft engaging narratives.",
        verbose=True,
        llm=llm
    )

    research_task = Task(
        description=f"""
            Identify the next big trend in {topic}.
            Focus on identifying pros and cons and the overall narrative.

            Your final report should clearly articulate the key points, 
            its market opportunities, and potential risks.
        """,
        expected_output=" A 3 paragraphs long report on the latest AI trends.", 
        max_iter=1, 
        tools=[search_tool], 
        agent=researcher
    )

    write_task = Task(
        description=f"""
            Compose an insightful article on {topic}.
            Focus on the latest trends and how it's impacting the industry.
            This article should be easy to understand, engaging and positive.
        """,
        expected_output=f"A 5 paragraph article on {topic} advancements", 
        tools=[search_tool],
        agent=writer
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential
    )

    result = crew.kickoff()
    st.write(result)


