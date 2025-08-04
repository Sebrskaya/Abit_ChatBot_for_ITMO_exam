# agent.py
from langchain.agents import initialize_agent, AgentType
from local_llm import load_local_llm
from tools import CourseRecommenderTool, ProgramComparatorTool

def get_agent_executor(retriever):
    llm = load_local_llm()

    tools = [
        CourseRecommenderTool(),
        ProgramComparatorTool(retriever=retriever),
    ]

    agent_executor = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent_executor