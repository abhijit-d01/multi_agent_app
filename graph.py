from langgraph.graph import StateGraph, END
from state import AgentState
from agents import data_analyst_node, writer_node
from functools import partial

def create_workflow(api_key: str, dataframes: dict):
    """
    Creates the Data Analysis Graph.
    We inject both the API Key and the DataFrames into the nodes.
    """
    
    # 1. Initialize the Graph
    workflow = StateGraph(AgentState)

    # 2. Define Nodes
    # Inject dependencies using partial
    workflow.add_node("data_analyst", partial(data_analyst_node, api_key=api_key, dataframes=dataframes))
    workflow.add_node("writer", partial(writer_node, api_key=api_key))

    # 3. Define Edges
    # Start -> Data Analyst
    workflow.set_entry_point("data_analyst")
    
    # Data Analyst -> Writer
    workflow.add_edge("data_analyst", "writer")
    
    # Writer -> End
    workflow.add_edge("writer", END)

    # 4. Compile
    return workflow.compile()