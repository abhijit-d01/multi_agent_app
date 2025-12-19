from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    """
    The state of the graph.
    
    Attributes:
        messages: A list of messages. We use the `operator.add` reducer 
                  so that new messages are appended to the list rather 
                  than overwriting it.
    """
    messages: Annotated[List[BaseMessage], operator.add]