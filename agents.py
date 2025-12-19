from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from state import AgentState
import pandas as pd
import time
import warnings

# Suppress annoying async cleanup warnings from the Google GenAI library
# These occur because the async client is garbage collected in a sync context
warnings.filterwarnings("ignore", message=".*AsyncClient.aclose.*")
warnings.filterwarnings("ignore", message=".*Task was destroyed but it is pending.*")

# Global cache to prevent creating new LLM clients for every request
_llm_cache = {}

def clean_content(content):
    """Extract string content from potential list/dict LLM responses."""
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            elif isinstance(part, str):
                text_parts.append(part)
        return " ".join(text_parts)
    return str(content)

def get_llm(api_key):
    """Get a cached LLM instance to avoid connection pool cleanup issues."""
    if api_key not in _llm_cache:
        _llm_cache[api_key] = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview", 
            google_api_key=api_key, 
            temperature=0,
            max_retries=5
        )
    return _llm_cache[api_key]

def data_analyst_node(state: AgentState, api_key: str, dataframes: dict):
    print("--- DATA ANALYST WORKING ---")
    llm = get_llm(api_key)
    
    user_query = state['messages'][-1].content
    
    if not dataframes:
        return {"messages": [AIMessage(content="No files selected.")]}

    df_list = list(dataframes.values())
    file_names = list(dataframes.keys())
    
    # Determine input format for the agent to ensure correct variable naming (df vs df1, df2...)
    if len(df_list) == 1:
        agent_input = df_list[0]
        mapping_info = f"1. '{file_names[0]}' is loaded into the variable `df`."
    else:
        agent_input = df_list
        mapping_info = "The files are loaded into specific variables (df1, df2, etc.):\n"
        for i, name in enumerate(file_names):
            mapping_info += f"{i+1}. '{name}' is loaded into variable `df{i+1}`\n"
        mapping_info += "\nIMPORTANT: The variable `df` does NOT exist. Use `df1`, `df2` etc."
    
    agent = create_pandas_dataframe_agent(
        llm,
        agent_input,
        verbose=True,
        allow_dangerous_code=True,
        agent_type="zero-shot-react-description",
        agent_executor_kwargs={"handle_parsing_errors": True},
        include_df_in_prompt=False, 
        number_of_head_rows=5,
        prefix=f"""
        You are a Data Analyst working with pandas dataframe(s) IN MEMORY.
        
        DATA MAPPING:
        {mapping_info}
        
        Goal: Answer the user's question using Python logic.
        
        RULES:
        1. **VARIABLES**: Use ONLY `df` (single file) OR `df1`, `df2` (multi-file) as mapped above. No `pd.read_csv`.
        2. **CLEANING**: Normalize column names. Clean currency symbols/mixed types before math. Handle NaNs.
        3. **FORMAT**: Return JUST the numeric answer and source file. No code explanation.
        4. **TOOLS**: Use 'python_repl_ast'.
        """
    )
    
    result_text = ""
    try:
        max_attempts = 5
        base_delay = 30
        
        for attempt in range(max_attempts):
            try:
                response = agent.invoke({"input": user_query})
                result_text = clean_content(response['output'])
                break
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if attempt < max_attempts - 1:
                        wait_time = min(base_delay * (2 ** attempt), 120)
                        print(f"Rate limit hit. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                raise e

    except Exception as e:
        result_text = f"Error analyzing data: {str(e)}"
    
    return {"messages": [AIMessage(content=f"**Data Analyst:**\n{result_text}")]}

def writer_node(state: AgentState, api_key: str):
    print("--- WRITER WORKING ---")
    llm = get_llm(api_key)
    
    analyst_response = state['messages'][-1].content
    user_question = state['messages'][-2].content
    
    prompt = ChatPromptTemplate.from_template(
        "You are a strict technical editor. "
        "Summarize the analysis below into a single, direct sentence based on the user's question.\n\n"
        "User Question: {question}\n"
        "Input Analysis: {analysis}\n\n"
        "Rules:\n"
        "1. **Concise**: No greetings or filler words.\n"
        "2. **Direct**: Start immediately with the subject/value.\n"
        "3. **Context**: Briefly label what the number represents.\n"
        "4. **No Filenames**: Do NOT mention the source filename.\n"
        "5. Use a compact bulleted list for multiple values."
    )
    chain = prompt | llm
    
    max_writer_retries = 3
    final_text = ""
    for attempt in range(max_writer_retries):
        try:
            response = chain.invoke({"analysis": analyst_response, "question": user_question})
            final_text = clean_content(response.content)
            break
        except Exception as e:
             if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < max_writer_retries - 1:
                 time.sleep(30)
                 continue
             final_text = f"Error generating report: {str(e)}"
    
    return {"messages": [AIMessage(content=f"**Final Answer:**\n{final_text}")]}