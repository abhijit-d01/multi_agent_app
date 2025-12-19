# **Multi-Agent Data Analysis Tool \- User Guide**

## **1\. Architecture Overview**

![Architecture Diagram](/Sales Dataset/Architecture.png)
This application is built on a **Multi-Agent System (MAS)** architecture using **LangChain** and **Google Gemini** (Gemini 2.5 Flash). Instead of a single AI trying to do everything, the responsibilities are split into two specialized agents: one for logical execution and one for communication.

### **Key Technologies**

* **LangChain**: Orchestrates the flow between agents and tools.  
* **Google Gemini 1.5 Flash**: Provides the underlying intelligence for both code generation and text summarization.  
* **Pandas**: The engine used by the Analyst agent to manipulate data in-memory.

## **2\. Agent Structure**

### **Agent 1: The Data Analyst ("The Brain")**

* **Type**: Zero-shot ReAct (Reasoning \+ Acting) Agent.  
* **Capabilities**:  
  * Accesses a Python REPL (Read-Eval-Print Loop) environment.  
  * Writes and executes valid Python/Pandas code on the fly.  
  * Can handle data cleaning, complex filtering, aggregations, and math.  
* **Input**: Raw CSV/Excel dataframes and the user's natural language query.  
* **Output**: A raw, fact-based numeric or textual answer (often unpolished).

### **Agent 2: The Writer ("The Voice")**

* **Type**: Instruction-tuned Summarization Chain.  
* **Capabilities**:  
  * Understands context and business tone.  
  * Strips away technical jargon (e.g., "The dataframe index 0 shows...").  
* **Input**: The raw output from the Data Analyst and the original user question.  
* **Output**: A concise, executive-level summary sentence or bullet point.

## **3\. Processing Workflow**

When a user asks a question (e.g., "Which SKU sold the most?"), the system follows this linear pipeline:

1. **State Initialization**: The system receives the file(s) and the question.  
2. **Analyst Execution**:  
   * The Analyst inspects the dataframe columns.  
   * It writes Python code to solve the query (e.g., df.groupby('SKU')\['Qty'\].sum()).  
   * It executes the code and captures the result.  
3. **Handoff**: The raw result (e.g., "SKU-99, 500 units") is passed to the Writer.  
4. **Refinement**: The Writer formats this into a human-readable response (e.g., *"The highest selling item is SKU-99 with 500 units sold."*).  
5. **Final Output**: The result is displayed to the user.

## **4\. Usage Steps**

### **Prerequisites**

* Python 3.9 or higher.  
* A valid Google Cloud API Key with access to Gemini models stored in .env file  
* **Install Dependencies**:

`pip install -r requirements.txt`

* **Start the Application**:

`streamlit run app.py`

## **5\. Screenshots:** **![][image2]![][image3]![][image4]**

## **6\. Production Readiness Roadmap**

To transition this application from a prototype to a production environment, consider implementing the following enhancements:

### **Security & Sandboxing**

* **Sandboxed Execution**: Currently, the agent executes Python code locally. In production, this poses a security risk. Use isolated environments like **Docker containers** or secure sandbox services (e.g., E2B) to execute generated code safely.  
* **PII Filtering**: Implement a pre-processing layer to detect and redact Personally Identifiable Information (PII) from user queries or datasets before sending them to the LLM.

### **Observability & Monitoring**

* **Tracing**: Integrate with tools like **LangSmith** to trace agent thought processes, latency, and token usage.  
* **Feedback Loops**: Add a mechanism for users to rate answers (thumbs up/down). Use this data to fine-tune the Writer agent's prompts.

### **Reliability**

* **Validation Guardrails**: Use libraries like **Guardrails AI** to validate the structure and content of the Analyst's output (e.g., ensuring no Hallucinated columns are used) before passing it to the Writer.  
* **State Persistence**: Instead of in-memory state, use a persistent backend (e.g., **Redis** or **PostgreSQL**) to manage conversation history, allowing for long-running sessions and crash recovery.

### **Performance**

* **Semantic Caching**: Implement caching (e.g., **GPTCache**) for common queries. If a user asks "Total Revenue" twice, serve the second answer from the cache to save costs and reduce latency.
