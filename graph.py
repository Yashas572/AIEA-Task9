from typing import TypedDict, List
import subprocess
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, START, END
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class GraphState(TypedDict):
    question: str
    relevant_context: List[str]
    reasoning: str
    prolog_query: str
    query_result: str

def run_prolog_query(raw_query: str) -> str:
    query = raw_query.strip()
    if query.endswith("."):
        query = query[:-1].strip()
    m = query.split("(", 1)
    if len(m) < 2 or not query.endswith(")"):
        return "Error: Unable to parse query."
    inside = query[len(m[0]) + 1 : -1].strip()
    candidates = [arg.strip() for arg in inside.split(",")]
    var_name = None
    for tok in candidates:
        if tok and tok[0].isupper():
            var_name = tok
            break
    if not var_name:
        return "Error: Unable to determine variable from query."
    wrapper = f"findall({var_name}, ({query}), L), write(L)"
    proc = subprocess.Popen(
        ["swipl", "-q", "-s", "prolog_kb.pl", "-g", wrapper, "-g", "halt."],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    out, err = proc.communicate()
    if err:
        return f"Error: {err.strip()}"
    return out.strip()

if not os.path.exists("chroma_db"):
    loader = TextLoader("prolog_kb.pl", encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    vectordb.add_documents(chunks)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
retriever = vectordb.as_retriever()

def build_graph() -> StateGraph:
    llm = ChatOpenAI(model_name="gpt-4")
    chain_prompt = PromptTemplate.from_template(
        "You are given the following Prolog facts and rules:\n"
        "{relevant_context}\n\n"
        "Here is a user question:\n"
        "\"{question}\"\n\n"
        "Think step by step about how to translate this question into a Prolog query using the given facts and rules. "
        "Show your entire reasoning, then output ONLY the final Prolog query on a separate line prefixed by \"QUERY: \". "
        "Do not include any other text after that line."
    )
    cot_chain = LLMChain(llm=llm, prompt=chain_prompt, output_key="reasoning")
    builder = StateGraph(GraphState)

    def retrieve_context(state: GraphState) -> dict:
        docs = retriever.get_relevant_documents(state["question"])
        return {"relevant_context": [doc.page_content for doc in docs]}

    builder.add_node("RetrieveContext", retrieve_context)

    def chain_of_thought(state: GraphState) -> dict:
        joined = "\n".join(state["relevant_context"])
        cot_output = cot_chain.predict_and_parse(
            question=state["question"],
            relevant_context=joined
        )
        return {"reasoning": cot_output}

    builder.add_node("ChainOfThought", chain_of_thought)

    def extract_query(state: GraphState) -> dict:
        text = state["reasoning"]
        for line in text.splitlines():
            if line.strip().startswith("QUERY:"):
                return {"prolog_query": line.strip()[len("QUERY:"):].strip()}
        return {"prolog_query": ""}

    builder.add_node("ExtractQuery", extract_query)

    def execute_prolog(state: GraphState) -> dict:
        q = state["prolog_query"]
        if not q:
            return {"query_result": "No query to run."}
        result = run_prolog_query(q)
        return {"query_result": result}

    builder.add_node("ExecuteProlog", execute_prolog)

    builder.add_edge(START, "RetrieveContext")
    builder.add_edge("RetrieveContext", "ChainOfThought")
    builder.add_edge("ChainOfThought", "ExtractQuery")
    builder.add_edge("ExtractQuery", "ExecuteProlog")
    builder.add_edge("ExecuteProlog", END)

    return builder.compile()

if __name__ == "__main__":
    import sys
    user_question = sys.argv[1] if len(sys.argv) > 1 else "Which classes can be scheduled in roomA?"
    graph = build_graph()
    initial_state: GraphState = {
        "question": user_question,
        "relevant_context": [],
        "reasoning": "",
        "prolog_query": "",
        "query_result": ""
    }
    final_state = graph.invoke(initial_state)
    print("=== Reasoning (LLM CoT) ===\n")
    print(final_state["reasoning"], "\n")
    print("=== Extracted Prolog Query ===\n")
    print(final_state["prolog_query"], "\n")
    print("=== Prolog Result ===\n")
    print(final_state["query_result"])

