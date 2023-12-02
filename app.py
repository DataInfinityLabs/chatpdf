import gradio as gr
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
import os

def loading_pdf():
    return "Loading..."

def pdf_changes(pdf_doc, repo_id, api_token):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_token

    if api_token == "":
        return "API Key required!"

    loader = PyPDFLoader(pdf_doc)
    # Split the PDF into Pages
    pages = loader.load_and_split()

    # Define chunk size, overlap, and separators
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=64,
        separators=['\n\n', '\n', '(?=>\\. )', ' ', '']
    )
    docs = text_splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings()

    global db
    global chain

    db = FAISS.from_documents(docs, embeddings) 
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":1, "max_length":1000000})
    chain = load_qa_chain(llm, chain_type="stuff")
    
    return "Ready"

def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def bot(history):
    response = infer(history[-1][0])
    history[-1][1] = response
    return history

def infer(question):
    docs = db.similarity_search(question)
    result = chain.run(input_documents=docs, question=question)
    return result

css="""
#col-container {max-width: 700px; margin-left: auto; margin-right: auto;}
"""

title = """
<div style="text-align: center;max-width: 700px;">
    <h1>Chat with PDF</h1>
    <p style="text-align: center;">Upload a .PDF from your computer, click the "Load PDF to LangChain" button, <br />
    when everything is ready, you can start asking questions about the pdf ;)</p>
</div>
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML(title)
        
        with gr.Column():
            pdf_doc = gr.File(label="Load a pdf", file_types=['.pdf'], type="filepath")
            repo_id = gr.Dropdown(label="LLM", choices=["google/flan-ul2", "OpenAssistant/oasst-sft-1-pythia-12b", "bigscience/bloomz"], value="google/flan-ul2")
            api_token = gr.Textbox(label="Hugging Face API Token", placeholder="Enter your Hugging Face API Token")
            with gr.Row():
                langchain_status = gr.Textbox(label="Status", placeholder="", interactive=False)
                load_pdf = gr.Button("Load pdf to Langchain")
        
        chatbot = gr.Chatbot([], elem_id="chatbot")
        question = gr.Textbox(label="Question", placeholder="Type your question and hit Enter ")
        submit_btn = gr.Button("Send message")
    
    repo_id.change(pdf_changes, inputs=[pdf_doc, repo_id, api_token], outputs=[langchain_status], queue=False)
    load_pdf.click(pdf_changes, inputs=[pdf_doc, repo_id, api_token], outputs=[langchain_status], queue=False)
    question.submit(add_text, [chatbot, question], [chatbot, question]).then(
        bot, chatbot, chatbot
    )
    submit_btn.click(add_text, [chatbot, question], [chatbot, question]).then(
        bot, chatbot, chatbot
    )

demo.launch(debug=True)
