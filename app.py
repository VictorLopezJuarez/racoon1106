import chainlit as cl
from langchain_openai import OpenAI
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

# Load environment variables from a .env file
load_dotenv()

# Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path = "faiss_index"

def create_vector_db():
    # Load data from FAQ sheet
    loader = CSVLoader(file_path='qanda.csv', source_column="question")
    data = loader.load()
    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)
    # Save vector database locally
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():

    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings, allow_dangerous_deserialization=True)

    # Create a retriever for querying the vector database
    #retriever = vectordb.as_retriever(score_threshold=0.7, top_k=1)  # Limit the number of retrieved documents to 1
    retriever = vectordb.as_retriever(score_threshold=0.5, search_kwargs={'k': 2})  # Limit the number of retrieved documents to 1
                                        #score_threshold=0.5, on 5 June at 16h01 ranges from 0 to 1, lower value less strict, higher value higher strict.

    # Initialize the language model (llm) using OpenAI
    llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Creating a Prompt/Context to influence the chatbox behavior
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from the "response" section in the source document context without making many changes. You can add or remove words in your answers according to the questions.
    If the answer is not found in the context, kindly state "I'm sorry, I don't have the information you're looking for at the moment. For further assistance, please reach out to us at info-emildai@dcu.ie." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=False,  #switched to False on 5 June at 11h27
                                        chain_type_kwargs={"prompt": PROMPT})
  
    return chain

# Initialize the chain outside of the main function
create_vector_db()
chain = get_qa_chain()

@cl.on_message
async def main(message: cl.Message):
    # Use the chain to answer the user's question
    response = chain.invoke({"query": message.content})
    
    # Extract the result from the response, cleaner responses
    result = response['result'].replace('ANSWER:', '').replace('answer:', '').replace('Answer:', '').replace('context', '').replace('?', '').strip()

    # Send the response back to the user
    await cl.Message(
        content=f"{result}",
    ).send()

# Run the app / added 6 June at 11h52
#if __name__ == "__main__":
#   port = int(os.environ.get("PORT", 10000))  # Use the PORT environment variable
#   cl.run(port=port)         # Bind to 0.0.0.0 to accept all incoming connections