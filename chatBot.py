import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

#set openAi API key
os.environ['OPENAI_API_KEY'] = ""

#Initialize the chatOpenAI Model
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

documents = [
    Document(
    page_content="The global financial markets are experiencing volatility as investors react to new inflation data and shifts in central bank policies.",
    metadata={"source": "financial news"},
    ),
    Document(
        page_content="NASA's James Webb Space Telescope continues to deliver breathtaking images of distant galaxies, providing new insights into the early universe.",
        metadata={"source": "space exploration news"},
    ),
    Document(
        page_content="A major breakthrough in renewable energy technology promises more efficient solar panels, potentially accelerating the transition away from fossil fuels.",
        metadata={"source": "environmental news"},
    ),
    Document(
        page_content="The entertainment world is abuzz with the announcement of several high-profile movie sequels and prequels set for release next year.",
        metadata={"source": "entertainment news"},
    ),
    Document(
        page_content="Health experts are advising the public on new dietary guidelines aimed at reducing chronic diseases and promoting overall wellness.",
        metadata={"source": "health and wellness news"},
    ),
]

#Create a vector store using the documents and embedding model
vectorstore = Chroma.from_documents(
    documents,
    embedding=embedding_model
)

#Similarity Search
results = vectorstore.similarity_search("wellness")

# for result in results:
#     print("---------------------------")
#     print(result.page_content)
#     print(result.metadata)
#     print("---------------------------")


#embed a query using the embedding model
# query_embedding = embedding_model.embed_query("wellness")


#print the length of the query embedding
# print(len(query_embedding))

#Similarity Search using embedding
# results = vectorstore.similarity_search_by_vector(query_embedding)

# for result in results:
#     print("-----111111111111111----------------------")
#     print(result.page_content)
#     print(result.metadata)
#     print("---------------------------")


# create a retriver from the vector
retriver = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":1},
)

# perform batch retrival using the retriver
# batch_result = retriver.batch(["entertainment","financial"])

# for result in batch_result:
#     print("-----------------------")
#     for doc in result:
#         print(doc.page_content)
#         print(doc.metadata)

# Define a message template for the chatbot
message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""

#create a chat prompt template from the message
prompt = ChatPromptTemplate.from_messages([("human", message)])

chain = {"context" : retriver, "question" : RunnablePassthrough()} | prompt | llm

response = chain.invoke("current state of 2025")

print(response.content)