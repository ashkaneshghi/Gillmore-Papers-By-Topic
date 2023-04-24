import streamlit as st
from streamlit_chat import message

from langchain.embeddings.base import Embeddings
from typing import List
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS

class LocalHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_id): 
        # Should use the GPU by default
        self.model = SentenceTransformer(model_id)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using a locally running
           Hugging Face Sentence Transformer model
        Args:
            texts: The list of texts to embed.
        Returns:
            List of embeddings, one for each text.
        """
        embeddings =self.model.encode(texts)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using a locally running HF 
        Sentence trnsformer. 
        Args:
            text: The text to embed.
        Returns:
            Embeddings for the text.
        """
        embedding = self.model.encode(text)
        return list(map(float, embedding))

embeddings = LocalHuggingFaceEmbeddings('all-mpnet-base-v2')

vector_store = FAISS.load_local("GillPapers_index", embeddings)

def find_similar_n(prompt,n=3):
    docs = vector_store.similarity_search(prompt, k=n)
    # meta1 = docs[0].metadata
    # meta2 = docs[1].metadata
    # meta3 = docs[2].metadata
    # meta = [meta1,meta2,meta3]
    # return docs
    for i in range(0,n):
        meta = docs[i].metadata
        title = meta['TITLE']
        date = meta['Publication_date']
        doi = meta['DOI']
        URL = meta['WARWICK_URL']
        st.write(f'{i+1}.  {title}')
        st.write(f'Publication Date: {date}')
        st.write(f'DOI: {doi}')
        st.write(f'WARWICK URL: {URL}')
        st.write('')

st.set_page_config(page_title='Gillmore', layout='wide')
st.title('Gillmore')
st.markdown(" > Powered by -  Gillmore Centre for Financial Technology ")
# image = Image.open('Sidebar.jpg')
# st.sidebar.image(image)

# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def get_text():
    input_text = st.text_input("Topic: ","", key="input")
    return input_text

def get_num():
    num = st.number_input("Number of  Outputs: ", min_value=1, max_value=10, value=3, step=1)
    return num

user_input = get_text()
num = get_num()

if user_input:
    find_similar_n(user_input,num)
    
    
#     st.session_state.past.append(user_input)
#     st.session_state.generated.append(output1)

# with st.expander("", expanded=True):   
#     for i in range(len(st.session_state['generated'])-1, -1, -1):
#         st.info(st.session_state["past"][i])
#         st.success(st.session_state["generated"][i])