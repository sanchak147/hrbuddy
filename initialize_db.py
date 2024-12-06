from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from PyPDF2 import PdfReader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

def initialize_vector_store():
    try:
        print("Starting vector store initialization...")
        
        # Define the embedding model
        print("Loading embedding model...")
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        embedding_function = HuggingFaceEmbeddings(model_name=model_name)
        print("Embedding model loaded successfully")

        # Function to load PDFs
        def load_pdfs(directory):
            print(f"Loading PDFs from directory: {directory}")
            pdf_content = {}
            if not os.path.exists(directory):
                raise FileNotFoundError(f"Directory not found: {directory}")
            
            files = os.listdir(directory)
            if not any(f.endswith('.pdf') for f in files):
                raise ValueError(f"No PDF files found in {directory}")
                
            for filename in files:
                if filename.endswith('.pdf'):
                    print(f"Processing file: {filename}")
                    file_path = os.path.join(directory, filename)
                    try:
                        reader = PdfReader(file_path)
                        text = ''.join([page.extract_text() for page in reader.pages])
                        if text.strip():  # Check if extracted text is not empty
                            pdf_content[os.path.splitext(filename)[0]] = text
                            print(f"Successfully extracted {len(text)} characters from {filename}")
                        else:
                            print(f"Warning: No text extracted from {filename}")
                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")
            
            print(f"Loaded {len(pdf_content)} PDF files")
            return pdf_content

        # Load PDFs
        pdf_directory ="/var/www/itc-poc/app/data/pdf"
        pdf_content = load_pdfs(pdf_directory)
        
        if not pdf_content:
            raise ValueError("No PDF content was loaded")

        # Combine all text
        print("Combining text from all PDFs...")
        all_text = "\n\n".join(pdf_content.values())
        print(f"Combined text length: {len(all_text)} characters")

        # Text splitting
        print("Splitting text into chunks...")
        character_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " ", ""],
            chunk_size=500,
            chunk_overlap=20
        )
        character_splitter_texts = character_splitter.split_text(all_text)
        print(f"Created {len(character_splitter_texts)} text chunks")
        
        if not character_splitter_texts:
            raise ValueError("No text chunks were created")

        # Initialize Chroma vector store
        print("Initializing Chroma vector store...")
        persist_directory = './itc_poc_db'
        
        # Clear existing data if any
        if os.path.exists(persist_directory):
            print(f"Removing existing vector store at {persist_directory}")
            import shutil
            shutil.rmtree(persist_directory)
        
        vectorstore = Chroma.from_texts(
            texts=character_splitter_texts,
            collection_name='itc_poc_policies',
            embedding=embedding_function,
            persist_directory=persist_directory
        )

        # Verify the vector store
        print("Verifying vector store...")
        test_query = "test query"
        results = vectorstore.similarity_search(test_query, k=1)
        print(f"Verification search returned {len(results)} results")

        # Persist the vector store
        print("Persisting vector store...")
        vectorstore.persist()
        print(f'Chroma DB created and persisted successfully in {persist_directory}')
        
        return True

    except Exception as e:
        print(f"Error during vector store initialization: {str(e)}")
        return False

if __name__ == "__main__":
    success = initialize_vector_store()
    if not success:
        print("Failed to initialize vector store")
    else:
        print("Vector store initialization completed successfully")
