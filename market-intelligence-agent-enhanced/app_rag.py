import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from datetime import datetime, timedelta
import json
import uuid
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    CSVLoader, 
    PyPDFLoader, 
    TextLoader, 
    JSONLoader
)
from langchain.document_loaders.blob_loaders import Blob
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Gemini
import requests
import pandas as pd
import tempfile
import shutil
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Market Intelligence Agent")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Secret key for JWT token
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("SECRET_KEY environment variable not set")

# API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize vector database
PERSIST_DIRECTORY = os.path.join(os.getcwd(), "chroma_db")
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

# Initialize embedding function
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize Chroma client
client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

# Create collections for different types of data
try:
    user_files_collection = client.get_or_create_collection(
        name="user_files",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )
    market_data_collection = client.get_or_create_collection(
        name="market_data",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )
    competitor_data_collection = client.get_or_create_collection(
        name="competitor_data",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )
    customer_data_collection = client.get_or_create_collection(
        name="customer_data",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )
except Exception as e:
    logger.error(f"Error initializing ChromaDB collections: {e}")
    # Create a fallback in-memory database if persistent fails
    client = chromadb.Client()
    user_files_collection = client.create_collection(
        name="user_files",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )
    market_data_collection = client.create_collection(
        name="market_data",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )
    competitor_data_collection = client.create_collection(
        name="competitor_data",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )
    customer_data_collection = client.create_collection(
        name="customer_data",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )

# Initialize Gemini LLM if API key is available
llm = None
if GEMINI_API_KEY:
    try:
        llm = Gemini(model="gemini-pro", google_api_key=GEMINI_API_KEY)
    except Exception as e:
        logger.error(f"Error initializing Gemini LLM: {e}")

# Text splitter for document chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

# Create a temporary directory for file uploads
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Pydantic models
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = None

class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    content_type: str
    size: int
    upload_time: str

class AnalysisRequest(BaseModel):
    query: str
    market_domain: str
    file_ids: Optional[List[str]] = None
    include_competitor_analysis: bool = True
    include_market_trends: bool = True
    include_customer_insights: bool = True

class ChatMessage(BaseModel):
    message: str
    analysis_type: str
    analysis_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    timestamp: str
    sources: Optional[List[str]] = None

# File processing functions
async def process_file(file_path: str, file_id: str, content_type: str, filename: str):
    """Process uploaded file and add to vector database"""
    try:
        documents = []
        metadata = {"source": filename, "file_id": file_id}
        
        # Process based on file type
        if content_type == "text/csv":
            loader = CSVLoader(file_path)
            documents = loader.load()
        elif content_type == "application/pdf":
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        elif content_type == "application/json":
            # Custom JSON loading logic
            with open(file_path, 'r') as f:
                data = json.load(f)
            # Convert JSON to text for embedding
            text = json.dumps(data, indent=2)
            documents = text_splitter.create_documents([text], [metadata])
        elif content_type == "text/plain":
            loader = TextLoader(file_path)
            documents = loader.load()
        elif content_type.startswith("audio/") or content_type.startswith("video/"):
            # For audio/video, we'd need transcription services
            # This is a placeholder - in a real implementation, you'd use a service like Whisper API
            metadata["note"] = "Audio/video file - transcription required"
            documents = text_splitter.create_documents(
                ["Audio/video file uploaded: " + filename], [metadata]
            )
        else:
            # Default handling for other file types
            metadata["note"] = f"Unsupported file type: {content_type}"
            documents = text_splitter.create_documents(
                [f"File uploaded: {filename} (type: {content_type})"], [metadata]
            )
        
        # Split documents into chunks if they're not already chunked
        if documents and not isinstance(documents[0], dict):
            documents = text_splitter.split_documents(documents)
        
        # Add to vector database
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [f"{file_id}-chunk-{i}" for i in range(len(texts))]
        
        user_files_collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        return len(texts)
    except Exception as e:
        logger.error(f"Error processing file {filename}: {e}")
        raise

# RAG system functions
def create_rag_chain(collection_name: str, query: str):
    """Create a RAG chain for answering queries"""
    if not llm:
        raise ValueError("LLM not initialized. Check Gemini API key.")
    
    # Get the appropriate collection
    if collection_name == "user_files":
        collection = user_files_collection
    elif collection_name == "market_data":
        collection = market_data_collection
    elif collection_name == "competitor_data":
        collection = competitor_data_collection
    elif collection_name == "customer_data":
        collection = customer_data_collection
    else:
        raise ValueError(f"Unknown collection: {collection_name}")
    
    # Create a Chroma vector store from the collection
    vectorstore = Chroma(
        client=client,
        collection_name=collection.name,
        embedding_function=embedding_function
    )
    
    # Create a retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # Create a prompt template
    template = """
    You are a market intelligence assistant that provides accurate and helpful information.
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Create the chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    # Run the chain
    result = chain({"query": query})
    
    # Extract sources
    sources = []
    if "source_documents" in result:
        for doc in result["source_documents"]:
            if "source" in doc.metadata and doc.metadata["source"] not in sources:
                sources.append(doc.metadata["source"])
    
    return {
        "answer": result["result"],
        "sources": sources
    }

# API endpoints
@app.post("/upload-file", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a file for analysis"""
    try:
        # Generate a unique file ID
        file_id = str(uuid.uuid4())
        
        # Create file path
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
        
        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the file
        await process_file(file_path, file_id, file.content_type, file.filename)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        return FileUploadResponse(
            file_id=file_id,
            filename=file.filename,
            content_type=file.content_type,
            size=file_size,
            upload_time=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/comprehensive-analysis")
async def run_comprehensive_analysis(request: AnalysisRequest):
    """Run a comprehensive market intelligence analysis"""
    try:
        # Generate a unique analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Start with basic response structure
        response = {
            "analysis_id": analysis_id,
            "status": "processing"
        }
        
        # In a real implementation, this would be an async task
        # For now, we'll simulate a synchronous response
        
        # Collect data from various sources based on the query and domain
        results = {}
        
        # 1. Competitor Analysis
        if request.include_competitor_analysis:
            competitors = get_competitors(request.query, request.market_domain)
            results["competitor_analysis"] = {
                "competitors": competitors,
                "summary": f"Competitor analysis for {request.market_domain} market related to '{request.query}'"
            }
        
        # 2. Market Trends
        if request.include_market_trends:
            trends = get_market_trends(request.query, request.market_domain)
            results["market_trends"] = {
                "trends": trends,
                "summary": f"Market trends for {request.market_domain} market related to '{request.query}'"
            }
        
        # 3. Customer Insights
        if request.include_customer_insights:
            segments = get_customer_segments(request.query, request.market_domain)
            results["customer_insights"] = {
                "segments": segments,
                "summary": f"Customer insights for {request.market_domain} market related to '{request.query}'"
            }
        
        # 4. Generate overall summary using Gemini if available
        summary = f"Analysis of {request.query} in the {request.market_domain} market"
        if llm:
            try:
                prompt = f"""
                Generate a concise summary of market intelligence findings for the query: "{request.query}" 
                in the {request.market_domain} domain.
                
                Competitor information: {json.dumps(results.get('competitor_analysis', {}))}
                Market trends: {json.dumps(results.get('market_trends', {}))}
                Customer insights: {json.dumps(results.get('customer_insights', {}))}
                
                Provide a 2-3 sentence executive summary highlighting the most important insights.
                """
                summary_result = llm(prompt)
                summary = summary_result.strip()
            except Exception as e:
                logger.error(f"Error generating summary with Gemini: {e}")
        
        # Complete response
        response = {
            "analysis_id": analysis_id,
            "status": "completed",
            "summary": summary,
            **results
        }
        
        # Store the analysis results for later retrieval
        # In a real implementation, this would be stored in a database
        # For now, we'll simulate by returning the complete results
        
        return response
    except Exception as e:
        logger.error(f"Error running analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatMessage):
    """Chat with the market intelligence agent"""
    try:
        # Determine which collections to query based on analysis type
        collections = ["user_files"]  # Always include user files
        
        if request.analysis_type == "competitor":
            collections.append("competitor_data")
        elif request.analysis_type == "market":
            collections.append("market_data")
        elif request.analysis_type == "customer":
            collections.append("customer_data")
        elif request.analysis_type == "general":
            collections.extend(["competitor_data", "market_data", "customer_data"])
        
        # Collect answers from each relevant collection
        all_answers = []
        all_sources = []
        
        for collection_name in collections:
            try:
                result = create_rag_chain(collection_name, request.message)
                all_answers.append(result["answer"])
                all_sources.extend(result["sources"])
            except Exception as e:
                logger.error(f"Error querying {collection_name}: {e}")
        
        # If we couldn't get answers from RAG, use Gemini directly
        if not all_answers and llm:
            try:
                prompt = f"""
                You are a market intelligence assistant. Answer the following question 
                about {request.analysis_type} analysis:
                
                Question: {request.message}
                
                If you don't know the answer, just say that you don't have enough information.
                """
                response = llm(prompt)
                all_answers.append(response.strip())
            except Exception as e:
                logger.error(f"Error generating response with Gemini: {e}")
        
        # Combine answers or provide a fallback
        if all_answers:
            # In a real implementation, we might use Gemini to synthesize multiple answers
            # For simplicity, we'll just use the first answer
            final_answer = all_answers[0]
        else:
            final_answer = "I'm sorry, I don't have enough information to answer that question."
        
        # Remove duplicates from sources
        unique_sources = list(set(all_sources))
        
        return ChatResponse(
            response=final_answer,
            timestamp=datetime.now().isoformat(),
            sources=unique_sources if unique_sources else None
        )
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api-keys")
async def get_api_keys():
    """Get status of configured API keys"""
    return {
        "google_api_key": bool(GOOGLE_API_KEY),
        "newsapi_key": bool(NEWSAPI_KEY),
        "alpha_vantage_key": bool(ALPHA_VANTAGE_KEY),
        "tavily_api_key": bool(TAVILY_API_KEY),
        "gemini_api_key": bool(GEMINI_API_KEY)
    }

# Helper functions for data retrieval
def get_competitors(query: str, domain: str) -> List[Dict[str, Any]]:
    """Get competitor information from various sources"""
    # In a real implementation, this would query external APIs and databases
    # For now, we'll return sample data
    competitors = [
        {
            "name": "Company A",
            "market_share": "32%",
            "strengths": ["Product Innovation", "Brand Recognition", "Global Reach"],
            "weaknesses": ["High Prices", "Customer Support"]
        },
        {
            "name": "Company B",
            "market_share": "28%",
            "strengths": ["Competitive Pricing", "User Experience", "Marketing"],
            "weaknesses": ["Limited Product Range", "Technical Debt"]
        },
        {
            "name": "Company C",
            "market_share": "15%",
            "strengths": ["Customer Service", "Reliability", "Niche Focus"],
            "weaknesses": ["Limited Resources", "Slow Innovation"]
        }
    ]
    
    # If NewsAPI key is available, try to get real data
    if NEWSAPI_KEY:
        try:
            # Get news about competitors in the domain
            url = f"https://newsapi.org/v2/everything?q={query}+{domain}+competitors&sortBy=relevancy&apiKey={NEWSAPI_KEY}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                # Process news data to extract competitor information
                # This would require NLP processing in a real implementation
                # For now, we'll just log that we got data
                logger.info(f"Retrieved {len(data.get('articles', []))} news articles about competitors")
        except Exception as e:
            logger.error(f"Error fetching competitor news: {e}")
    
    return competitors

def get_market_trends(query: str, domain: str) -> List[Dict[str, Any]]:
    """Get market trends from various sources"""
    # In a real implementation, this would query external APIs and databases
    # For now, we'll return sample data
    trends = [
        {
            "trend": "AI Integration",
            "impact": "High",
            "timeframe": "Short-term (0-6 months)"
        },
        {
            "trend": "Sustainability Focus",
            "impact": "Medium",
            "timeframe": "Medium-term (6-18 months)"
        },
        {
            "trend": "Remote Work Solutions",
            "impact": "High",
            "timeframe": "Long-term (18+ months)"
        }
    ]
    
    # If Alpha Vantage key is available, try to get real market data
    if ALPHA_VANTAGE_KEY and domain in ["finance", "technology"]:
        try:
            # Get market data for relevant sector
            symbol = "SPY"  # Default to S&P 500
            if domain == "technology":
                symbol = "QQQ"  # NASDAQ-100
            
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol={symbol}&apikey={ALPHA_VANTAGE_KEY}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                # Process market data to extract trends
                # This would require financial analysis in a real implementation
                # For now, we'll just log that we got data
                logger.info(f"Retrieved market data for {symbol}")
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
    
    return trends

def get_customer_segments(query: str, domain: str) -> List[Dict[str, Any]]:
    """Get customer segment information from various sources"""
    # In a real implementation, this would query external APIs and databases
    # For now, we'll return sample data
    segments = [
        {
            "segment": "Enterprise",
            "needs": ["Scalability", "Security", "Integration"],
            "pain_points": ["Complex Implementation", "High Costs"],
            "growth_rate": "8% annually"
        },
        {
            "segment": "SMB",
            "needs": ["Affordability", "Ease of Use", "Quick Setup"],
            "pain_points": ["Limited Resources", "Technical Expertise"],
            "growth_rate": "12% annually"
        },
        {
            "segment": "Startups",
            "needs": ["Flexibility", "Modern Features", "Pay-as-you-go"],
            "pain_points": ["Cash Flow", "Rapid Changes"],
            "growth_rate": "15% annually"
        }
    ]
    
    # If Tavily API key is available, try to get real customer data
    if TAVILY_API_KEY:
        try:
            # Search for customer insights
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": TAVILY_API_KEY,
                "query": f"customer segments {domain} {query}",
                "search_depth": "advanced",
                "include_domains": ["statista.com", "forrester.com", "gartner.com"]
            }
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                data = response.json()
                # Process search results to extract customer insights
                # This would require NLP processing in a real implementation
                # For now, we'll just log that we got data
                logger.info(f"Retrieved {len(data.get('results', []))} search results about customer segments")
        except Exception as e:
            logger.error(f"Error searching for customer insights: {e}")
    
    return segments

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
