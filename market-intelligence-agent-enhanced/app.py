import os
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status, Form, UploadFile, File, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import jwt  # Changed from from jose import jwt
from passlib.context import CryptContext
import json
from market_intelligence_agent_enhanced.supabase_client import (
    create_user as supabase_create_user,
    get_user_by_email as supabase_get_user_by_email,
    save_user_api_key as supabase_save_user_api_key,
    get_all_user_api_keys as supabase_get_all_user_api_keys,
    is_supabase_connected,
    save_analysis_result as supabase_save_analysis_result,
    get_analysis_results_for_user as supabase_get_analysis_results_for_user,
    get_analysis_result_by_id as supabase_get_analysis_result_by_id,
    update_analysis_status as supabase_update_analysis_status
)
import logging # Added for logging
import uuid # Added for file uploads
import shutil # Added for file uploads
import chromadb # Added for ChromaDB
from chromadb.utils import embedding_functions # Added for ChromaDB
from langchain.text_splitter import RecursiveCharacterTextSplitter # Added for text splitting
from langchain.document_loaders import ( # Added for document loading
    CSVLoader, 
    PyPDFLoader, 
    TextLoader, 
    JSONLoader
)
# from langchain.document_loaders.blob_loaders import Blob # Not used in process_file from app_rag
from langchain.embeddings import HuggingFaceEmbeddings # Added for ChromaDB embeddings
from langchain.vectorstores import Chroma # Added for RAG chain
# from langchain.chains import RetrievalQA # Will be replaced by agent for chat
from langchain.prompts import PromptTemplate # Still useful for specific prompts if needed, but agent handles its own
from langchain_community.llms import Gemini # Added for LLM
import requests # Added for mock data functions
from langchain.agents import initialize_agent, AgentType, Tool as LangchainTool # For the new agent
from market_intelligence_agent_enhanced.langchain_tools import all_tools # External API tools
from market_intelligence_agent_enhanced.langgraph_chat import run_chat_graph # Import Langgraph runner


# Load environment variables from .env file
load_dotenv()

# Configure logging (as in app_rag.py)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Market Intelligence Agent API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security configurations
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-for-jwt")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Models
class User(BaseModel):
    id: str # Supabase user ID (UUID)
    email: str
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    # created_at: Optional[datetime] = None # Supabase typically adds this, not needed in Pydantic model for response

class UserInDB(User): # This model now represents the structure from Supabase 'users' table
    hashed_password: str
    # created_at: Optional[datetime] = None # Supabase typically adds this

class Token(BaseModel):
    access_token: str
    token_type: str
    user: User # Will include id, email, full_name

class TokenData(BaseModel): # Remains the same, 'sub' in JWT is email
    email: Optional[str] = None

class ApiKeys(BaseModel):
    google_api_key: Optional[str] = None
    newsapi_key: Optional[str] = None
    alpha_vantage_key: Optional[str] = None
    tavily_api_key: Optional[str] = None

class MarketIntelligenceQuery(BaseModel):
    query: str
    market_domain: str

class SpecificQuestion(BaseModel):
    question: str
    state_id: str

class PasswordReset(BaseModel):
    email: str

# --- Pydantic Models from app_rag.py ---
class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    content_type: str
    size: int
    upload_time: str

class AnalysisRequest(BaseModel): # As per instruction, copied for now
    query: str
    market_domain: str
    file_ids: Optional[List[str]] = None
    include_competitor_analysis: bool = True
    include_market_trends: bool = True
    include_customer_insights: bool = True

class ChatMessage(BaseModel): 
    message: str
    analysis_type: str # Example: 'competitor', 'market', 'customer', 'general' - used by old chat, less relevant for Langgraph agent directly but kept for now
    analysis_id: Optional[str] = None # To link to a specific comprehensive analysis
    session_id: Optional[str] = None # Added for Langgraph chat session management

class ChatResponse(BaseModel): 
    response: str
    timestamp: str
    sources: Optional[List[str]] = None


# --- Text Splitter (from app_rag.py) ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

# --- Global variables for ChromaDB and File Handling (to be initialized in startup_event) ---
PERSIST_DIRECTORY: Optional[str] = None
UPLOAD_DIR: Optional[str] = None
embedding_function: Optional[HuggingFaceEmbeddings] = None # Langchain embedding wrapper
chromadb_ef: Optional[embedding_functions.SentenceTransformerEmbeddingFunction] = None # Chroma native embedding function
chroma_client: Optional[chromadb.Client] = None
user_files_collection: Optional[chromadb.Collection] = None
market_data_collection: Optional[chromadb.Collection] = None
competitor_data_collection: Optional[chromadb.Collection] = None
customer_data_collection: Optional[chromadb.Collection] = None

# --- LLM Instance ---
llm: Optional[Gemini] = None

# --- API Keys from Environment (for system status endpoint and mock data functions) ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Used by app_rag.py's LLM init

# Helper functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

async def get_user_from_supabase(email: str) -> UserInDB | None:
    user_data = await supabase_get_user_by_email(email)
    if user_data:
        # Assuming 'id' is part of user_data from Supabase
        # And 'disabled' field might also exist, or we default it
        return UserInDB(
            id=str(user_data.get("id")), # Ensure ID is string
            email=user_data.get("email"),
            full_name=user_data.get("full_name"),
            hashed_password=user_data.get("hashed_password"),
            disabled=user_data.get("disabled", False) # Default to False if not present
        )
    return None

async def authenticate_user(email: str, password: str) -> UserInDB | None:
    user = await get_user_from_supabase(email)
    if not user or user.disabled: # Also check if user is disabled
        return None # User not found or inactive
    if not verify_password(password, user.hashed_password):
        return None # Password incorrect
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(payload=to_encode, key=SECRET_KEY, algorithm=ALGORITHM)  # Modified
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(jwt=token, key=SECRET_KEY, algorithms=[ALGORITHM])  # Modified
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except jwt.exceptions.PyJWTError:  # Modified
        raise credentials_exception
    user_in_db = await get_user_from_supabase(email=token_data.email)
    if user_in_db is None:
        raise credentials_exception
    # The User model for response should not include hashed_password
    # current_user will be an instance of UserInDB from Supabase
    return user_in_db

async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)): # current_user is UserInDB
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    # Return the UserInDB object, but FastAPI will serialize it based on response_model
    # For /users/me, response_model is User, so hashed_password won't be sent.
    return current_user

# Mock Market Intelligence Agent for demonstration
class MockMarketIntelligenceAgent:
    def process_query(self, query, market_domain):
        # This is a mock implementation
        return {
            "trends": [
                {
                    "trend_name": "AI Integration in SaaS",
                    "description": "Increasing adoption of AI capabilities in SaaS products.",
                    "supporting_evidence": ["Microsoft's integration of GPT-4", "Salesforce Einstein AI adoption up 45% YoY"],
                    "estimated_impact": "High",
                    "timeframe": "Short-term",
                    "confidence_score": 0.89
                },
                {
                    "trend_name": "Vertical SaaS Specialization",
                    "description": "Growth of industry-specific SaaS solutions.",
                    "supporting_evidence": ["Healthcare SaaS funding increased 35%", "Industry-specific solutions showing 2.3x retention rates"],
                    "estimated_impact": "Medium",
                    "timeframe": "Medium-term",
                    "confidence_score": 0.76
                }
            ],
            "opportunities": [
                {
                    "title": "AI-Powered Customer Support",
                    "description": "Implementing AI chatbots for customer support can reduce response time and improve satisfaction.",
                    "potential_impact": "High",
                    "implementation_difficulty": "Medium",
                    "recommended_actions": ["Evaluate AI chatbot providers", "Start with simple use cases", "Gradually expand capabilities"]
                }
            ],
            "risks": [
                {
                    "title": "Increasing Competition",
                    "description": "New entrants with AI-first approaches may disrupt established players.",
                    "severity": "Medium",
                    "likelihood": "High",
                    "mitigation_strategies": ["Accelerate AI adoption", "Focus on unique value proposition", "Strengthen customer relationships"]
                }
            ]
        }
    
    def answer_specific_question(self, question, state_id):
        # This is a mock implementation
        return {
            "answer": "Based on the analysis, the most promising market opportunity is AI-powered customer support solutions, which can reduce response times by up to 60% while improving customer satisfaction scores by an average of 15%.",
            "sources": ["Industry report by Gartner", "Case study from Salesforce implementation"],
            "confidence": 0.87
        }

# Get agent instance
def get_agent(api_keys: ApiKeys = None):
    # In a real implementation, this would use the actual agent with API keys
    return MockMarketIntelligenceAgent()

# Background task for processing market intelligence queries
async def process_market_intelligence_query(query: str, market_domain: str, user_id: str, user_email: str):
    # Get user's API keys from Supabase
    # This part still uses the old model, will need adjustment if api_keys_db is fully removed
    # For now, assuming it's handled by the user_id passed to get_agent
    # user_api_keys_dict = await supabase_get_all_user_api_keys(user_id=user_id)
    # mapped_api_keys = ApiKeys(**user_api_keys_dict) if user_api_keys_dict else ApiKeys()
    
    # Initialize agent - get_agent might need to be async if it fetches keys itself
    # For now, let's assume get_agent is synchronous and takes ApiKeys model
    # This is a temporary simplification:
    user_api_keys_dict = await supabase_get_all_user_api_keys(user_id=user_id)
    mapped_api_keys = ApiKeys(**user_api_keys_dict) if user_api_keys_dict else ApiKeys()
    agent = get_agent(mapped_api_keys)
    
    analysis_id = None # Will be set by Supabase
    try:
        result_data = agent.process_query(query, market_domain)
        # Ensure result_data is JSON serializable. The mock agent returns a dict.
        analysis_id = await supabase_save_analysis_result(
            user_id=user_id,
            query=query,
            market_domain=market_domain,
            result_data=result_data,
            status="completed"
        )
        if analysis_id:
            print(f"Successfully saved analysis result {analysis_id} for user {user_id}")
        else:
            print(f"Failed to save analysis result for user {user_id}")

    except Exception as e:
        print(f"Error processing market intelligence query for user {user_id}: {e}")
        # Attempt to save error information
        error_analysis_id = await supabase_save_analysis_result(
            user_id=user_id,
            query=query,
            market_domain=market_domain,
            result_data={"error_detail": str(e)}, # Store error message in result_data
            status="failed",
            error_message=str(e)
        )
        if error_analysis_id:
            print(f"Successfully saved error analysis {error_analysis_id} for user {user_id}")
        else:
            print(f"Failed to save error analysis for user {user_id}")

# Routes
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate_user(email=form_data.username, password=form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password, or user disabled",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires # 'sub' is email
    )
    # User object for the token response
    token_user = User(id=str(user.id), email=user.email, full_name=user.full_name, disabled=user.disabled)
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": token_user
    }

@app.post("/users", response_model=User)
async def create_new_user(email: str = Form(...), password: str = Form(...), full_name: Optional[str] = Form(None)):
    existing_user = await supabase_get_user_by_email(email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    hashed_password = get_password_hash(password)
    
    # Call Supabase client's create_user function
    # Ensure your supabase_client.create_user returns a dict that matches UserInDB or can be mapped to it
    created_user_data = await supabase_create_user(email=email, hashed_password=hashed_password, full_name=full_name)
    
    if not created_user_data:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create user in database.",
        )
    
    # Construct the User response model
    # created_user_data is a dict from Supabase, should include 'id', 'email', 'full_name', 'disabled'
    return User(
        id=str(created_user_data.get("id")),
        email=created_user_data.get("email"),
        full_name=created_user_data.get("full_name"),
        disabled=created_user_data.get("disabled", False)
    )

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: UserInDB = Depends(get_current_active_user)):
    # current_user is UserInDB (from Supabase), FastAPI serializes to User model
    return current_user

@app.post("/reset-password")
async def reset_password(reset_data: PasswordReset):
    # In a real app, this would send an email with reset instructions
    # For this demo, we'll just print a message
    print(f"Password reset requested for {reset_data.email}")
    return {"message": "If your email is registered, you will receive password reset instructions"}

@app.get("/api-keys", response_model=ApiKeys)
async def get_user_api_keys_from_db(current_user: UserInDB = Depends(get_current_active_user)):
    # current_user from get_current_active_user is UserInDB, which has the 'id'
    user_id = str(current_user.id)
    keys_dict = await supabase_get_all_user_api_keys(user_id=user_id)
    if keys_dict is None: # Error occurred in supabase_client
        raise HTTPException(status_code=500, detail="Could not fetch API keys from database.")
    # Map dict to ApiKeys Pydantic model. Missing keys will be None by default.
    return ApiKeys(**keys_dict)

@app.post("/api-keys")
async def set_user_api_keys_in_db(api_keys_model: ApiKeys, current_user: UserInDB = Depends(get_current_active_user)):
    user_id = str(current_user.id)
    success_all = True
    
    # Iterate through the fields of the Pydantic model
    for service_name, key_value in api_keys_model.model_dump().items():
        if key_value is not None: # Only save if a key is provided
            # service_name will be e.g. "google_api_key"
            # key_value is the actual API key string
            success = await supabase_save_user_api_key(user_id=user_id, service_name=service_name, api_key=key_value)
            if not success:
                success_all = False
                # Log or handle partial failure if needed
                print(f"Failed to save API key for service: {service_name}") # Basic logging

    if success_all:
        return {"message": "API keys updated successfully"}
    else:
        # If any key failed to save, return a more informative error.
        # This simple message can be improved to list which keys failed.
        raise HTTPException(status_code=500, detail="Failed to update one or more API keys.")

@app.post("/market-intelligence")
async def market_intelligence(
    query: MarketIntelligenceQuery, 
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user)
):
    # Add task to background processing
    background_tasks.add_task(
        process_market_intelligence_query,
        query.query,
        query.market_domain,
        str(current_user.id), # Pass user_id
        current_user.email
    )

    return {
        "message": "Query processing started",
        "status": "processing",
        "query": query.query,
        "market_domain": query.market_domain
    }

@app.post("/specific-question")
async def specific_question(
    question: SpecificQuestion,
    current_user: User = Depends(get_current_active_user)
):
    # Get user's API keys from Supabase
    user_api_keys_dict = await supabase_get_all_user_api_keys(user_id=str(current_user.id))
    mapped_api_keys = ApiKeys(**user_api_keys_dict) if user_api_keys_dict else ApiKeys()

    # Initialize agent
    agent = get_agent(mapped_api_keys)
    
    # Process specific question
    try:
        result = agent.answer_specific_question(question.question, question.state_id)
        return {
            "answer": result,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/analysis-results", response_model=List[Dict[str, Any]]) # Adjust response_model as needed
async def get_all_analysis_results(
    current_user: UserInDB = Depends(get_current_active_user),
    limit: int = 20, 
    offset: int = 0
):
    user_id = str(current_user.id)
    results = await supabase_get_analysis_results_for_user(user_id=user_id, limit=limit, offset=offset)
    if results is None: # Indicates an error during fetch
        raise HTTPException(status_code=500, detail="Could not fetch analysis results.")
    return results

@app.get("/analysis-results/{result_id}", response_model=Dict[str, Any]) # Adjust response_model
async def get_single_analysis_result(
    result_id: str, # Result ID is now a string (UUID from Supabase)
    current_user: UserInDB = Depends(get_current_active_user)
):
    user_id = str(current_user.id)
    result = await supabase_get_analysis_result_by_id(user_id=user_id, result_id=result_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis result with ID {result_id} not found or access denied."
        )
    return result

# Serve static files for the frontend
FRONTEND_BUILD_DIR = os.path.abspath("frontend/market-intel-ui/dist")
app.mount("/", StaticFiles(directory=FRONTEND_BUILD_DIR, html=True), name="static")

# Add a test user for development (optional, consider if needed with Supabase)
@app.on_event("startup")
async def startup_event():
    if await is_supabase_connected():
        print("Supabase connection check successful on startup.")
    else:
        print("Warning: Supabase connection check failed on startup.")
    
    global PERSIST_DIRECTORY, UPLOAD_DIR
    global embedding_function, chromadb_ef, chroma_client # Chroma related
    global user_files_collection, market_data_collection, competitor_data_collection, customer_data_collection # Collections
    global llm, user_documents_tool # LLM instance and new RAG tool

    # Initialize directories (already done in previous merge, ensure vars are global)
    PERSIST_DIRECTORY = os.path.join(os.getcwd(), "chroma_db_app")
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    UPLOAD_DIR = os.path.join(os.getcwd(), "uploads_app") # Renamed
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    logger.info(f"ChromaDB persist directory: {PERSIST_DIRECTORY}")
    logger.info(f"Uploads directory: {UPLOAD_DIR}")

    # Initialize embedding function
    try:
        embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        logger.info("HuggingFace embedding function initialized.")
    except Exception as e:
        logger.error(f"Error initializing HuggingFaceEmbeddings: {e}")
        # Handle case where embeddings can't be initialized (e.g. no internet, model not found)
        # The app might need to operate in a degraded mode or fail startup.
        # For now, functions relying on it will fail if it's None.

    # Initialize Chroma client and collections
    if embedding_function is not None:
        try:
            chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
            logger.info("ChromaDB persistent client initialized.")

            # User files collection (primary for uploads)
            # The embedding_function parameter for get_or_create_collection should be an instance of chromadb.api.types.EmbeddingFunction
            # not a LangChain Embedding object directly for chromadb versions < 0.4.0
            # For chromadb >= 0.4.0, it can handle LangChain embeddings or SentenceTransformerEmbeddingFunction
            # Assuming chromadb version used by app_rag.py (0.4.22) which supports SentenceTransformerEmbeddingFunction
            
            # This is how app_rag.py did it, which is compatible with chromadb 0.4.x
            chromadb_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            user_files_collection = chroma_client.get_or_create_collection(
                name="user_files_app", # Renamed
                embedding_function=chromadb_ef # Use the chromadb specific function wrapper
            )
            logger.info(f"ChromaDB collection '{user_files_collection.name}' loaded/created.")
            
            # This is how app_rag.py did it, which is compatible with chromadb 0.4.x
            chromadb_ef = embedding_functions.SentenceTransformerEmbeddingFunction( # Initialize it here
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            user_files_collection = chroma_client.get_or_create_collection(
                name="user_files_app", 
                embedding_function=chromadb_ef 
            )
            logger.info(f"ChromaDB collection '{user_files_collection.name}' loaded/created.")
            
            # Initialize other collections as defined in app_rag.py
            market_data_collection = chroma_client.get_or_create_collection(
                name="market_data_app", 
                embedding_function=chromadb_ef
            )
            logger.info(f"ChromaDB collection '{market_data_collection.name}' loaded/created.")
            
            competitor_data_collection = chroma_client.get_or_create_collection(
                name="competitor_data_app", 
                embedding_function=chromadb_ef
            )
            logger.info(f"ChromaDB collection '{competitor_data_collection.name}' loaded/created.")

            customer_data_collection = chroma_client.get_or_create_collection(
                name="customer_data_app", 
                embedding_function=chromadb_ef
            )
            logger.info(f"ChromaDB collection '{customer_data_collection.name}' loaded/created.")

        except Exception as e:
            logger.error(f"Error initializing ChromaDB client or collections: {e}")
            if embedding_function: 
                logger.warning("Falling back to in-memory ChromaDB client.")
                chroma_client = chromadb.Client() 
                if chromadb_ef is None: # Ensure chromadb_ef is initialized for in-memory fallback as well
                     chromadb_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-MiniLM-L6-v2")

                user_files_collection = chroma_client.get_or_create_collection(name="user_files_app_memory", embedding_function=chromadb_ef)
                logger.info(f"In-memory ChromaDB collection '{user_files_collection.name}' created.")
                market_data_collection = chroma_client.get_or_create_collection(name="market_data_app_memory", embedding_function=chromadb_ef)
                logger.info(f"In-memory ChromaDB collection '{market_data_collection.name}' created.")
                competitor_data_collection = chroma_client.get_or_create_collection(name="competitor_data_app_memory", embedding_function=chromadb_ef)
                logger.info(f"In-memory ChromaDB collection '{competitor_data_collection.name}' created.")
                customer_data_collection = chroma_client.get_or_create_collection(name="customer_data_app_memory", embedding_function=chromadb_ef)
                logger.info(f"In-memory ChromaDB collection '{customer_data_collection.name}' created.")
            else:
                logger.error("Cannot initialize ChromaDB without embedding function.")
    
    # Initialize User Documents Tool (RAG Retriever Tool)
    if chroma_client and user_files_collection and embedding_function:
        try:
            user_files_vectorstore = Chroma(
                client=chroma_client,
                collection_name=user_files_collection.name, # Use the .name attribute
                embedding_function=embedding_function # Langchain wrapper
            )
            user_files_retriever = user_files_vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3} 
            )

            def query_user_documents(query: str) -> str:
                """Searches and retrieves relevant information from documents uploaded by the user."""
                logger.info(f"Querying user documents with: {query}")
                if not user_files_retriever:
                    return "User documents retriever is not available."
                try:
                    docs = user_files_retriever.get_relevant_documents(query)
                    if not docs:
                        return "No relevant documents found for your query in uploaded files."
                    return "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" for doc in docs])
                except Exception as e_retrieval:
                    logger.error(f"Error during document retrieval: {e_retrieval}", exc_info=True)
                    return "Error retrieving documents."

            user_documents_tool = LangchainTool( # Renamed to avoid conflict with FastAPI's Tool
                name="UserUploadedDocumentsSearch",
                func=query_user_documents, # This function should be synchronous for initialize_agent
                description="Searches and retrieves relevant information from documents uploaded by the user. Use this to answer questions based on user-provided files. Input is the search query."
            )
            logger.info("User documents RAG tool initialized successfully.")
        except Exception as e_tool_init:
            logger.error(f"Failed to initialize user_documents_tool: {e_tool_init}", exc_info=True)
            user_documents_tool = None # Ensure it's None if failed
    else:
        logger.warning("Chroma client, user_files_collection, or embedding_function not available. User documents tool not initialized.")
        user_documents_tool = None


    # Initialize Gemini LLM
    if GEMINI_API_KEY:
        try:
            llm = Gemini(model="gemini-pro", google_api_key=GEMINI_API_KEY)
            logger.info("Gemini LLM initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing Gemini LLM: {e}. LLM functionalities will be unavailable.")
            llm = None # Ensure llm is None if initialization fails
    else:
        logger.warning("GEMINI_API_KEY not found. LLM functionalities will be unavailable.")
        llm = None

    # Optionally, create a default user if it doesn't exist (be cautious with this in production)
    # Example:
    # test_email = "test@example.com"
    # if not await supabase_get_user_by_email(test_email):
    #     print(f"Creating default user {test_email}")
    #     hashed_password = get_password_hash("password")
    #     await supabase_create_user(test_email, hashed_password, "Test User")
    # else:
    #     print(f"Default user {test_email} already exists.")


# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# --- File Processing Function (from app_rag.py, adapted) ---
async def process_uploaded_file(
    file_path: str, 
    file_id: str, # This is the UUID generated for the file
    content_type: str, 
    filename: str,
    # user_id: str # Optional: if files should be associated with a user in Chroma metadata
):
    """Process uploaded file and add to user_files_collection vector database"""
    global user_files_collection, text_splitter # Ensure access to global vars

    if user_files_collection is None:
        logger.error("user_files_collection is not initialized. Cannot process file.")
        raise HTTPException(status_code=500, detail="Vector database not available.")
    if text_splitter is None: # Should be initialized globally
        logger.error("text_splitter is not initialized. Cannot process file.")
        raise HTTPException(status_code=500, detail="Text splitter not available.")

    try:
        documents = []
        # Base metadata for all chunks from this file
        # Add user_id to metadata if you want to scope documents by user later
        # metadata = {"source": filename, "file_id": file_id, "user_id": user_id}
        metadata = {"source": filename, "file_id": file_id}
        
        logger.info(f"Processing file {filename} (ID: {file_id}) of type {content_type}")

        if content_type == "text/csv":
            loader = CSVLoader(file_path)
            documents = loader.load()
        elif content_type == "application/pdf":
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        elif content_type == "application/json":
            # Custom JSON loading logic from app_rag.py
            with open(file_path, 'r') as f:
                data = json.load(f)
            text_content = json.dumps(data, indent=2) # Convert JSON to text for embedding
            # Create Langchain Document objects, applying metadata to each
            # documents = [Document(page_content=text_content, metadata=metadata)] # If not splitting JSON further
            # If splitting large JSON text:
            documents = text_splitter.create_documents([text_content], metadatas=[metadata])
        elif content_type == "text/plain":
            loader = TextLoader(file_path)
            documents = loader.load()
        elif content_type.startswith("audio/") or content_type.startswith("video/"):
            metadata["note"] = "Audio/video file - transcription required (feature not implemented)"
            logger.warning(f"Audio/video file {filename} uploaded. Transcription not implemented.")
            documents = text_splitter.create_documents(
                ["Audio/video file uploaded: " + filename], metadatas=[metadata]
            )
        else:
            metadata["note"] = f"Unsupported file type: {content_type}"
            logger.warning(f"Unsupported file type {content_type} for file {filename}.")
            documents = text_splitter.create_documents(
                [f"File uploaded: {filename} (type: {content_type})"], metadatas=[metadata]
            )
        
        # Chunk documents if not already chunked by loader (some loaders might chunk)
        # The JSON loader example above uses text_splitter.create_documents which chunks.
        # CSVLoader, PyPDFLoader, TextLoader typically return one Document per file or page.
        # We need to ensure all documents are split.
        
        final_texts: List[str] = []
        final_metadatas: List[Dict[str, Any]] = []

        if documents:
            # If loader returns list of Langchain Document objects
            if not isinstance(documents[0], str): # Check if it's not already just text
                # Apply the common metadata to each document from the loader if it doesn't have one
                for doc in documents:
                    doc.metadata.update(metadata) # Add/override common metadata
                
                # Split documents further if necessary
                # Some loaders (like PyPDFLoader) might already split by page.
                # We apply RecursiveCharacterTextSplitter for finer-grained chunks.
                split_documents = text_splitter.split_documents(documents)
                final_texts = [doc.page_content for doc in split_documents]
                final_metadatas = [doc.metadata for doc in split_documents]
            else: # If documents is already a list of strings (shouldn't happen with current loaders)
                final_texts = documents
                final_metadatas = [metadata] * len(documents) # Apply base metadata to all

        if not final_texts:
            logger.warning(f"No text content extracted from file {filename} (ID: {file_id}).")
            # Optionally, still add a placeholder document to Chroma to acknowledge the file upload
            # user_files_collection.add(documents=[f"Placeholder for empty/unreadable file: {filename}"], metadatas=[metadata], ids=[f"{file_id}-empty"])
            return 0 # No chunks added

        # Generate unique IDs for each chunk
        chunk_ids = [f"{file_id}-chunk-{i}" for i in range(len(final_texts))]
        
        logger.info(f"Adding {len(final_texts)} chunks to ChromaDB for file {filename} (ID: {file_id}).")
        user_files_collection.add(
            documents=final_texts,
            metadatas=final_metadatas,
            ids=chunk_ids
        )
        
        return len(final_texts) # Return number of chunks added
    except Exception as e:
        logger.error(f"Error processing file {filename} (ID: {file_id}): {e}", exc_info=True)
        # Re-raise as HTTPException to be caught by FastAPI error handling
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


# --- Endpoints from app_rag.py (adapted) ---

@app.post("/upload-file", response_model=FileUploadResponse)
async def upload_user_file(
    file: UploadFile = File(...), 
    current_user: UserInDB = Depends(get_current_active_user) # Added authentication
):
    """Upload and process a file for analysis, associated with the current user."""
    global UPLOAD_DIR # Ensure access to global UPLOAD_DIR

    if UPLOAD_DIR is None:
        logger.error("UPLOAD_DIR is not initialized.")
        raise HTTPException(status_code=500, detail="File upload directory not configured.")

    try:
        file_id = str(uuid.uuid4()) # Generate a unique file ID
        original_filename = file.filename if file.filename else "unknownfile"
        
        # Sanitize filename slightly (more robust sanitization might be needed)
        safe_filename = os.path.basename(original_filename).replace("..", "").replace("/", "")
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{safe_filename}")
        
        logger.info(f"User {current_user.email} (ID: {current_user.id}) uploading file: {original_filename} as {file_id}")

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the file and add to ChromaDB
        # Pass current_user.id if process_uploaded_file is adapted to use it in metadata
        num_chunks = await process_uploaded_file(
            file_path=file_path, 
            file_id=file_id, 
            content_type=str(file.content_type), # Ensure content_type is string
            filename=original_filename
            # user_id=str(current_user.id) # If associating with user
        )
        
        file_size = os.path.getsize(file_path)
        
        logger.info(f"File {original_filename} (ID: {file_id}) processed. Size: {file_size}, Chunks: {num_chunks}. Uploaded by user {current_user.email}.")

        return FileUploadResponse(
            file_id=file_id,
            filename=original_filename,
            content_type=str(file.content_type),
            size=file_size,
            upload_time=datetime.utcnow().isoformat() # Use UTC time
        )
    except HTTPException:
        raise # Re-raise HTTPExceptions from process_uploaded_file
    except Exception as e:
        logger.error(f"Error uploading file for user {current_user.email}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")
    finally:
        # Clean up the temporary file from server storage after processing
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
            except Exception as e_remove:
                logger.error(f"Error cleaning up temporary file {file_path}: {e_remove}")


@app.get("/system-api-keys-status")
async def get_system_api_keys_status(current_user: UserInDB = Depends(get_current_active_user)):
    """
    (Admin/Debug) Get status of system-configured API keys from environment variables.
    Requires authenticated user. Add role-based access for production.
    """
    # Add role check here if needed: if current_user.role != "admin": raise HTTPException(...)
    logger.info(f"User {current_user.email} requested system API key status.")
    return {
        "google_api_key_configured": bool(GOOGLE_API_KEY),
        "newsapi_key_configured": bool(NEWSAPI_KEY),
        "alpha_vantage_key_configured": bool(ALPHA_VANTAGE_KEY),
        "tavily_api_key_configured": bool(TAVILY_API_KEY),
        "gemini_api_key_configured": bool(GEMINI_API_KEY)
        # Note: This does not check if the keys are *valid*, only if they are present.
    }


# --- Langchain Agent Creation ---
def create_market_intelligence_agent(llm_instance: Gemini, user_id: Optional[str] = None):
    """
    Creates and initializes a Langchain agent with external API tools and a RAG retriever tool.
    user_id is available for future use (e.g., tool customization or memory).
    """
    global user_documents_tool # Access the globally initialized RAG tool

    if not llm_instance:
        logger.error("LLM instance not provided or not initialized. Cannot create agent.")
        return None # Or raise an exception

    tools_for_agent = list(all_tools) # Start with external API tools
    
    if user_documents_tool:
        tools_for_agent.append(user_documents_tool)
        logger.info("User documents RAG tool added to the agent.")
    else:
        logger.warning("User documents RAG tool not available. Agent will not have RAG capabilities for user files.")

    try:
        # Ensure tools have unique names if any issues arise, though Langchain usually handles it.
        agent_executor = initialize_agent(
            tools_for_agent,
            llm_instance,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True, # Good for debugging agent's thought process
            handle_parsing_errors="Check your output and make sure it conforms to the expected format.", # Provide guidance on error
            max_iterations=10, # Prevent overly long chains
            early_stopping_method="generate" # Stop if LLM generates a stop sequence
        )
        logger.info(f"Langchain agent created with {len(tools_for_agent)} tools. User ID: {user_id if user_id else 'N/A'}")
        return agent_executor
    except Exception as e:
        logger.error(f"Error creating Langchain agent: {e}", exc_info=True)
        return None


# --- Old RAG Chain Creation Function (Commented out / To be removed) ---
# def create_rag_chain(collection_name: str, query: str, current_user_id: str):
#     """Create a RAG chain for answering queries from a specific collection."""
#     # ... (implementation from previous steps) ...
#     # This function is now replaced by the agent-based approach for /chat.
#     pass


# --- Mock Data Retrieval Functions (from app_rag.py, adapted) ---
# These will be replaced by agent calls in /comprehensive-analysis
# In a future step, these could be replaced by agentic tools or more sophisticated RAG.

def get_competitors(query: str, domain: str) -> List[Dict[str, Any]]:
    """Mock: Get competitor information."""
    logger.info(f"Mock get_competitors called for query: '{query}', domain: '{domain}'")
    # Simulate external API call or database query
    base_competitors = [
        {"name": "Innovatech Solutions", "market_share": "30%", "strengths": ["Cutting-edge tech", "Strong R&D"], "weaknesses": ["High price point", "Slow market adoption"]},
        {"name": "MarketLead Inc.", "market_share": "25%", "strengths": ["Large customer base", "Brand recognition"], "weaknesses": ["Outdated tech stack", "Less agile"]},
        {"name": "AgileCore Ltd.", "market_share": "18%", "strengths": ["Flexible solutions", "Customer-centric"], "weaknesses": ["Smaller scale", "Limited marketing budget"]}
    ]
    # If NewsAPI key is available, try to get real data (example from app_rag)
    if NEWSAPI_KEY:
        try:
            url = f"https://newsapi.org/v2/everything?q={domain}%20{query}%20competitors&sortBy=relevancy&apiKey={NEWSAPI_KEY}"
            response = requests.get(url)
            response.raise_for_status() # Raise an exception for HTTP errors
            data = response.json()
            articles = data.get('articles', [])
            if articles:
                logger.info(f"NEWSAPI: Retrieved {len(articles)} articles for competitor analysis.")
                # Simplified: just add a note about news findings
                base_competitors.append({
                    "name": "News-Identified Trends/Competitors", 
                    "summary": f"Found {len(articles)} relevant news articles. First title: {articles[0]['title'] if articles else 'N/A'}"
                })
        except requests.exceptions.RequestException as e:
            logger.error(f"NEWSAPI error in get_competitors: {e}")
        except Exception as e: # Catch other potential errors like JSONDecodeError
            logger.error(f"Error processing NEWSAPI response in get_competitors: {e}")
    return base_competitors

def get_market_trends(query: str, domain: str) -> List[Dict[str, Any]]:
    """Mock: Get market trends."""
    logger.info(f"Mock get_market_trends called for query: '{query}', domain: '{domain}'")
    base_trends = [
        {"trend": "AI-driven Automation", "impact": "High", "timeframe": "1-2 years", "description": "Increased adoption of AI for automating business processes."},
        {"trend": "Personalized Customer Experiences", "impact": "High", "timeframe": "Ongoing", "description": "Businesses focusing on tailoring experiences to individual customer needs."},
        {"trend": "Sustainability and Green Tech", "impact": "Medium", "timeframe": "2-5 years", "description": "Growing demand for environmentally friendly products and services."}
    ]
    # If Alpha Vantage key is available (example from app_rag)
    if ALPHA_VANTAGE_KEY and domain.lower() in ["finance", "technology", "software", "saas"]: # Example domains
        try:
            symbol = "QQQ" if domain.lower() in ["technology", "software", "saas"] else "SPY" # Nasdaq for tech, S&P500 for general finance
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol={symbol}&apikey={ALPHA_VANTAGE_KEY}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if "Monthly Time Series" in data:
                logger.info(f"Alpha Vantage: Retrieved market data for {symbol}.")
                base_trends.append({
                    "trend": f"{symbol} Market Data", 
                    "details": f"Successfully fetched time series data for {symbol}. Analysis would be performed here."
                })
        except requests.exceptions.RequestException as e:
            logger.error(f"Alpha Vantage error in get_market_trends: {e}")
        except Exception as e:
            logger.error(f"Error processing Alpha Vantage response in get_market_trends: {e}")
    return base_trends

def get_customer_segments(query: str, domain: str) -> List[Dict[str, Any]]:
    """Mock: Get customer segments."""
    logger.info(f"Mock get_customer_segments called for query: '{query}', domain: '{domain}'")
    base_segments = [
        {"segment": "Early Adopters", "needs": ["Newest features", "Innovation"], "pain_points": ["High cost", "Potential bugs"], "size": "15%"},
        {"segment": "Pragmatists (Mainstream)", "needs": ["Proven solutions", "Reliability", "Good support"], "pain_points": ["Complex integrations", "Disruption to workflow"], "size": "60%"},
        {"segment": "Conservatives (Laggards)", "needs": ["Mature technology", "Low risk", "Simplicity"], "pain_points": ["Change aversion", "Lack of technical skills"], "size": "25%"}
    ]
    # If Tavily API key is available (example from app_rag)
    if TAVILY_API_KEY:
        try:
            search_query = f"customer segments in {domain} for {query}"
            payload = {"api_key": TAVILY_API_KEY, "query": search_query, "search_depth": "basic", "max_results": 3}
            response = requests.post("https://api.tavily.com/search", json=payload)
            response.raise_for_status()
            data = response.json()
            results = data.get('results', [])
            if results:
                logger.info(f"Tavily API: Retrieved {len(results)} search results for customer segments.")
                base_segments.append({
                    "segment": "Tavily Search Insights",
                    "details": f"Found {len(results)} web results. Example: {results[0]['title'] if results else 'N/A'}"
                })
        except requests.exceptions.RequestException as e:
            logger.error(f"Tavily API error in get_customer_segments: {e}")
        except Exception as e:
            logger.error(f"Error processing Tavily API response in get_customer_segments: {e}")
    return base_segments


# --- Endpoints from app_rag.py (adapted and merged) ---

@app.post("/comprehensive-analysis", response_model=Dict[str, Any]) # Using generic Dict for now
async def run_comprehensive_analysis_endpoint(
    request: AnalysisRequest, # This Pydantic model was already moved
    current_user: UserInDB = Depends(get_current_active_user) # Added authentication
):
    """
    Run a comprehensive market intelligence analysis using mock data functions.
    (This endpoint's RAG capabilities will be enhanced later)
    """
    logger.info(f"User {current_user.email} (ID: {current_user.id}) initiated comprehensive analysis for query: '{request.query}', domain: '{request.market_domain}'")
    global llm # Ensure access to the global LLM instance

    if not llm:
        logger.error("LLM not available for comprehensive analysis.")
        raise HTTPException(status_code=503, detail="LLM service is unavailable.")

    agent_executor = create_market_intelligence_agent(llm, user_id=str(current_user.id))
    if not agent_executor:
        logger.error("Failed to create market intelligence agent.")
        raise HTTPException(status_code=500, detail="Could not initialize analysis agent.")

    # Craft a detailed prompt for the agent
    analysis_prompt = (
        f"Provide a comprehensive market analysis for the query: '{request.query}' "
        f"within the market domain: '{request.market_domain}'. "
        "Your analysis should synthesize information from various sources. "
        "Consider including the following aspects if relevant and information is found: "
        "1. Recent news and developments related to the query. "
        "2. Financial overview or stock performance if the query relates to a publicly traded company (e.g., use MSFT, AAPL as symbols). "
        "3. Broader market trends or web search results that provide context. "
        "4. Relevant information from any documents the user might have uploaded regarding this query (if any). "
        "Present your findings in a structured report format. "
        "If specific information (e.g., stock data for a non-public entity) cannot be found, state that clearly."
    )
    
    logger.info(f"Comprehensive analysis prompt for agent: {analysis_prompt}")

    try:
        # Run the agent with the crafted prompt
        agent_response = await agent_executor.arun(analysis_prompt) # Use arun for async execution
        logger.info(f"Agent response for comprehensive analysis: {agent_response[:200]}...") # Log snippet
        
        # The response model is Dict[str, Any], so we wrap the agent's string output.
        # The agent's direct output is a string. We don't get structured sources here without more complex parsing or agent design.
        return {
            "analysis_id": str(uuid.uuid4()), # Generate a new ID for this response
            "query": request.query,
            "market_domain": request.market_domain,
            "analysis_output": agent_response, # Direct output from the agent
            "status": "completed",
            "executive_summary": "Summary to be generated from agent output if needed, or agent provides it.", # Placeholder
            # Note: Mock data functions (get_competitors etc.) are no longer directly called here.
            # The agent is expected to use its tools (News, AlphaVantage, Tavily, UserDocs) to gather this info.
        }
    except Exception as e:
        logger.error(f"Error during comprehensive analysis agent execution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during agent-based analysis: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat_with_market_intelligence_agent( # Renamed for clarity
    request: ChatMessage, 
    current_user: UserInDB = Depends(get_current_active_user)
):
    """Chat with the market intelligence agent."""
    logger.info(f"User {current_user.email} (ID: {str(current_user.id)}) initiated Langgraph chat. Message: '{request.message}'")
    
    session_id = request.session_id or str(uuid.uuid4()) # Use provided or generate new
    logger.info(f"Using session_id: {session_id} for chat with user {current_user.email}")

    if not llm: # llm is a global, checked here before passing to langgraph implicitly
        logger.error("LLM not available for Langgraph chat.")
        raise HTTPException(status_code=503, detail="LLM service is unavailable.")
    if not create_market_intelligence_agent: # create_market_intelligence_agent is a global function
        logger.error("Market intelligence agent factory not available for Langgraph chat.")
        raise HTTPException(status_code=503, detail="Agent factory service is unavailable.")

    try:
        # run_chat_graph will use the globally available `llm` and `create_market_intelligence_agent` from this file (app.py)
        # This is okay as they are initialized during startup before any request hits this endpoint.
        response_content = await run_chat_graph(
            user_id=str(current_user.id),
            session_id=session_id,
            input_query=request.message
        )
        
        logger.info(f"Langgraph chat response for session {session_id}: {response_content[:200]}...")

        return ChatResponse(
            response=response_content,
            timestamp=datetime.utcnow().isoformat(),
            sources=["Langgraph Agent response"] # Simplified source attribution for now
        )
    except Exception as e:
        logger.error(f"Error during Langgraph chat execution for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing your message with Langgraph: {str(e)}")
