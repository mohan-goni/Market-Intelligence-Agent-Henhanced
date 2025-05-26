import os
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

# In-memory database (replace with real DB in production)
users_db = {}
api_keys_db = {}
analysis_results_db = {}

# Models
class User(BaseModel):
    email: str
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user: User

class TokenData(BaseModel):
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

# Helper functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db, email: str):
    if email in db:
        user_dict = db[email]
        return UserInDB(**user_dict)
    return None

def authenticate_user(db, email: str, password: str):
    user = get_user(db, email)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
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
    user = get_user(users_db, email=token_data.email)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
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
def process_market_intelligence_query(query: str, market_domain: str, user_email: str):
    # Get user's API keys
    api_keys = api_keys_db.get(user_email, ApiKeys())
    
    # Initialize agent
    agent = get_agent(api_keys)
    
    # Process query
    try:
        result = agent.process_query(query, market_domain)
        
        # Store result
        if user_email not in analysis_results_db:
            analysis_results_db[user_email] = []
        
        analysis_results_db[user_email].append({
            "id": len(analysis_results_db[user_email]) + 1,
            "query": query,
            "market_domain": market_domain,
            "result": result,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "completed"
        })
    except Exception as e:
        # Store error
        if user_email not in analysis_results_db:
            analysis_results_db[user_email] = []
        
        analysis_results_db[user_email].append({
            "id": len(analysis_results_db[user_email]) + 1,
            "query": query,
            "market_domain": market_domain,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
            "status": "failed"
        })

# Routes
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": User(email=user.email, full_name=user.full_name)
    }

@app.post("/users", response_model=User)
async def create_user(email: str = Form(...), password: str = Form(...), full_name: Optional[str] = Form(None)):
    if email in users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    hashed_password = get_password_hash(password)
    user = UserInDB(email=email, full_name=full_name, hashed_password=hashed_password)
    users_db[email] = user.dict()
    return User(email=email, full_name=full_name)

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

@app.post("/reset-password")
async def reset_password(reset_data: PasswordReset):
    # In a real app, this would send an email with reset instructions
    # For this demo, we'll just print a message
    print(f"Password reset requested for {reset_data.email}")
    return {"message": "If your email is registered, you will receive password reset instructions"}

@app.get("/api-keys", response_model=ApiKeys)
async def get_api_keys(current_user: User = Depends(get_current_active_user)):
    return api_keys_db.get(current_user.email, ApiKeys())

@app.post("/api-keys")
async def set_api_keys(api_keys: ApiKeys, current_user: User = Depends(get_current_active_user)):
    api_keys_db[current_user.email] = api_keys
    return {"message": "API keys updated successfully"}

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
    # Get user's API keys
    api_keys = api_keys_db.get(current_user.email, ApiKeys())
    
    # Initialize agent
    agent = get_agent(api_keys)
    
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

@app.get("/analysis-results")
async def get_analysis_results(current_user: User = Depends(get_current_active_user)):
    return analysis_results_db.get(current_user.email, [])

@app.get("/analysis-results/{result_id}")
async def get_analysis_result(
    result_id: int,
    current_user: User = Depends(get_current_active_user)
):
    user_results = analysis_results_db.get(current_user.email, [])
    for result in user_results:
        if result["id"] == result_id:
            return result
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Analysis result not found"
    )

# Serve static files for the frontend
FRONTEND_BUILD_DIR = os.path.abspath("frontend/market-intel-ui/dist")
app.mount("/", StaticFiles(directory=FRONTEND_BUILD_DIR, html=True), name="static")

# Add a test user for development
@app.on_event("startup")
async def startup_event():
    # Add a test user
    if "test@example.com" not in users_db:
        hashed_password = get_password_hash("password")
        users_db["test@example.com"] = {
            "email": "test@example.com",
            "full_name": "Test User",
            "hashed_password": hashed_password,
            "disabled": False
        }
    
    # Add the organization email
    if "marketintelligenceagent@gmail.com" not in users_db:
        hashed_password = get_password_hash("admin123")
        users_db["marketintelligenceagent@gmail.com"] = {
            "email": "marketintelligenceagent@gmail.com",
            "full_name": "Market Intelligence Admin",
            "hashed_password": hashed_password,
            "disabled": False
        }

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
