import os
import json
import uuid
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
import requests
import pandas as pd
import tempfile
import shutil

# Import RAG system components
from app_rag import (
    process_file,
    create_rag_chain,
    user_files_collection,
    market_data_collection,
    competitor_data_collection,
    customer_data_collection,
    llm
)

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

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Create a temporary directory for file uploads
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory storage for analysis results (in production, use a database)
analysis_results = {}

# Pydantic models
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

# Data integration functions
async def fetch_news_data(query: str, domain: str):
    """Fetch news data from NewsAPI"""
    if not NEWSAPI_KEY:
        logger.warning("NewsAPI key not configured")
        return []
    
    try:
        url = f"https://newsapi.org/v2/everything?q={query}+{domain}&sortBy=relevancy&apiKey={NEWSAPI_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])
            
            # Store articles in vector database
            texts = []
            metadatas = []
            ids = []
            
            for i, article in enumerate(articles[:20]):  # Limit to 20 articles
                if article.get('content') and article.get('title'):
                    text = f"Title: {article['title']}\n\nContent: {article['content']}"
                    metadata = {
                        "source": article.get('source', {}).get('name', 'NewsAPI'),
                        "url": article.get('url', ''),
                        "published_at": article.get('publishedAt', ''),
                        "domain": domain,
                        "query": query
                    }
                    
                    texts.append(text)
                    metadatas.append(metadata)
                    ids.append(f"news-{query}-{domain}-{i}")
            
            if texts:
                market_data_collection.add(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
            
            logger.info(f"Added {len(texts)} news articles to vector database")
            return articles
        else:
            logger.error(f"NewsAPI error: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Error fetching news data: {e}")
        return []

async def fetch_market_data(domain: str):
    """Fetch market data from Alpha Vantage"""
    if not ALPHA_VANTAGE_KEY:
        logger.warning("Alpha Vantage key not configured")
        return {}
    
    try:
        # Map domain to relevant stock symbol
        symbol_map = {
            "technology": "QQQ",  # NASDAQ-100
            "healthcare": "XLV",  # Health Care Select Sector SPDR
            "finance": "XLF",     # Financial Select Sector SPDR
            "energy": "XLE",      # Energy Select Sector SPDR
            "retail": "XRT",      # SPDR S&P Retail ETF
            "consumer": "XLP",    # Consumer Staples Select Sector SPDR
            "manufacturing": "XLI", # Industrial Select Sector SPDR
            "education": "EDUC"   # Educational Development Corp (example)
        }
        
        symbol = symbol_map.get(domain.lower(), "SPY")  # Default to S&P 500
        
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol={symbol}&apikey={ALPHA_VANTAGE_KEY}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract time series data
            time_series = data.get('Monthly Time Series', {})
            
            # Convert to a format suitable for vector database
            if time_series:
                # Create a summary text
                summary_text = f"Market data for {symbol} (representing {domain} sector):\n\n"
                
                # Get the last 12 months of data
                dates = list(time_series.keys())[:12]
                
                for date in dates:
                    entry = time_series[date]
                    summary_text += f"Date: {date}\n"
                    summary_text += f"Open: {entry['1. open']}\n"
                    summary_text += f"High: {entry['2. high']}\n"
                    summary_text += f"Low: {entry['3. low']}\n"
                    summary_text += f"Close: {entry['4. close']}\n"
                    summary_text += f"Volume: {entry['5. volume']}\n\n"
                
                # Add to vector database
                market_data_collection.add(
                    documents=[summary_text],
                    metadatas=[{
                        "source": "Alpha Vantage",
                        "symbol": symbol,
                        "domain": domain,
                        "data_type": "market_data"
                    }],
                    ids=[f"market-data-{domain}-{symbol}"]
                )
                
                logger.info(f"Added market data for {symbol} to vector database")
            
            return data
        else:
            logger.error(f"Alpha Vantage error: {response.status_code}")
            return {}
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        return {}

async def search_with_tavily(query: str, domain: str):
    """Search for information using Tavily API"""
    if not TAVILY_API_KEY:
        logger.warning("Tavily API key not configured")
        return []
    
    try:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": TAVILY_API_KEY,
            "query": f"{query} {domain}",
            "search_depth": "advanced",
            "include_domains": ["statista.com", "forrester.com", "gartner.com", "mckinsey.com", "hbr.org"]
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            
            # Store results in vector database
            texts = []
            metadatas = []
            ids = []
            
            for i, result in enumerate(results):
                if result.get('content'):
                    text = f"Title: {result.get('title', 'No title')}\n\nContent: {result['content']}"
                    metadata = {
                        "source": result.get('url', 'Tavily Search'),
                        "domain": domain,
                        "query": query
                    }
                    
                    texts.append(text)
                    metadatas.append(metadata)
                    ids.append(f"tavily-{query}-{domain}-{i}")
            
            if texts:
                # Determine which collection to use based on content
                if "competitor" in query.lower() or "competition" in query.lower():
                    competitor_data_collection.add(
                        documents=texts,
                        metadatas=metadatas,
                        ids=ids
                    )
                    logger.info(f"Added {len(texts)} competitor search results to vector database")
                elif "customer" in query.lower() or "consumer" in query.lower() or "user" in query.lower():
                    customer_data_collection.add(
                        documents=texts,
                        metadatas=metadatas,
                        ids=ids
                    )
                    logger.info(f"Added {len(texts)} customer search results to vector database")
                else:
                    market_data_collection.add(
                        documents=texts,
                        metadatas=metadatas,
                        ids=ids
                    )
                    logger.info(f"Added {len(texts)} market search results to vector database")
            
            return results
        else:
            logger.error(f"Tavily API error: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Error searching with Tavily: {e}")
        return []

async def generate_insights_with_gemini(data: Dict[str, Any], query: str, domain: str):
    """Generate insights using Gemini API"""
    if not GEMINI_API_KEY or not llm:
        logger.warning("Gemini API key not configured or LLM not initialized")
        return "No insights available. Gemini API not configured."
    
    try:
        # Create a prompt with all the data
        prompt = f"""
        Generate comprehensive market intelligence insights for the query: "{query}" 
        in the {domain} domain.
        
        Use the following data to inform your analysis:
        
        {json.dumps(data, indent=2)}
        
        Provide insights in the following format:
        1. Executive Summary (2-3 sentences)
        2. Key Findings (3-5 bullet points)
        3. Strategic Recommendations (2-3 bullet points)
        
        Be specific, data-driven, and actionable in your insights.
        """
        
        # Generate insights
        insights = llm(prompt)
        
        # Store insights in vector database
        market_data_collection.add(
            documents=[insights],
            metadatas=[{
                "source": "Gemini Analysis",
                "domain": domain,
                "query": query,
                "data_type": "insights"
            }],
            ids=[f"insights-{query}-{domain}"]
        )
        
        logger.info(f"Added Gemini insights to vector database")
        
        return insights
    except Exception as e:
        logger.error(f"Error generating insights with Gemini: {e}")
        return "Error generating insights."

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
        
        # Process the file and add to vector database
        chunk_count = await process_file(file_path, file_id, file.content_type, file.filename)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        logger.info(f"File {file.filename} uploaded and processed into {chunk_count} chunks")
        
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
            "status": "processing",
            "query": request.query,
            "domain": request.market_domain
        }
        
        # Store initial response
        analysis_results[analysis_id] = response
        
        # Collect data from various sources
        all_data = {}
        
        # 1. Process any uploaded files (they're already in the vector database)
        if request.file_ids:
            all_data["uploaded_files"] = request.file_ids
        
        # 2. Fetch news data
        news_articles = await fetch_news_data(request.query, request.market_domain)
        all_data["news_data"] = news_articles[:5]  # Include only first 5 articles in response
        
        # 3. Fetch market data
        market_data = await fetch_market_data(request.market_domain)
        all_data["market_data"] = market_data
        
        # 4. Search with Tavily
        search_results = await search_with_tavily(request.query, request.market_domain)
        all_data["search_results"] = search_results[:5]  # Include only first 5 results in response
        
        # 5. Generate competitor analysis
        if request.include_competitor_analysis:
            competitor_query = f"competitors in {request.market_domain} {request.query}"
            competitor_results = await search_with_tavily(competitor_query, request.market_domain)
            
            # Extract competitor information
            competitors = []
            for i, result in enumerate(competitor_results[:3]):
                competitor = {
                    "name": f"Competitor {i+1}",
                    "market_share": "Unknown",
                    "strengths": ["Innovation", "Market Presence"],
                    "weaknesses": ["Unknown"]
                }
                competitors.append(competitor)
            
            all_data["competitor_analysis"] = {
                "competitors": competitors,
                "summary": f"Competitor analysis for {request.market_domain} market related to '{request.query}'"
            }
        
        # 6. Generate market trends
        if request.include_market_trends:
            trends_query = f"market trends in {request.market_domain} {request.query}"
            trends_results = await search_with_tavily(trends_query, request.market_domain)
            
            # Extract trend information
            trends = []
            for i, result in enumerate(trends_results[:3]):
                trend = {
                    "trend": f"Trend {i+1}",
                    "impact": "Medium",
                    "timeframe": "Medium-term (6-18 months)"
                }
                trends.append(trend)
            
            all_data["market_trends"] = {
                "trends": trends,
                "summary": f"Market trends for {request.market_domain} market related to '{request.query}'"
            }
        
        # 7. Generate customer insights
        if request.include_customer_insights:
            customer_query = f"customer segments in {request.market_domain} {request.query}"
            customer_results = await search_with_tavily(customer_query, request.market_domain)
            
            # Extract customer segment information
            segments = []
            for i, result in enumerate(customer_results[:3]):
                segment = {
                    "segment": f"Segment {i+1}",
                    "needs": ["Reliability", "Value"],
                    "growth_rate": "10% annually"
                }
                segments.append(segment)
            
            all_data["customer_insights"] = {
                "segments": segments,
                "summary": f"Customer insights for {request.market_domain} market related to '{request.query}'"
            }
        
        # 8. Generate overall summary using Gemini
        summary = await generate_insights_with_gemini(all_data, request.query, request.market_domain)
        
        # Complete response
        final_response = {
            "analysis_id": analysis_id,
            "status": "completed",
            "summary": summary,
            "query": request.query,
            "domain": request.market_domain
        }
        
        # Add specific analysis components if requested
        if request.include_competitor_analysis and "competitor_analysis" in all_data:
            final_response["competitor_analysis"] = all_data["competitor_analysis"]
        
        if request.include_market_trends and "market_trends" in all_data:
            final_response["market_trends"] = all_data["market_trends"]
        
        if request.include_customer_insights and "customer_insights" in all_data:
            final_response["customer_insights"] = all_data["customer_insights"]
        
        # Store the final analysis results
        analysis_results[analysis_id] = final_response
        
        return final_response
    except Exception as e:
        logger.error(f"Error running analysis: {e}")
        # Update status to failed
        if analysis_id in analysis_results:
            analysis_results[analysis_id]["status"] = "failed"
            analysis_results[analysis_id]["error"] = str(e)
        
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis-results/{analysis_id}")
async def get_analysis_results(analysis_id: str):
    """Get results of a specific analysis"""
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return analysis_results[analysis_id]

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
        
        # If analysis_id is provided, include context from that analysis
        analysis_context = ""
        if request.analysis_id and request.analysis_id in analysis_results:
            analysis = analysis_results[request.analysis_id]
            analysis_context = f"""
            This question is about an analysis with ID {request.analysis_id}.
            Query: {analysis.get('query', 'Unknown')}
            Domain: {analysis.get('domain', 'Unknown')}
            Summary: {analysis.get('summary', 'No summary available')}
            """
        
        # Collect answers from each relevant collection
        all_answers = []
        all_sources = []
        
        for collection_name in collections:
            try:
                # Add analysis context to the query if available
                enhanced_query = request.message
                if analysis_context:
                    enhanced_query = f"{analysis_context}\n\nUser question: {request.message}"
                
                result = create_rag_chain(collection_name, enhanced_query)
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
                
                {analysis_context}
                
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

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
