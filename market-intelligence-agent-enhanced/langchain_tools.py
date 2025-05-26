import os
import requests
import json
from langchain.tools import Tool
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# Load environment variables - primarily for standalone testing or if this module is used independently.
# In app.py, dotenv is already loaded.
load_dotenv()

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# --- NewsAPI Tool ---

def search_news(query: str, domain: Optional[str] = None) -> str:
    """
    Searches for recent news articles using NewsAPI.
    Useful for finding current events, company news, and market updates.
    Input should be a search query string, optionally with a market domain.
    """
    if not NEWSAPI_KEY:
        return "Error: NewsAPI key not configured. Please set the NEWSAPI_KEY environment variable."

    base_url = "https://newsapi.org/v2/everything"
    search_query = query
    if domain:
        search_query += f" AND (site:{domain} OR domain:{domain})" # Attempt to search within a domain if provided

    params = {
        "q": search_query,
        "apiKey": NEWSAPI_KEY,
        "pageSize": 5,  # Limit to 5 articles
        "sortBy": "relevancy" # or 'publishedAt' for latest
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        data = response.json()
        
        articles = data.get("articles", [])
        if not articles:
            return f"No news articles found for query: '{query}'" + (f" in domain '{domain}'" if domain else "")

        formatted_articles = []
        for i, article in enumerate(articles):
            title = article.get('title', 'N/A')
            source = article.get('source', {}).get('name', 'N/A')
            description = article.get('description', 'N/A')
            url = article.get('url', '#')
            # published_at = article.get('publishedAt', 'N/A')

            formatted_articles.append(
                f"Article {i+1}:\n"
                f"  Title: {title}\n"
                f"  Source: {source}\n"
                # f"  Published: {published_at}\n"
                f"  Description: {description}\n"
                f"  URL: {url}\n"
            )
        
        return "\n".join(formatted_articles) + f"\n\nFound {len(articles)} articles."

    except requests.exceptions.RequestException as e:
        return f"Error calling NewsAPI: {e}"
    except json.JSONDecodeError:
        return "Error: Could not decode JSON response from NewsAPI."
    except Exception as e:
        return f"An unexpected error occurred while fetching news: {e}"

news_tool = Tool(
    name="NewsAPISearch",
    func=search_news,
    description="Searches for recent news articles using NewsAPI. Useful for finding current events, company news, and market updates. Input should be a search query string, optionally with a market domain (e.g., 'apple.com')."
)

# --- Alpha Vantage Tool ---

def get_stock_data(symbol: str, function: str = "TIME_SERIES_DAILY_ADJUSTED") -> str:
    """
    Fetches financial data for a given stock symbol from Alpha Vantage.
    Input should be a stock symbol (e.g., 'MSFT').
    Optionally, specify the Alpha Vantage function to call (e.g., 'TIME_SERIES_DAILY_ADJUSTED', 'OVERVIEW', 'INCOME_STATEMENT').
    Default function is 'TIME_SERIES_DAILY_ADJUSTED'.
    """
    if not ALPHA_VANTAGE_KEY:
        return "Error: Alpha Vantage API key not configured. Please set the ALPHA_VANTAGE_KEY environment variable."

    base_url = "https://www.alphavantage.co/query"
    params = {
        "symbol": symbol,
        "function": function.upper(), # Ensure function name is uppercase
        "apikey": ALPHA_VANTAGE_KEY
    }

    # Add specific parameters for certain functions if needed
    if function.upper() == "TIME_SERIES_DAILY_ADJUSTED":
        params["outputsize"] = "compact" # compact for last 100 days, full for full history

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        if "Error Message" in data:
            return f"Alpha Vantage API Error for symbol '{symbol}', function '{function}': {data['Error Message']}"
        if "Information" in data and "API call frequency" in data["Information"]: # Handle rate limiting messages
             return f"Alpha Vantage API Information: {data['Information']}. This might indicate a rate limit."


        if function.upper() == "TIME_SERIES_DAILY_ADJUSTED":
            time_series = data.get("Time Series (Daily)", {})
            if not time_series:
                return f"No daily time series data found for symbol '{symbol}'."
            
            # Get latest few entries
            latest_dates = sorted(time_series.keys(), reverse=True)[:3]
            summary = f"Recent Daily Prices for {symbol}:\n"
            for date in latest_dates:
                prices = time_series[date]
                summary += (
                    f"  Date: {date}\n"
                    f"    Open: {prices.get('1. open')}\n"
                    f"    High: {prices.get('2. high')}\n"
                    f"    Low: {prices.get('3. low')}\n"
                    f"    Close: {prices.get('4. close')}\n"
                    f"    Adjusted Close: {prices.get('5. adjusted close')}\n"
                    f"    Volume: {prices.get('6. volume')}\n"
                )
            return summary

        elif function.upper() == "OVERVIEW":
            if not data or all(value is None for value in data.values()): # Check if data is empty or all None
                 return f"No overview data found for symbol '{symbol}'. The symbol might be invalid or not covered."
            overview_summary = f"Company Overview for {data.get('Symbol', symbol)} ({data.get('Name', 'N/A')}):\n"
            overview_summary += f"  Description: {data.get('Description', 'N/A')[:500]}...\n" # Truncate long descriptions
            overview_summary += f"  Industry: {data.get('Industry', 'N/A')}\n"
            overview_summary += f"  Sector: {data.get('Sector', 'N/A')}\n"
            overview_summary += f"  Market Cap: {data.get('MarketCapitalization', 'N/A')}\n"
            overview_summary += f"  P/E Ratio: {data.get('PERatio', 'N/A')}\n"
            overview_summary += f"  EPS: {data.get('EPS', 'N/A')}\n"
            overview_summary += f"  52 Week High: {data.get('52WeekHigh', 'N/A')}\n"
            overview_summary += f"  52 Week Low: {data.get('52WeekLow', 'N/A')}\n"
            return overview_summary
        
        # For other functions, return a string representation of the JSON, truncated if too long
        # This provides flexibility but might need custom parsing for specific use cases later.
        raw_json_str = json.dumps(data, indent=2)
        if len(raw_json_str) > 2000: # Arbitrary limit to keep it manageable for LLM
            raw_json_str = raw_json_str[:2000] + "\n... (data truncated)"
        return f"Data for {symbol}, function {function}:\n{raw_json_str}"

    except requests.exceptions.RequestException as e:
        return f"Error calling Alpha Vantage API: {e}"
    except json.JSONDecodeError:
        return "Error: Could not decode JSON response from Alpha Vantage."
    except Exception as e:
        return f"An unexpected error occurred while fetching stock data: {e}"

alpha_vantage_tool = Tool(
    name="AlphaVantageStockData",
    func=get_stock_data,
    description="Fetches financial data for a given stock symbol from Alpha Vantage. Useful for getting stock prices (e.g., TIME_SERIES_DAILY_ADJUSTED), company overviews (OVERVIEW), or other financial metrics (e.g., INCOME_STATEMENT, BALANCE_SHEET, EARNINGS). Input should be a stock symbol (e.g., 'MSFT') and optionally the type of data desired (function name like 'OVERVIEW'). Default is 'TIME_SERIES_DAILY_ADJUSTED'."
)

# --- Tavily API Tool ---

def tavily_search(query: str, search_depth: str = "basic", max_results: int = 5) -> str:
    """
    Performs a web search using Tavily API.
    Useful for finding information on a wide range of topics, research, and general knowledge.
    Input should be a search query string. Optional: search_depth ('basic' or 'advanced'), max_results (number).
    """
    if not TAVILY_API_KEY:
        return "Error: Tavily API key not configured. Please set the TAVILY_API_KEY environment variable."

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": search_depth,
        "include_answer": False, # We are primarily interested in search results content
        "max_results": max_results
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        
        results = data.get("results", [])
        if not results:
            return f"No search results found for query: '{query}' using Tavily."

        formatted_results = []
        for i, result in enumerate(results):
            title = result.get('title', 'N/A')
            url = result.get('url', '#')
            content_snippet = result.get('content', 'N/A')[:500] # Get a snippet of content

            formatted_results.append(
                f"Result {i+1}:\n"
                f"  Title: {title}\n"
                f"  URL: {url}\n"
                f"  Content Snippet: {content_snippet}...\n"
            )
        
        return "\n".join(formatted_results) + f"\n\nFound {len(results)} search results."

    except requests.exceptions.RequestException as e:
        return f"Error calling Tavily API: {e}"
    except json.JSONDecodeError:
        return "Error: Could not decode JSON response from Tavily API."
    except Exception as e:
        return f"An unexpected error occurred during Tavily search: {e}"

tavily_search_tool = Tool(
    name="TavilySearch",
    func=tavily_search,
    description="Performs a comprehensive web search using Tavily API. Useful for finding information on a wide range of topics, research, and general knowledge. Input should be a search query string. You can optionally specify 'search_depth' (e.g., 'basic', 'advanced') and 'max_results' (e.g., 3, 5) by providing a JSON string as input like '{\"query\": \"your query\", \"search_depth\": \"advanced\", \"max_results\": 5}' or just the query string for defaults."
)

# --- List of all tools ---
all_tools = [news_tool, alpha_vantage_tool, tavily_search_tool]

# Example Usage (for testing this file directly)
# if __name__ == "__main__":
#     # Test NewsAPI
#     print("--- NewsAPI Test ---")
#     news_result = search_news("Apple new product announcements", domain="apple.com")
#     # news_result = search_news("Tesla stock news")
#     print(news_result)
#     print("\n")

#     # Test Alpha Vantage
#     print("--- Alpha Vantage Test ---")
#     # stock_data_daily = get_stock_data("IBM", function="TIME_SERIES_DAILY_ADJUSTED")
#     # print(stock_data_daily)
#     # print("\n")
#     stock_data_overview = get_stock_data("MSFT", function="OVERVIEW")
#     print(stock_data_overview)
#     print("\n")
#     # stock_data_earnings = get_stock_data("GOOGL", function="EARNINGS")
#     # print(stock_data_earnings)
#     # print("\n")


#     # Test Tavily Search
#     print("--- Tavily Search Test ---")
#     # tavily_result_basic = tavily_search("latest advancements in quantum computing", max_results=3)
#     # print(tavily_result_basic)
#     # print("\n")
    
#     # Example of JSON input for Tavily (if needed, though func now has typed args)
#     # tavily_result_advanced = tavily_search('{"query": "market sentiment for EV stocks", "search_depth": "advanced", "max_results": 3}')
#     # print(tavily_result_advanced)
#     # print("\n")
#     tavily_direct_call = tavily_search(query="impact of AI on software development jobs", search_depth="advanced", max_results=2)
#     print(tavily_direct_call)

#     # Test using the Tool interface
#     # print("--- Tool Interface Test ---")
#     # print(news_tool.run("NVIDIA quarterly earnings"))
#     # print(alpha_vantage_tool.run("AAPL")) # Default TIME_SERIES_DAILY_ADJUSTED
#     # print(alpha_vantage_tool.run('{"symbol": "TSLA", "function": "OVERVIEW"}')) # JSON input for more control
#     # print(tavily_search_tool.run("future of renewable energy"))
#     # print(tavily_search_tool.run('{"query": "best python libraries for data science", "max_results": 2}'))

```
