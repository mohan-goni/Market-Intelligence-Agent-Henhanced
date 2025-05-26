# Market Intelligence Agent - README

## Overview

Market Intelligence Agent is a full-stack application that provides advanced market intelligence analysis through a modern, responsive user interface. The application combines powerful backend processing with an intuitive frontend to deliver actionable market insights.

## Features

- **User Authentication**: Secure login, registration, and password recovery
- **API Key Management**: Securely store and manage external API keys
- **Data Integration**: Configure and manage multiple data sources
- **Market Trends Analysis**: Analyze current market trends and patterns
- **Competitor Analysis**: Compare competitors and their market positioning
- **Customer Insights**: Analyze customer feedback and sentiment
- **Report Generation**: Create and download comprehensive reports

## Tech Stack

### Frontend
- React with TypeScript
- Tailwind CSS for styling
- React Query for data fetching
- React Router for navigation
- React Hook Form for form handling

### Backend
- FastAPI (Python)
- JWT authentication
- Background task processing
- LangChain for NLP processing
- Google Gemini API integration

## Project Structure

```
market-intelligence-agent-enhanced/
├── .env.example               # Example environment variables
├── .gitignore                 # Git ignore file
├── app.py                     # FastAPI backend application
├── enhancement_plan.md        # Document outlining planned enhancements
├── frontend/                  # React TypeScript frontend
│   └── market-intel-ui/
│       ├── public/            # Static assets
│       ├── src/               # Frontend source code
│       │   ├── components/    # Reusable UI components
│       │   ├── contexts/      # React contexts (auth, etc.)
│       │   ├── hooks/         # Custom React hooks
│       │   ├── pages/         # Page components
│       │   ├── services/      # API service functions
│       │   └── utils/         # Utility functions
│       ├── package.json       # Frontend dependencies
│       └── vite.config.ts     # Vite configuration
├── langchain_tools.py         # Custom Langchain tools for external APIs
├── langgraph_chat.py          # Langgraph chat flow implementation
├── requirements.txt           # Python backend dependencies
├── setup_and_run.sh           # Script to setup and run the application
├── supabase_client.py         # Supabase client utilities
├── validation_report.md       # Report on testing and validation
└── README.md                  # This file
```

## Getting Started

### Prerequisites

- Node.js (v16+)
- Python (v3.8+)
- npm or yarn

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/market-intelligence-agent.git # Replace with actual repo URL
cd market-intelligence-agent-enhanced
```

2. **Set up the backend**

   (Ensure you are in the `market-intelligence-agent-enhanced` directory)
```bash
# Create and activate a virtual environment
python3 -m venv venv  # Use python3 explicitly if needed
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies from within the market-intelligence-agent-enhanced directory
pip install -r requirements.txt
```

3. **Set up the frontend**

```bash
cd frontend/market-intel-ui
npm install  # or yarn install
```

### Configuration

1. **API Keys**

Create a `.env` file in the `market-intelligence-agent-enhanced` directory (you can copy from `.env.example`):

```
SECRET_KEY=your_jwt_secret_key
GOOGLE_API_KEY=your_google_api_key
NEWSAPI_KEY=your_newsapi_key
ALPHA_VANTAGE_KEY=your_alpha_vantage_key
TAVILY_API_KEY=your_tavily_api_key
```

2. **Frontend Configuration**

Create a `.env` file in the `frontend/market-intel-ui` directory:

```
REACT_APP_API_URL=http://localhost:8000
```

### Running the Application

1. **Start the backend**

```bash
# From the market-intelligence-agent-enhanced directory
uvicorn app:app --host 0.0.0.0 --port 8000 --reload 
# Or run the setup_and_run.sh script which handles this.
```

2. **Start the frontend**

```bash
# From the frontend/market-intel-ui directory
npm start  # or yarn start
```

3. **Access the application**

Open your browser and navigate to `http://localhost:3000` (or the port specified by Vite, usually 5173 if 3000 is taken)

## API Documentation

Once the backend is running, you can access the API documentation at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Testing

### Frontend Tests

```bash
cd frontend/market-intel-ui
npm test  # or yarn test
```

## Deployment

### Frontend Deployment

The frontend can be built for production using:

```bash
cd frontend/market-intel-ui
npm run build  # or yarn build
```

This creates a `build` directory with optimized production files that can be served by any static file server.

### Backend Deployment

The backend can be deployed using various methods, including:

- Docker containers
- Cloud platforms (AWS, Google Cloud, Azure)
- Traditional web servers with WSGI/ASGI servers like Gunicorn or Uvicorn

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) for NLP processing
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- [React](https://reactjs.org/) for the frontend framework
- [Tailwind CSS](https://tailwindcss.com/) for styling
