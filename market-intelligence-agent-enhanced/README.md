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
market_intelligence_project/
├── frontend/                  # React TypeScript frontend
│   └── market-intel-ui/
│       ├── public/            # Static assets
│       ├── src/
│       │   ├── components/    # Reusable UI components
│       │   ├── contexts/      # React contexts (auth, etc.)
│       │   ├── hooks/         # Custom React hooks
│       │   ├── pages/         # Page components
│       │   ├── services/      # API service functions
│       │   ├── tests/         # Frontend tests
│       │   └── utils/         # Utility functions
│       ├── package.json       # Frontend dependencies
│       └── tsconfig.json      # TypeScript configuration
├── MIA.py                     # Market Intelligence Agent core
├── app.py                     # FastAPI backend application
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Getting Started

### Prerequisites

- Node.js (v16+)
- Python (v3.8+)
- npm or yarn

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/market-intelligence-agent.git
cd market-intelligence-agent
```

2. **Set up the backend**

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

3. **Set up the frontend**

```bash
cd frontend/market-intel-ui
npm install  # or yarn install
```

### Configuration

1. **API Keys**

Create a `.env` file in the root directory with the following variables:

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
# From the root directory
uvicorn app:app --reload
```

2. **Start the frontend**

```bash
# From the frontend/market-intel-ui directory
npm start  # or yarn start
```

3. **Access the application**

Open your browser and navigate to `http://localhost:3000`

## Default Credentials

For testing purposes, the following credentials are pre-configured:

- **Email**: test@example.com
- **Password**: password

- **Admin Email**: marketintelligenceagent@gmail.com
- **Admin Password**: admin123

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

### Backend Tests

```bash
# From the root directory
pytest
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
