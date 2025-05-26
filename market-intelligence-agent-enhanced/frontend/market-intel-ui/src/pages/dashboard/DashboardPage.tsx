import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import Sidebar from '../../components/layout/Sidebar';
import Header from '../../components/layout/Header';
import ChatInterface from '../../components/chat/ChatInterface';
import FileUpload from '../../components/file-upload/FileUpload';

const DashboardPage: React.FC = () => {
  const { user } = useAuth();
  const [isLoading, setIsLoading] = useState(false);
  const [query, setQuery] = useState('');
  const [domain, setDomain] = useState('');
  const [isRunningAgent, setIsRunningAgent] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<any>(null);
  const [uploadedFileIds, setUploadedFileIds] = useState<string[]>([]);
  const [analysisId, setAnalysisId] = useState<string | undefined>(undefined);
  const [showChat, setShowChat] = useState(false);
  const [apiStatus, setApiStatus] = useState({
    google_api: false,
    news_api: false,
    alpha_vantage: false,
    tavily_api: false,
    gemini_api: false
  });

  const domains = [
    { value: 'technology', label: 'Technology' },
    { value: 'healthcare', label: 'Healthcare' },
    { value: 'finance', label: 'Finance' },
    { value: 'retail', label: 'Retail' },
    { value: 'manufacturing', label: 'Manufacturing' },
    { value: 'energy', label: 'Energy' },
    { value: 'education', label: 'Education' },
    { value: 'entertainment', label: 'Entertainment' },
    { value: 'food', label: 'Food & Beverage' },
    { value: 'transportation', label: 'Transportation' }
  ];

  // Fetch API keys status on component mount
  useEffect(() => {
    const fetchApiStatus = async () => {
      try {
        const response = await fetch('/api-keys', {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        });
        
        if (response.ok) {
          const data = await response.json();
          setApiStatus({
            google_api: !!data.google_api_key,
            news_api: !!data.newsapi_key,
            alpha_vantage: !!data.alpha_vantage_key,
            tavily_api: !!data.tavily_api_key,
            gemini_api: !!data.gemini_api_key
          });
        }
      } catch (error) {
        console.error('Error fetching API status:', error);
      }
    };
    
    fetchApiStatus();
  }, []);

  const handleFilesUploaded = (fileIds: string[]) => {
    setUploadedFileIds(fileIds);
  };

  const handleRunAgent = async () => {
    if (!query.trim()) {
      alert('Please enter a query');
      return;
    }

    setIsRunningAgent(true);
    
    try {
      // Make actual API call to backend
      const response = await fetch('/comprehensive-analysis', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          query: query,
          market_domain: domain || 'general',
          file_ids: uploadedFileIds.length > 0 ? uploadedFileIds : undefined,
          include_competitor_analysis: true,
          include_market_trends: true,
          include_customer_insights: true
        })
      });
      
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Store the analysis ID for polling
      setAnalysisId(data.analysis_id);
      
      // Start polling for results
      pollAnalysisResults(data.analysis_id);
    } catch (error) {
      console.error('Error running analysis:', error);
      setIsRunningAgent(false);
      alert('Failed to run analysis. Please try again later.');
    }
  };
  
  const pollAnalysisResults = async (id: string) => {
    try {
      const response = await fetch(`/analysis-results/${id}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.status === 'completed') {
        setAnalysisResults(data);
        setIsRunningAgent(false);
        setShowChat(true);
      } else if (data.status === 'failed') {
        setIsRunningAgent(false);
        alert(`Analysis failed: ${data.error || 'Unknown error'}`);
      } else {
        // Still processing, poll again after a delay
        setTimeout(() => pollAnalysisResults(id), 2000);
      }
    } catch (error) {
      console.error('Error polling analysis results:', error);
      setIsRunningAgent(false);
      alert('Failed to retrieve analysis results. Please try again later.');
    }
  };

  return (
    <div className="flex h-screen bg-gray-100 dark:bg-gray-900">
      <Sidebar />
      
      <div className="flex flex-col flex-1 overflow-hidden">
        <Header />
        
        <main className="flex-1 overflow-y-auto p-5">
          <div className="container mx-auto">
            <h1 className="text-2xl font-semibold text-gray-800 dark:text-white mb-6">
              Dashboard
            </h1>
            
            {/* New Market Intelligence Query Section */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-5 mb-6">
              <h2 className="text-lg font-medium text-gray-700 dark:text-white mb-4">
                Market Intelligence Query
              </h2>
              <div className="space-y-4">
                <div>
                  <label htmlFor="domain" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Domain
                  </label>
                  <select
                    id="domain"
                    value={domain}
                    onChange={(e) => setDomain(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:text-white"
                  >
                    <option value="">Select a domain</option>
                    {domains.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </div>
                
                <div>
                  <label htmlFor="query" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Query
                  </label>
                  <input
                    id="query"
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="e.g., AI innovations in K-12 EdTech"
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:text-white"
                  />
                </div>
                
                {/* File Upload Section */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Upload Files (Optional)
                  </label>
                  <FileUpload onFilesUploaded={handleFilesUploaded} />
                </div>
                
                <div className="pt-2">
                  <button
                    type="button"
                    onClick={handleRunAgent}
                    disabled={isRunningAgent || !query.trim()}
                    className={`w-full flex justify-center py-3 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white ${
                      isRunningAgent || !query.trim()
                        ? 'bg-indigo-400 cursor-not-allowed'
                        : 'bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500'
                    }`}
                  >
                    {isRunningAgent ? (
                      <>
                        <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Running Analysis...
                      </>
                    ) : (
                      'Run Agent'
                    )}
                  </button>
                </div>
              </div>
            </div>
            
            {/* Analysis Results Section */}
            {analysisResults && (
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-5 mb-6">
                <h2 className="text-lg font-medium text-gray-700 dark:text-white mb-4">
                  Analysis Results
                </h2>
                <div className="space-y-6">
                  <div>
                    <h3 className="text-md font-medium text-gray-700 dark:text-white mb-2">Summary</h3>
                    <p className="text-gray-600 dark:text-gray-300">{analysisResults.summary}</p>
                  </div>
                  
                  <div>
                    <h3 className="text-md font-medium text-gray-700 dark:text-white mb-2">Competitor Insights</h3>
                    <div className="overflow-x-auto">
                      <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                        <thead className="bg-gray-50 dark:bg-gray-700">
                          <tr>
                            <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Competitor</th>
                            <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Market Share</th>
                            <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Key Strengths</th>
                          </tr>
                        </thead>
                        <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                          {analysisResults.competitor_analysis?.competitors?.map((competitor: any, index: number) => (
                            <tr key={index}>
                              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">{competitor.name}</td>
                              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">{competitor.market_share}</td>
                              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">{competitor.strengths.join(', ')}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="text-md font-medium text-gray-700 dark:text-white mb-2">Market Trends</h3>
                    <div className="overflow-x-auto">
                      <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                        <thead className="bg-gray-50 dark:bg-gray-700">
                          <tr>
                            <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Trend</th>
                            <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Impact</th>
                            <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Timeframe</th>
                          </tr>
                        </thead>
                        <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                          {analysisResults.market_trends?.trends?.map((trend: any, index: number) => (
                            <tr key={index}>
                              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">{trend.trend}</td>
                              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">{trend.impact}</td>
                              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">{trend.timeframe}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="text-md font-medium text-gray-700 dark:text-white mb-2">Customer Insights</h3>
                    <div className="overflow-x-auto">
                      <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                        <thead className="bg-gray-50 dark:bg-gray-700">
                          <tr>
                            <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Segment</th>
                            <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Needs</th>
                            <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Growth Rate</th>
                          </tr>
                        </thead>
                        <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                          {analysisResults.customer_insights?.segments?.map((segment: any, index: number) => (
                            <tr key={index}>
                              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">{segment.segment}</td>
                              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">{segment.needs.join(', ')}</td>
                              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">{segment.growth_rate}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                  
                  <div className="flex space-x-4">
                    <Link 
                      to="/competitor-analysis" 
                      className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 transition"
                    >
                      View Detailed Competitor Analysis
                    </Link>
                    <Link 
                      to="/market-trends" 
                      className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 transition"
                    >
                      View Market Trends
                    </Link>
                    <Link 
                      to="/customer-insights" 
                      className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 transition"
                    >
                      View Customer Insights
                    </Link>
                  </div>
                </div>
              </div>
            )}
            
            {/* Chat Interface */}
            {showChat && (
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-5 mb-6">
                <h2 className="text-lg font-medium text-gray-700 dark:text-white mb-4">
                  Ask Questions About Your Analysis
                </h2>
                <div className="h-96">
                  <ChatInterface 
                    analysisType="general" 
                    analysisId={analysisId}
                    placeholder="Ask questions about your analysis results..."
                  />
                </div>
              </div>
            )}
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-6">
              {/* Recent Analysis Card */}
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-5">
                <h2 className="text-lg font-medium text-gray-700 dark:text-white mb-4">Recent Analysis</h2>
                {isLoading ? (
                  <div className="animate-pulse flex flex-col space-y-4">
                    <div className="h-4 bg-gray-300 dark:bg-gray-600 rounded w-3/4"></div>
                    <div className="h-4 bg-gray-300 dark:bg-gray-600 rounded w-1/2"></div>
                    <div className="h-4 bg-gray-300 dark:bg-gray-600 rounded w-5/6"></div>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <p className="text-gray-600 dark:text-gray-300">No recent analyses</p>
                    <Link 
                      to="/data-integration" 
                      className="inline-block text-indigo-600 dark:text-indigo-400 hover:text-indigo-800 dark:hover:text-indigo-300"
                    >
                      Start a new analysis →
                    </Link>
                  </div>
                )}
              </div>
              
              {/* API Status Card */}
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-5">
                <h2 className="text-lg font-medium text-gray-700 dark:text-white mb-4">API Status</h2>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600 dark:text-gray-300">Google API</span>
                    <span className={`px-2 py-1 text-xs rounded-full ${
                      apiStatus.google_api 
                        ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                        : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                    }`}>
                      {apiStatus.google_api ? 'Configured' : 'Not Configured'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600 dark:text-gray-300">News API</span>
                    <span className={`px-2 py-1 text-xs rounded-full ${
                      apiStatus.news_api 
                        ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                        : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                    }`}>
                      {apiStatus.news_api ? 'Configured' : 'Not Configured'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600 dark:text-gray-300">Alpha Vantage</span>
                    <span className={`px-2 py-1 text-xs rounded-full ${
                      apiStatus.alpha_vantage 
                        ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                        : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                    }`}>
                      {apiStatus.alpha_vantage ? 'Configured' : 'Not Configured'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600 dark:text-gray-300">Tavily API</span>
                    <span className={`px-2 py-1 text-xs rounded-full ${
                      apiStatus.tavily_api 
                        ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                        : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                    }`}>
                      {apiStatus.tavily_api ? 'Configured' : 'Not Configured'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600 dark:text-gray-300">Gemini API</span>
                    <span className={`px-2 py-1 text-xs rounded-full ${
                      apiStatus.gemini_api 
                        ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                        : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                    }`}>
                      {apiStatus.gemini_api ? 'Configured' : 'Not Configured'}
                    </span>
                  </div>
                  <Link 
                    to="/data-integration" 
                    className="inline-block text-indigo-600 dark:text-indigo-400 hover:text-indigo-800 dark:hover:text-indigo-300"
                  >
                    Configure API Keys →
                  </Link>
                </div>
              </div>
              
              {/* Quick Actions Card */}
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-5">
                <h2 className="text-lg font-medium text-gray-700 dark:text-white mb-4">Quick Actions</h2>
                <div className="space-y-3">
                  <Link 
                    to="/data-integration" 
                    className="block p-3 bg-gray-100 dark:bg-gray-700 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600 transition"
                  >
                    <span className="text-gray-700 dark:text-gray-200">Configure Data Sources</span>
                  </Link>
                  <Link 
                    to="/market-trends" 
                    className="block p-3 bg-gray-100 dark:bg-gray-700 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600 transition"
                  >
                    <span className="text-gray-700 dark:text-gray-200">Analyze Market Trends</span>
                  </Link>
                  <Link 
                    to="/competitor-analysis" 
                    className="block p-3 bg-gray-100 dark:bg-gray-700 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600 transition"
                  >
                    <span className="text-gray-700 dark:text-gray-200">Competitor Analysis</span>
                  </Link>
                </div>
              </div>
            </div>
            
            {/* Welcome Message */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-5 mb-6">
              <h2 className="text-lg font-medium text-gray-700 dark:text-white mb-2">
                Welcome, {user?.full_name || user?.email || 'User'}!
              </h2>
              <p className="text-gray-600 dark:text-gray-300 mb-4">
                Get started with Market Intelligence Agent by configuring your data sources and API keys.
                Once set up, you can analyze market trends, track competitors, and gain valuable insights.
              </p>
              <div className="flex space-x-4">
                <Link 
                  to="/data-integration" 
                  className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 transition"
                >
                  Configure Data Sources
                </Link>
                <a 
                  href="https://github.com/gonimohan/market-intelligence-agent" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded-md hover:bg-gray-300 dark:hover:bg-gray-600 transition"
                >
                  View Documentation
                </a>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};

export default DashboardPage;
