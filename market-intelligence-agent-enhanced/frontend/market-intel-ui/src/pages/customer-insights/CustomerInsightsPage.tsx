import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import Sidebar from '../../components/layout/Sidebar';
import Header from '../../components/layout/Header';
import ChatInterface from '../../components/chat/ChatInterface';

const CustomerInsightsPage: React.FC = () => {
  const [industry, setIndustry] = useState('');
  const [customerSegment, setCustomerSegment] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<any>(null);
  const [showChat, setShowChat] = useState(false);
  const [analysisId, setAnalysisId] = useState<string | undefined>(undefined);

  const industries = [
    { value: 'technology', label: 'Technology' },
    { value: 'healthcare', label: 'Healthcare' },
    { value: 'finance', label: 'Finance' },
    { value: 'retail', label: 'Retail' },
    { value: 'manufacturing', label: 'Manufacturing' },
    { value: 'energy', label: 'Energy' },
    { value: 'education', label: 'Education' }
  ];

  const customerSegments = [
    { value: 'all', label: 'All Segments' },
    { value: 'enterprise', label: 'Enterprise' },
    { value: 'smb', label: 'Small & Medium Business' },
    { value: 'consumer', label: 'Consumer' },
    { value: 'government', label: 'Government & Public Sector' },
    { value: 'education', label: 'Education' }
  ];

  const handleAnalyzeCustomerInsights = async () => {
    if (!industry || !customerSegment) {
      alert('Please select both industry and customer segment');
      return;
    }

    setIsAnalyzing(true);
    
    try {
      // Make actual API call to backend
      const response = await fetch('/customer-insights-analysis', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          industry,
          customer_segment: customerSegment
        })
      });
      
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      
      const data = await response.json();
      setAnalysisResults(data.results);
      setAnalysisId(data.analysis_id);
      setShowChat(true);
    } catch (error) {
      console.error('Error analyzing customer insights:', error);
      alert('Failed to analyze customer insights. Please try again later.');
    } finally {
      setIsAnalyzing(false);
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
              Customer Insights
            </h1>
            
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Left Column - Input Form */}
              <div className="lg:col-span-2">
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-5 mb-6">
                  <h2 className="text-lg font-medium text-gray-700 dark:text-white mb-4">
                    Analyze Customer Insights
                  </h2>
                  <p className="text-gray-600 dark:text-gray-300 mb-6">
                    Select your industry and customer segment to analyze customer behavior, preferences, and needs.
                  </p>
                  
                  <div className="space-y-4">
                    <div>
                      <label htmlFor="industry" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        Industry
                      </label>
                      <select
                        id="industry"
                        value={industry}
                        onChange={(e) => setIndustry(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:text-white"
                      >
                        <option value="">Select an industry</option>
                        {industries.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    </div>
                    
                    <div>
                      <label htmlFor="customerSegment" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        Customer Segment
                      </label>
                      <select
                        id="customerSegment"
                        value={customerSegment}
                        onChange={(e) => setCustomerSegment(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:text-white"
                      >
                        <option value="">Select a customer segment</option>
                        {customerSegments.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    </div>
                    
                    <div className="pt-2">
                      <button
                        type="button"
                        onClick={handleAnalyzeCustomerInsights}
                        disabled={isAnalyzing || !industry || !customerSegment}
                        className={`w-full flex justify-center py-3 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white ${
                          isAnalyzing || !industry || !customerSegment
                            ? 'bg-indigo-400 cursor-not-allowed'
                            : 'bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500'
                        }`}
                      >
                        {isAnalyzing ? (
                          <>
                            <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            Analyzing Customer Insights...
                          </>
                        ) : (
                          'Analyze Customer Insights'
                        )}
                      </button>
                    </div>
                  </div>
                </div>
                
                {/* Analysis Results */}
                {analysisResults && (
                  <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-5 mb-6">
                    <h2 className="text-lg font-medium text-gray-700 dark:text-white mb-4">
                      Customer Insights Analysis
                    </h2>
                    
                    <div className="space-y-6">
                      <div>
                        <h3 className="text-md font-medium text-gray-700 dark:text-white mb-2">Customer Segments</h3>
                        <div className="overflow-x-auto">
                          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                            <thead className="bg-gray-50 dark:bg-gray-700">
                              <tr>
                                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Segment</th>
                                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Needs</th>
                                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Pain Points</th>
                                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Growth Rate</th>
                              </tr>
                            </thead>
                            <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                              {analysisResults.segments.map((segment: any, index: number) => (
                                <tr key={index}>
                                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">{segment.segment}</td>
                                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">{segment.needs.join(', ')}</td>
                                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">{segment.pain_points.join(', ')}</td>
                                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">{segment.growth_rate}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div>
                          <h3 className="text-md font-medium text-gray-700 dark:text-white mb-2">Sentiment Analysis</h3>
                          <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-md">
                            <div className="mb-3">
                              <span className="font-medium text-gray-700 dark:text-gray-300">Overall Sentiment:</span>
                              <span className="ml-2 text-gray-600 dark:text-gray-400">{analysisResults.sentiment_analysis.overall_sentiment}</span>
                            </div>
                            <div className="mb-3">
                              <span className="font-medium text-gray-700 dark:text-gray-300">Key Positive Themes:</span>
                              <span className="ml-2 text-gray-600 dark:text-gray-400">{analysisResults.sentiment_analysis.key_positive_themes.join(', ')}</span>
                            </div>
                            <div>
                              <span className="font-medium text-gray-700 dark:text-gray-300">Key Negative Themes:</span>
                              <span className="ml-2 text-gray-600 dark:text-gray-400">{analysisResults.sentiment_analysis.key_negative_themes.join(', ')}</span>
                            </div>
                          </div>
                        </div>
                        
                        <div>
                          <h3 className="text-md font-medium text-gray-700 dark:text-white mb-2">Buying Behavior</h3>
                          <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-md">
                            <div className="mb-3">
                              <span className="font-medium text-gray-700 dark:text-gray-300">Decision Makers:</span>
                              <span className="ml-2 text-gray-600 dark:text-gray-400">{analysisResults.buying_behavior.decision_makers.join(', ')}</span>
                            </div>
                            <div className="mb-3">
                              <span className="font-medium text-gray-700 dark:text-gray-300">Purchase Cycle:</span>
                              <span className="ml-2 text-gray-600 dark:text-gray-400">{analysisResults.buying_behavior.purchase_cycle}</span>
                            </div>
                            <div>
                              <span className="font-medium text-gray-700 dark:text-gray-300">Key Influences:</span>
                              <span className="ml-2 text-gray-600 dark:text-gray-400">{analysisResults.buying_behavior.key_influences.join(', ')}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
              
              {/* Right Column - Chat Interface */}
              <div className="lg:col-span-1">
                {showChat ? (
                  <div className="bg-white dark:bg-gray-800 rounded-lg shadow h-full">
                    <ChatInterface 
                      analysisType="customer" 
                      analysisId={analysisId}
                      placeholder="Ask about customer insights..."
                    />
                  </div>
                ) : (
                  <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-5 h-full flex flex-col items-center justify-center text-center">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 text-gray-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                    </svg>
                    <h3 className="text-lg font-medium text-gray-700 dark:text-white mb-2">Chat Assistant</h3>
                    <p className="text-gray-500 dark:text-gray-400 mb-4">
                      Run customer insights analysis to chat with the AI assistant about your results.
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};

export default CustomerInsightsPage;
