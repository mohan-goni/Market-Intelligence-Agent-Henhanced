import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import Sidebar from '../../components/layout/Sidebar';
import Header from '../../components/layout/Header';
import ChatInterface from '../../components/chat/ChatInterface';

const CompetitorAnalysisPage: React.FC = () => {
  const [industry, setIndustry] = useState('');
  const [competitor, setCompetitor] = useState('');
  const [competitors, setCompetitors] = useState<string[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<any>(null);
  const [showChat, setShowChat] = useState(false);

  const industries = [
    { value: 'technology', label: 'Technology' },
    { value: 'healthcare', label: 'Healthcare' },
    { value: 'finance', label: 'Finance' },
    { value: 'retail', label: 'Retail' },
    { value: 'manufacturing', label: 'Manufacturing' },
    { value: 'energy', label: 'Energy' },
    { value: 'education', label: 'Education' }
  ];

  const handleAddCompetitor = () => {
    if (competitor.trim() && !competitors.includes(competitor.trim())) {
      setCompetitors([...competitors, competitor.trim()]);
      setCompetitor('');
    }
  };

  const handleRemoveCompetitor = (index: number) => {
    setCompetitors(competitors.filter((_, i) => i !== index));
  };

  const handleAnalyzeCompetitors = () => {
    if (!industry || competitors.length === 0) {
      alert('Please select an industry and add at least one competitor');
      return;
    }

    setIsAnalyzing(true);
    
    // Simulate API call with timeout
    setTimeout(() => {
      setAnalysisResults({
        competitors: competitors.map((name, index) => ({
          name,
          marketShare: `${Math.floor(Math.random() * 30) + 10}%`,
          strengths: ['Innovation', 'Brand recognition', 'Customer service'][index % 3],
          weaknesses: ['High prices', 'Limited market reach', 'Slow innovation'][index % 3],
          recentDevelopments: ['Launched new product line', 'Expanded to new market', 'Acquired smaller competitor'][index % 3]
        })),
        marketPositioning: {
          priceLeaders: [competitors[0]],
          qualityLeaders: [competitors[competitors.length - 1]],
          innovationLeaders: [competitors[Math.floor(competitors.length / 2)]]
        },
        competitiveLandscape: {
          fragmentation: 'Medium',
          barriersToEntry: 'High due to established brand loyalty',
          threatOfNewEntrants: 'Medium'
        }
      });
      setIsAnalyzing(false);
      setShowChat(true);
    }, 2000);
  };

  return (
    <div className="flex h-screen bg-gray-100 dark:bg-gray-900">
      <Sidebar />
      
      <div className="flex flex-col flex-1 overflow-hidden">
        <Header />
        
        <main className="flex-1 overflow-y-auto p-5">
          <div className="container mx-auto">
            <h1 className="text-2xl font-semibold text-gray-800 dark:text-white mb-6">
              Competitor Analysis
            </h1>
            
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Left Column - Input Form */}
              <div className="lg:col-span-2">
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-5 mb-6">
                  <h2 className="text-lg font-medium text-gray-700 dark:text-white mb-4">
                    Analyze Competitors
                  </h2>
                  <p className="text-gray-600 dark:text-gray-300 mb-6">
                    Enter your competitors and select your industry to analyze competitive landscape.
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
                      <label htmlFor="competitor" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        Competitors
                      </label>
                      <div className="flex">
                        <input
                          id="competitor"
                          type="text"
                          value={competitor}
                          onChange={(e) => setCompetitor(e.target.value)}
                          placeholder="Enter competitor name"
                          className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-l-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:text-white"
                        />
                        <button
                          type="button"
                          onClick={handleAddCompetitor}
                          className="px-4 py-2 bg-indigo-600 text-white rounded-r-md hover:bg-indigo-700 transition"
                        >
                          Add
                        </button>
                      </div>
                    </div>
                    
                    {competitors.length > 0 && (
                      <div className="mt-2">
                        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Added Competitors:</h4>
                        <ul className="space-y-2">
                          {competitors.map((comp, index) => (
                            <li key={index} className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-700 rounded-md">
                              <span className="text-sm text-gray-600 dark:text-gray-300">{comp}</span>
                              <button
                                type="button"
                                onClick={() => handleRemoveCompetitor(index)}
                                className="text-red-500 hover:text-red-700"
                              >
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                                  <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                                </svg>
                              </button>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                    
                    <div className="pt-2">
                      <button
                        type="button"
                        onClick={handleAnalyzeCompetitors}
                        disabled={isAnalyzing || !industry || competitors.length === 0}
                        className={`w-full flex justify-center py-3 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white ${
                          isAnalyzing || !industry || competitors.length === 0
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
                            Analyzing Competitors...
                          </>
                        ) : (
                          'Analyze Competitors'
                        )}
                      </button>
                    </div>
                  </div>
                </div>
                
                {/* Analysis Results */}
                {analysisResults && (
                  <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-5 mb-6">
                    <h2 className="text-lg font-medium text-gray-700 dark:text-white mb-4">
                      Analysis Results
                    </h2>
                    
                    <div className="space-y-6">
                      <div>
                        <h3 className="text-md font-medium text-gray-700 dark:text-white mb-2">Competitor Profiles</h3>
                        <div className="overflow-x-auto">
                          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                            <thead className="bg-gray-50 dark:bg-gray-700">
                              <tr>
                                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Competitor</th>
                                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Market Share</th>
                                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Key Strengths</th>
                                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Key Weaknesses</th>
                                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Recent Developments</th>
                              </tr>
                            </thead>
                            <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                              {analysisResults.competitors.map((competitor: any, index: number) => (
                                <tr key={index}>
                                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">{competitor.name}</td>
                                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">{competitor.marketShare}</td>
                                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">{competitor.strengths}</td>
                                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">{competitor.weaknesses}</td>
                                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">{competitor.recentDevelopments}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div>
                          <h3 className="text-md font-medium text-gray-700 dark:text-white mb-2">Market Positioning</h3>
                          <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-md">
                            <div className="mb-3">
                              <span className="font-medium text-gray-700 dark:text-gray-300">Price Leaders:</span>
                              <span className="ml-2 text-gray-600 dark:text-gray-400">{analysisResults.marketPositioning.priceLeaders.join(', ')}</span>
                            </div>
                            <div className="mb-3">
                              <span className="font-medium text-gray-700 dark:text-gray-300">Quality Leaders:</span>
                              <span className="ml-2 text-gray-600 dark:text-gray-400">{analysisResults.marketPositioning.qualityLeaders.join(', ')}</span>
                            </div>
                            <div>
                              <span className="font-medium text-gray-700 dark:text-gray-300">Innovation Leaders:</span>
                              <span className="ml-2 text-gray-600 dark:text-gray-400">{analysisResults.marketPositioning.innovationLeaders.join(', ')}</span>
                            </div>
                          </div>
                        </div>
                        
                        <div>
                          <h3 className="text-md font-medium text-gray-700 dark:text-white mb-2">Competitive Landscape</h3>
                          <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-md">
                            <div className="mb-3">
                              <span className="font-medium text-gray-700 dark:text-gray-300">Market Fragmentation:</span>
                              <span className="ml-2 text-gray-600 dark:text-gray-400">{analysisResults.competitiveLandscape.fragmentation}</span>
                            </div>
                            <div className="mb-3">
                              <span className="font-medium text-gray-700 dark:text-gray-300">Barriers to Entry:</span>
                              <span className="ml-2 text-gray-600 dark:text-gray-400">{analysisResults.competitiveLandscape.barriersToEntry}</span>
                            </div>
                            <div>
                              <span className="font-medium text-gray-700 dark:text-gray-300">Threat of New Entrants:</span>
                              <span className="ml-2 text-gray-600 dark:text-gray-400">{analysisResults.competitiveLandscape.threatOfNewEntrants}</span>
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
                      analysisType="competitor" 
                      analysisId={analysisResults ? "competitor-analysis-1" : undefined}
                      placeholder="Ask about competitor analysis..."
                    />
                  </div>
                ) : (
                  <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-5 h-full flex flex-col items-center justify-center text-center">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 text-gray-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                    </svg>
                    <h3 className="text-lg font-medium text-gray-700 dark:text-white mb-2">Chat Assistant</h3>
                    <p className="text-gray-500 dark:text-gray-400 mb-4">
                      Run competitor analysis to chat with the AI assistant about your results.
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

export default CompetitorAnalysisPage;
