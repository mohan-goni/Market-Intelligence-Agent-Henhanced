import React, { useState, useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { toast } from 'react-hot-toast';
import Sidebar from '../../components/layout/Sidebar';
import Header from '../../components/layout/Header';
import { apiKeyService, ApiKeys } from '../../services/authService';

const DataIntegrationPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('data-sources');
  const [isLoading, setIsLoading] = useState(false);
  const { register, handleSubmit, setValue } = useForm<ApiKeys>();

  useEffect(() => {
    const fetchApiKeys = async () => {
      try {
        setIsLoading(true);
        const keys = await apiKeyService.getApiKeys();
        if (keys) {
          setValue('google_api_key', keys.google_api_key || '');
          setValue('newsapi_key', keys.newsapi_key || '');
          setValue('alpha_vantage_key', keys.alpha_vantage_key || '');
          setValue('tavily_api_key', keys.tavily_api_key || '');
        }
      } catch (error) {
        console.error('Error fetching API keys:', error);
        toast.error('Failed to load API keys');
      } finally {
        setIsLoading(false);
      }
    };

    fetchApiKeys();
  }, [setValue]);

  const onSubmit = async (data: ApiKeys) => {
    try {
      setIsLoading(true);
      await apiKeyService.setApiKeys(data);
      toast.success('API keys updated successfully');
    } catch (error) {
      console.error('Error updating API keys:', error);
      toast.error('Failed to update API keys');
    } finally {
      setIsLoading(false);
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
              Data Integration
            </h1>
            <p className="text-gray-600 dark:text-gray-300 mb-6">
              Configure data sources and API connections for market intelligence gathering.
            </p>
            
            {/* Tabs */}
            <div className="mb-6 border-b border-gray-200 dark:border-gray-700">
              <ul className="flex flex-wrap -mb-px">
                <li className="mr-2">
                  <button
                    className={`inline-block p-4 rounded-t-lg ${
                      activeTab === 'data-sources'
                        ? 'text-indigo-600 border-b-2 border-indigo-600 dark:text-indigo-400 dark:border-indigo-400'
                        : 'text-gray-500 hover:text-gray-600 dark:text-gray-400 dark:hover:text-gray-300'
                    }`}
                    onClick={() => setActiveTab('data-sources')}
                  >
                    Data Sources
                  </button>
                </li>
                <li className="mr-2">
                  <button
                    className={`inline-block p-4 rounded-t-lg ${
                      activeTab === 'supabase'
                        ? 'text-indigo-600 border-b-2 border-indigo-600 dark:text-indigo-400 dark:border-indigo-400'
                        : 'text-gray-500 hover:text-gray-600 dark:text-gray-400 dark:hover:text-gray-300'
                    }`}
                    onClick={() => setActiveTab('supabase')}
                  >
                    Supabase Connection
                  </button>
                </li>
                <li className="mr-2">
                  <button
                    className={`inline-block p-4 rounded-t-lg ${
                      activeTab === 'api-keys'
                        ? 'text-indigo-600 border-b-2 border-indigo-600 dark:text-indigo-400 dark:border-indigo-400'
                        : 'text-gray-500 hover:text-gray-600 dark:text-gray-400 dark:hover:text-gray-300'
                    }`}
                    onClick={() => setActiveTab('api-keys')}
                  >
                    API Keys
                  </button>
                </li>
              </ul>
            </div>
            
            {/* Tab Content */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
              {/* Data Sources Tab */}
              {activeTab === 'data-sources' && (
                <div>
                  <h2 className="text-lg font-medium text-gray-700 dark:text-white mb-4">Data Sources</h2>
                  <p className="text-gray-600 dark:text-gray-300 mb-6">
                    Configure and manage your data sources for market intelligence gathering.
                  </p>
                  
                  <button className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 transition mb-6">
                    Add New Data Source
                  </button>
                  
                  <div className="space-y-4">
                    <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center">
                          <span className="text-blue-500 mr-2">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                              <path fillRule="evenodd" d="M4 4a2 2 0 012-2h8a2 2 0 012 2v12a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm3 1h6v4H7V5zm8 8v2h1v1H4v-1h1v-2a1 1 0 011-1h8a1 1 0 011 1z" clipRule="evenodd" />
                            </svg>
                          </span>
                          <span className="font-medium text-gray-700 dark:text-white">News API</span>
                        </div>
                        <span className="px-2 py-1 text-xs rounded-full bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                          Active
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-300">
                        Access to news articles from various sources worldwide.
                      </p>
                    </div>
                    
                    <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center">
                          <span className="text-purple-500 mr-2">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                              <path d="M2 11a1 1 0 011-1h2a1 1 0 011 1v5a1 1 0 01-1 1H3a1 1 0 01-1-1v-5zM8 7a1 1 0 011-1h2a1 1 0 011 1v9a1 1 0 01-1 1H9a1 1 0 01-1-1V7zM14 4a1 1 0 011-1h2a1 1 0 011 1v12a1 1 0 01-1 1h-2a1 1 0 01-1-1V4z" />
                            </svg>
                          </span>
                          <span className="font-medium text-gray-700 dark:text-white">Alpha Vantage</span>
                        </div>
                        <span className="px-2 py-1 text-xs rounded-full bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                          Active
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-300">
                        Financial market data and indicators.
                      </p>
                    </div>
                    
                    <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center">
                          <span className="text-green-500 mr-2">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                              <path fillRule="evenodd" d="M4.083 9h1.946c.089-1.546.383-2.97.837-4.118A6.004 6.004 0 004.083 9zM10 2a8 8 0 100 16 8 8 0 000-16zm0 2c-.076 0-.232.032-.465.262-.238.234-.497.623-.737 1.182-.389.907-.673 2.142-.766 3.556h3.936c-.093-1.414-.377-2.649-.766-3.556-.24-.56-.5-.948-.737-1.182C10.232 4.032 10.076 4 10 4zm3.971 5c-.089-1.546-.383-2.97-.837-4.118A6.004 6.004 0 0115.917 9h-1.946zm-2.003 2H8.032c.093 1.414.377 2.649.766 3.556.24.56.5.948.737 1.182.233.23.389.262.465.262.076 0 .232-.032.465-.262.238-.234.498-.623.737-1.182.389-.907.673-2.142.766-3.556zm1.166 4.118c.454-1.147.748-2.572.837-4.118h1.946a6.004 6.004 0 01-2.783 4.118zm-6.268 0C6.412 13.97 6.118 12.546 6.03 11H4.083a6.004 6.004 0 002.783 4.118z" clipRule="evenodd" />
                            </svg>
                          </span>
                          <span className="font-medium text-gray-700 dark:text-white">Market Web Scraper</span>
                        </div>
                        <span className="px-2 py-1 text-xs rounded-full bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
                          Configured
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-300">
                        Custom web scraping for market data from various websites.
                      </p>
                    </div>
                    
                    <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center">
                          <span className="text-yellow-500 mr-2">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-3a1 1 0 00-.867.5 1 1 0 11-1.731-1A3 3 0 0113 8a3.001 3.001 0 01-2 2.83V11a1 1 0 11-2 0v-1a1 1 0 011-1 1 1 0 100-2zm0 8a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
                            </svg>
                          </span>
                          <span className="font-medium text-gray-700 dark:text-white">Tavily Search</span>
                        </div>
                        <span className="px-2 py-1 text-xs rounded-full bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                          Active
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-300">
                        AI-powered search engine for market intelligence.
                      </p>
                    </div>
                    
                    <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center">
                          <span className="text-red-500 mr-2">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                              <path fillRule="evenodd" d="M5 5a3 3 0 015-2.236A3 3 0 0114.83 6H16a2 2 0 110 4h-5V9a1 1 0 10-2 0v1H4a2 2 0 110-4h1.17C5.06 5.687 5 5.35 5 5zm4 1V5a1 1 0 10-1 1h1zm3 0a1 1 0 10-1-1v1h1z" clipRule="evenodd" />
                              <path d="M9 11H3v5a2 2 0 002 2h4v-7zM11 18h4a2 2 0 002-2v-5h-6v7z" />
                            </svg>
                          </span>
                          <span className="font-medium text-gray-700 dark:text-white">Government Data</span>
                        </div>
                        <span className="px-2 py-1 text-xs rounded-full bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200">
                          Ready
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-300">
                        Public government datasets for market analysis.
                      </p>
                    </div>
                  </div>
                </div>
              )}
              
              {/* Supabase Connection Tab */}
              {activeTab === 'supabase' && (
                <div>
                  <h2 className="text-lg font-medium text-gray-700 dark:text-white mb-4">Supabase Connection</h2>
                  <p className="text-gray-600 dark:text-gray-300 mb-6">
                    Configure your Supabase connection for data storage and retrieval.
                  </p>
                  
                  <form className="space-y-6">
                    <div>
                      <label htmlFor="supabase-url" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        Supabase URL
                      </label>
                      <input
                        id="supabase-url"
                        type="text"
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:text-white"
                        placeholder="https://your-project.supabase.co"
                      />
                    </div>
                    
                    <div>
                      <label htmlFor="supabase-key" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        Supabase API Key
                      </label>
                      <input
                        id="supabase-key"
                        type="password"
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:text-white"
                        placeholder="your-supabase-api-key"
                      />
                    </div>
                    
                    <div>
                      <button
                        type="button"
                        className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 transition"
                      >
                        Test Connection
                      </button>
                    </div>
                    
                    <div>
                      <button
                        type="submit"
                        className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition"
                      >
                        Save Connection
                      </button>
                    </div>
                  </form>
                </div>
              )}
              
              {/* API Keys Tab */}
              {activeTab === 'api-keys' && (
                <div>
                  <h2 className="text-lg font-medium text-gray-700 dark:text-white mb-4">API Keys</h2>
                  <p className="text-gray-600 dark:text-gray-300 mb-6">
                    Configure your API keys for external data sources.
                  </p>
                  
                  <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
                    <div>
                      <label htmlFor="google-api-key" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        Google API Key
                      </label>
                      <input
                        id="google-api-key"
                        type="password"
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:text-white"
                        placeholder="Your Google API Key"
                        {...register('google_api_key')}
                      />
                      <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                        Used for Google Generative AI services
                      </p>
                    </div>
                    
                    <div>
                      <label htmlFor="newsapi-key" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        News API Key
                      </label>
                      <input
                        id="newsapi-key"
                        type="password"
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:text-white"
                        placeholder="Your News API Key"
                        {...register('newsapi_key')}
                      />
                      <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                        Used for retrieving news articles
                      </p>
                    </div>
                    
                    <div>
                      <label htmlFor="alpha-vantage-key" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        Alpha Vantage API Key
                      </label>
                      <input
                        id="alpha-vantage-key"
                        type="password"
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:text-white"
                        placeholder="Your Alpha Vantage API Key"
                        {...register('alpha_vantage_key')}
                      />
                      <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                        Used for financial data and market news
                      </p>
                    </div>
                    
                    <div>
                      <label htmlFor="tavily-api-key" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        Tavily API Key
                      </label>
                      <input
                        id="tavily-api-key"
                        type="password"
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:text-white"
                        placeholder="Your Tavily API Key"
                        {...register('tavily_api_key')}
                      />
                      <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                        Used for AI-powered search
                      </p>
                    </div>
                    
                    <div>
                      <button
                        type="submit"
                        disabled={isLoading}
                        className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 transition disabled:opacity-50"
                      >
                        {isLoading ? 'Saving...' : 'Save API Keys'}
                      </button>
                    </div>
                  </form>
                </div>
              )}
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};

export default DataIntegrationPage;
