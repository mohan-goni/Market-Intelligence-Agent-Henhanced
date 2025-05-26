import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from '../../contexts/AuthContext';

interface ChatMessage {
  id: string;
  sender: 'user' | 'agent';
  message: string;
  timestamp: string;
  sources?: string[];
}

interface ChatInterfaceProps {
  analysisType: 'competitor' | 'market' | 'customer' | 'general';
  analysisId?: string;
  placeholder?: string;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ 
  analysisType, 
  analysisId,
  placeholder = 'Ask a question...'
}) => {
  const { user } = useAuth();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [newMessage, setNewMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Add welcome message on component mount
  useEffect(() => {
    const welcomeMessage = getWelcomeMessage(analysisType);
    setMessages([
      {
        id: 'welcome',
        sender: 'agent',
        message: welcomeMessage,
        timestamp: new Date().toISOString()
      }
    ]);
    
    // Load chat history from backend
    fetchChatHistory();
  }, [analysisType, analysisId]);

  // Scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const getWelcomeMessage = (type: string) => {
    switch (type) {
      case 'competitor':
        return "I'm your Competitor Analysis assistant. Ask me about market positioning, competitive advantages, or any insights about your competitors.";
      case 'market':
        return "I'm your Market Trends assistant. Ask me about industry trends, growth opportunities, or market forecasts.";
      case 'customer':
        return "I'm your Customer Insights assistant. Ask me about customer segments, preferences, or behavior patterns.";
      default:
        return "I'm your Market Intelligence assistant. How can I help you today?";
    }
  };

  const fetchChatHistory = async () => {
    try {
      // In production, this would fetch chat history from the backend
      const response = await fetch(`/chat-history?analysis_type=${analysisType}${analysisId ? `&analysis_id=${analysisId}` : ''}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        const formattedMessages = data.map((item: any) => ([
          {
            id: `user-${item.timestamp}`,
            sender: 'user',
            message: item.user_message,
            timestamp: item.timestamp
          },
          {
            id: `agent-${item.timestamp}`,
            sender: 'agent',
            message: item.agent_response,
            timestamp: item.timestamp,
            sources: item.sources
          }
        ])).flat();
        
        setMessages(prev => [prev[0], ...formattedMessages]); // Keep welcome message
      }
    } catch (error) {
      console.error('Error fetching chat history:', error);
    }
  };

  const handleSendMessage = async () => {
    if (!newMessage.trim()) return;
    
    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      sender: 'user',
      message: newMessage,
      timestamp: new Date().toISOString()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setNewMessage('');
    setIsLoading(true);
    
    try {
      // Make actual API call to backend
      const response = await fetch('/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          message: newMessage,
          analysis_type: analysisType,
          analysis_id: analysisId
        })
      });
      
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      
      const data = await response.json();
      
      const agentMessage: ChatMessage = {
        id: `agent-${Date.now()}`,
        sender: 'agent',
        message: data.response,
        timestamp: data.timestamp,
        sources: data.sources
      };
      
      setMessages(prev => [...prev, agentMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      
      const errorMessage: ChatMessage = {
        id: `error-${Date.now()}`,
        sender: 'agent',
        message: "I'm sorry, I encountered an error processing your request. Please try again later.",
        timestamp: new Date().toISOString()
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Chat Header */}
      <div className="px-4 py-3 bg-indigo-600 text-white rounded-t-lg">
        <h3 className="text-lg font-medium">Chat Assistant</h3>
        <p className="text-sm text-indigo-200">
          {analysisType === 'competitor' && 'Competitor Analysis'}
          {analysisType === 'market' && 'Market Trends'}
          {analysisType === 'customer' && 'Customer Insights'}
          {analysisType === 'general' && 'General Intelligence'}
        </p>
      </div>
      
      {/* Messages Container */}
      <div className="flex-1 p-4 overflow-y-auto bg-gray-50 dark:bg-gray-700">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`mb-4 ${
              msg.sender === 'user' ? 'flex justify-end' : 'flex justify-start'
            }`}
          >
            <div
              className={`max-w-3/4 rounded-lg px-4 py-2 ${
                msg.sender === 'user'
                  ? 'bg-indigo-600 text-white'
                  : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200'
              }`}
            >
              <p className="text-sm">{msg.message}</p>
              {msg.sources && msg.sources.length > 0 && (
                <div className="mt-2 pt-2 border-t border-gray-200 dark:border-gray-600">
                  <p className="text-xs font-medium mb-1">Sources:</p>
                  <ul className="text-xs list-disc pl-4">
                    {msg.sources.map((source, index) => (
                      <li key={index}>{source}</li>
                    ))}
                  </ul>
                </div>
              )}
              <p className="text-xs text-right mt-1 opacity-70">
                {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </p>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      
      {/* Input Area */}
      <div className="p-4 border-t border-gray-200 dark:border-gray-600">
        <div className="flex">
          <textarea
            value={newMessage}
            onChange={(e) => setNewMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={placeholder}
            className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-l-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:text-white"
            rows={2}
            disabled={isLoading}
          />
          <button
            onClick={handleSendMessage}
            disabled={isLoading || !newMessage.trim()}
            className={`px-4 py-2 ${
              isLoading || !newMessage.trim()
                ? 'bg-indigo-400 cursor-not-allowed'
                : 'bg-indigo-600 hover:bg-indigo-700'
            } text-white rounded-r-md transition`}
          >
            {isLoading ? (
              <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-8.707l-3-3a1 1 0 00-1.414 1.414L10.586 9H7a1 1 0 100 2h3.586l-1.293 1.293a1 1 0 101.414 1.414l3-3a1 1 0 000-1.414z" clipRule="evenodd" />
              </svg>
            )}
          </button>
        </div>
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
          Press Enter to send, Shift+Enter for new line
        </p>
      </div>
    </div>
  );
};

export default ChatInterface;
