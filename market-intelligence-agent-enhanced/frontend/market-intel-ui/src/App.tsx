import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';

// Pages
import LoginPage from './pages/auth/LoginPage';
import RegisterPage from './pages/auth/RegisterPage';
import ForgotPasswordPage from './pages/auth/ForgotPasswordPage';
import DashboardPage from './pages/dashboard/DashboardPage';
import DataIntegrationPage from './pages/data-integration/DataIntegrationPage';
import CompetitorAnalysisPage from './pages/competitor-analysis/CompetitorAnalysisPage';
import MarketTrendsPage from './pages/market-trends/MarketTrendsPage';
import CustomerInsightsPage from './pages/customer-insights/CustomerInsightsPage';
import DownloadsPage from './pages/downloads/DownloadsPage';

// Create a simplified demo app that bypasses authentication
function App() {
  // Always redirect to dashboard for demo purposes
  return (
    <Router>
      <Routes>
        {/* Auth Routes - all redirect to dashboard for demo */}
        <Route path="/login" element={<Navigate to="/dashboard" replace />} />
        <Route path="/register" element={<Navigate to="/dashboard" replace />} />
        <Route path="/forgot-password" element={<Navigate to="/dashboard" replace />} />
        
        {/* Direct Routes without Protection */}
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/data-integration" element={<DataIntegrationPage />} />
        <Route path="/competitor-analysis" element={<CompetitorAnalysisPage />} />
        <Route path="/market-trends" element={<MarketTrendsPage />} />
        <Route path="/customer-insights" element={<CustomerInsightsPage />} />
        <Route path="/downloads" element={<DownloadsPage />} />
        
        {/* Redirect root to dashboard */}
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        
        {/* Catch all - redirect to dashboard */}
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Routes>
      <Toaster position="top-right" />
    </Router>
  );
}

export default App;
