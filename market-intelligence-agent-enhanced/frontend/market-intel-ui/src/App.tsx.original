import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';
import { Toaster } from 'react-hot-toast';

// Auth
import { AuthProvider } from './contexts/AuthContext';
import ProtectedRoute from './components/auth/ProtectedRoute';

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

// Create a client for React Query
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <Router>
          <Routes>
            {/* Auth Routes */}
            <Route path="/login" element={<LoginPage />} />
            <Route path="/register" element={<RegisterPage />} />
            <Route path="/forgot-password" element={<ForgotPasswordPage />} />
            
            {/* Protected Routes */}
            <Route path="/dashboard" element={
              <ProtectedRoute>
                <DashboardPage />
              </ProtectedRoute>
            } />
            <Route path="/data-integration" element={
              <ProtectedRoute>
                <DataIntegrationPage />
              </ProtectedRoute>
            } />
            <Route path="/competitor-analysis" element={
              <ProtectedRoute>
                <CompetitorAnalysisPage />
              </ProtectedRoute>
            } />
            <Route path="/market-trends" element={
              <ProtectedRoute>
                <MarketTrendsPage />
              </ProtectedRoute>
            } />
            <Route path="/customer-insights" element={
              <ProtectedRoute>
                <CustomerInsightsPage />
              </ProtectedRoute>
            } />
            <Route path="/downloads" element={
              <ProtectedRoute>
                <DownloadsPage />
              </ProtectedRoute>
            } />
            
            {/* Redirect root to dashboard if logged in, otherwise to login */}
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            
            {/* Catch all - redirect to dashboard */}
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Routes>
        </Router>
        <Toaster position="top-right" />
      </AuthProvider>
    </QueryClientProvider>
  );
}

export default App;
