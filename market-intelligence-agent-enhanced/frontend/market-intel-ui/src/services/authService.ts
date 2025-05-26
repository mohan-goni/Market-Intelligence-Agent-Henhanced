import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000,
});

// Add request interceptor to add auth token to requests
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Add response interceptor to handle common errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // Handle 401 Unauthorized errors (token expired)
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    // Handle network errors
    if (!error.response) {
      console.error('Network error:', error);
      throw new Error('Network error occurred. Please check your connection.');
    }
    return Promise.reject(error);
  }
);

export interface LoginResponse {
  access_token: string;
  token_type: string;
  user: {
    email: string;
    full_name?: string;
  };
}

export interface User {
  email: string;
  full_name?: string;
}

export interface ApiKeys {
  google_api_key?: string;
  newsapi_key?: string;
  alpha_vantage_key?: string;
  tavily_api_key?: string;
}

export const authService = {
  async login(email: string, password: string): Promise<LoginResponse> {
    try {
      const formData = new FormData();
      formData.append('username', email);
      formData.append('password', password);
      
      const response = await api.post('/token', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      return response.data;
    } catch (error) {
      console.error('Login error:', error);
      throw error;
    }
  },
  
  async register(email: string, password: string, fullName?: string): Promise<User> {
    try {
      const formData = new FormData();
      formData.append('email', email);
      formData.append('password', password);
      if (fullName) {
        formData.append('full_name', fullName);
      }
      
      const response = await api.post('/users', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      return response.data;
    } catch (error) {
      console.error('Registration error:', error);
      throw error;
    }
  },
  
  async getCurrentUser(): Promise<User> {
    try {
      const response = await api.get('/users/me');
      return response.data;
    } catch (error) {
      console.error('Get current user error:', error);
      throw error;
    }
  },
  
  async forgotPassword(email: string): Promise<void> {
    try {
      await api.post('/reset-password', { email });
    } catch (error) {
      console.error('Password reset error:', error);
      throw error;
    }
  },
};

export const apiKeyService = {
  async getApiKeys(): Promise<ApiKeys> {
    try {
      const response = await api.get('/api-keys');
      return response.data;
    } catch (error) {
      console.error('Get API keys error:', error);
      throw error;
    }
  },
  
  async setApiKeys(apiKeys: ApiKeys): Promise<void> {
    try {
      await api.post('/api-keys', apiKeys);
    } catch (error) {
      console.error('Set API keys error:', error);
      throw error;
    }
  },
};

export default api;