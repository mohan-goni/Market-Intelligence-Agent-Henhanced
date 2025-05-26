import React, { createContext, useContext, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'react-hot-toast';

// Define User interface
interface User {
  email: string;
  full_name?: string;
}

// Define AuthContext interface
interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, fullName?: string) => Promise<User>;
  logout: () => void;
  forgotPassword: (email: string) => Promise<void>;
}

// Create context with default values
const defaultContext: AuthContextType = {
  user: null,
  isAuthenticated: false,
  isLoading: false,
  login: async () => {},
  register: async () => ({ email: '' }),
  logout: () => {},
  forgotPassword: async () => {}
};

const AuthContext = createContext<AuthContextType>(defaultContext);

export const useAuth = () => {
  return useContext(AuthContext);
};

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  // DEMO MODE: Always set a demo user and authenticated state for testing
  const [user] = useState<User | null>({
    email: "demo@example.com",
    full_name: "Demo User"
  });
  const [isLoading] = useState<boolean>(false);
  const navigate = useNavigate();

  const login = async (email: string, password: string) => {
    toast.success('Demo mode: Login successful!');
    navigate('/dashboard');
  };

  const register = async (email: string, password: string, fullName?: string) => {
    toast.success('Demo mode: Registration successful!');
    const demoUser = { email, full_name: fullName };
    return demoUser;
  };

  const logout = () => {
    toast.success('Demo mode: Logout simulated');
    navigate('/login');
  };

  const forgotPassword = async (email: string) => {
    toast.success('Demo mode: Password reset simulation successful');
  };

  const value = {
    user,
    isAuthenticated: true, // Always authenticated for demo
    isLoading,
    login,
    register,
    logout,
    forgotPassword
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};
