import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { useAuth } from '../../contexts/AuthContext';

const PasswordResetService: React.FC = () => {
  const { forgotPassword } = useAuth();
  const [isSubmitted, setIsSubmitted] = useState(false);
  const { register, handleSubmit, formState: { errors } } = useForm();

  const onSubmit = async (data: any) => {
    try {
      await forgotPassword(data.email);
      setIsSubmitted(true);
    } catch (error) {
      console.error('Password reset error:', error);
      // Don't reveal if email exists or not for security
      setIsSubmitted(true);
    }
  };

  return (
    <div className="max-w-md mx-auto">
      {!isSubmitted ? (
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          <div>
            <label htmlFor="email" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Email Address
            </label>
            <input
              id="email"
              type="email"
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:text-white"
              placeholder="Enter your email address"
              {...register('email', { 
                required: 'Email is required',
                pattern: {
                  value: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
                  message: 'Invalid email address'
                }
              })}
            />
            {errors.email && (
              <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.email.message?.toString()}</p>
            )}
          </div>
          
          <div>
            <button
              type="submit"
              className="w-full px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 transition"
            >
              Send Reset Instructions
            </button>
          </div>
          
          <p className="text-sm text-gray-500 dark:text-gray-400 text-center">
            Password reset instructions will be sent to marketintelligenceagent@gmail.com for verification.
          </p>
        </form>
      ) : (
        <div className="text-center">
          <div className="mb-4 flex justify-center">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">Reset Instructions Sent</h3>
          <p className="text-gray-600 dark:text-gray-300 mb-4">
            If your email is registered in our system, you will receive password reset instructions shortly.
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Please check your inbox and follow the instructions to reset your password.
          </p>
        </div>
      )}
    </div>
  );
};

export default PasswordResetService;
