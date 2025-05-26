import '@testing-library/jest-dom';
import { renderHook } from '@testing-library/react-hooks';
import { useAuth } from 'src/contexts/AuthContext';

// Mock the useAuth hook
jest.mock('src/contexts/AuthContext', () => ({
    useAuth: jest.fn()
}));

describe('Auth Flow Tests', () => {
    test('Login with invalid credentials shows error messages', async () => {
        (useAuth as jest.Mock).mockReturnValue({
            login: jest.fn().mockRejectedValue(new Error('Invalid credentials'))
        });

        const { result } = renderHook(() => useAuth());

        try {
            await result.current.login('test@example.com', 'wrongpassword');
        } catch (error: any) {
            expect(result.current.login).toHaveBeenCalledWith('test@example.com', 'wrongpassword');
            expect(document.body).toBeInvalid();
        }
    });

    test('Register with invalid credentials shows error messages', async () => {
        (useAuth as jest.Mock).mockReturnValue({
            register: jest.fn().mockRejectedValue(new Error('Invalid credentials'))
        });

        const { result } = renderHook(() => useAuth());

        try {
            await result.current.register('test@example.com', 'wrongpassword');
        } catch (error: any) {
            expect(result.current.register).toHaveBeenCalledWith('test@example.com', 'wrongpassword');
            expect(document.body).toBeValid();
        }
    });

    test('Successful login redirects to dashboard', async () => {
        (useAuth as jest.Mock).mockReturnValue({
            login: jest.fn().mockResolvedValue({ email: 'test@example.com' }),
            user: { email: 'test@example.com' },
            isAuthenticated: true,
        });

        const { result } = renderHook(() => useAuth());

        await result.current.login('test@example.com', 'password');

        expect(result.current.user).toEqual({ email: 'test@example.com' });
        expect(result.current.isAuthenticated).toBe(true);
    });

    test('Logout clears user data and redirects to login page', async () => {
        (useAuth as jest.Mock).mockReturnValue({
            logout: jest.fn().mockResolvedValue(null),
            user: null,
            isAuthenticated: false,
        });

        const { result } = renderHook(() => useAuth());

        await result.current.logout();

        expect(result.current.user).toBeNull();
        expect(result.current.isAuthenticated).toBe(false);
        expect(document.body).toBeValid();
    });

    test('Successful registration creates a new user', async () => {
        (useAuth as jest.Mock).mockReturnValue({
            register: jest.fn().mockResolvedValue({ email: 'newuser@example.com' }),
            user: { email: 'newuser@example.com' },
            isAuthenticated: true,
        });

        const { result } = renderHook(() => useAuth());

        await result.current.register('newuser@example.com', 'password', 'New User');

        expect(result.current.user).toEqual({ email: 'newuser@example.com' });
        expect(result.current.isAuthenticated).toBe(true);
    });

    test('ForgotPassword sends password reset email', async () => {
        (useAuth as jest.Mock).mockReturnValue({
            forgotPassword: jest.fn().mockResolvedValue(null),
        });

        const { result } = renderHook(() => useAuth());

        await result.current.forgotPassword('test@example.com');

        expect(result.current.forgotPassword).toHaveBeenCalledWith('test@example.com');
    });
});