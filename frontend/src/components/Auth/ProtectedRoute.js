import React from 'react';
import { useAuth0 } from '@auth0/auth0-react';
import styled from 'styled-components';

const LoadingContainer = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
  background: #fff;
`;

const LoadingSpinner = styled.div`
  width: 40px;
  height: 40px;
  border: 4px solid #f3f3f3;
  border-top: 4px solid #007AFF;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

const AuthRequired = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  background: #fff;
  text-align: center;
  padding: 2rem;
`;

const AuthTitle = styled.h1`
  font-size: 2rem;
  font-weight: 700;
  color: #1d1d1f;
  margin-bottom: 1rem;
`;

const AuthMessage = styled.p`
  font-size: 1.1rem;
  color: #6e6e73;
  margin-bottom: 2rem;
  max-width: 500px;
`;

const AuthButton = styled.button`
  background: linear-gradient(135deg, #007AFF 0%, #5856D6 100%);
  color: white;
  border: none;
  border-radius: 12px;
  padding: 1rem 2rem;
  font-size: 1.1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
  box-shadow: 0 4px 12px rgba(0, 122, 255, 0.15);
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 122, 255, 0.25);
  }
`;

export default function ProtectedRoute({ children, requiredPermissions = [] }) {
  const { 
    isAuthenticated, 
    isLoading, 
    loginWithRedirect, 
    user,
    getAccessTokenSilently 
  } = useAuth0();

  // Check if user has required permissions
  const hasRequiredPermissions = () => {
    if (requiredPermissions.length === 0) return true;
    
    const userPermissions = user?.['https://your-api-identifier/permissions'] || [];
    return requiredPermissions.every(permission => 
      userPermissions.includes(permission)
    );
  };

  if (isLoading) {
    return (
      <LoadingContainer>
        <LoadingSpinner />
      </LoadingContainer>
    );
  }

  if (!isAuthenticated) {
    return (
      <AuthRequired>
        <AuthTitle>Authentication Required</AuthTitle>
        <AuthMessage>
          Please sign in to access the Diabetes Research Assistant. 
          This application requires authentication to ensure secure access to research data.
        </AuthMessage>
        <AuthButton onClick={() => loginWithRedirect()}>
          Sign In to Continue
        </AuthButton>
      </AuthRequired>
    );
  }

  if (!hasRequiredPermissions()) {
    return (
      <AuthRequired>
        <AuthTitle>Access Denied</AuthTitle>
        <AuthMessage>
          You don't have the required permissions to access this feature. 
          Please contact your administrator for access.
        </AuthMessage>
      </AuthRequired>
    );
  }

  return children;
} 