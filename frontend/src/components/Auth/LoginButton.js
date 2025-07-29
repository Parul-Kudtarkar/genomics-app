import React from 'react';
import styled from 'styled-components';
import { useAuth0 } from '@auth0/auth0-react';

const LoginButtonStyled = styled.button`
  background: linear-gradient(135deg, #007AFF 0%, #5856D6 25%, #AF52DE 50%, #FF2D92 75%, #FF9500 100%);
  color: white;
  border: none;
  border-radius: 12px;
  padding: 0.8rem 1.5rem;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
  box-shadow: 0 4px 12px rgba(0, 122, 255, 0.15);
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 122, 255, 0.25);
  }
  
  &:active {
    transform: translateY(0);
  }
`;

const LogoutButtonStyled = styled.button`
  background: #f5f5f7;
  color: #1d1d1f;
  border: 1px solid #e5e5e7;
  border-radius: 12px;
  padding: 0.8rem 1.5rem;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
  
  &:hover {
    background: #e5e5e7;
    border-color: #d1d1d6;
  }
`;

const UserInfo = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 0.5rem 1rem;
  background: #f5f5f7;
  border-radius: 12px;
  font-size: 0.9rem;
`;

const UserAvatar = styled.img`
  width: 32px;
  height: 32px;
  border-radius: 50%;
  border: 2px solid #007AFF;
`;

const UserDetails = styled.div`
  display: flex;
  flex-direction: column;
`;

const UserName = styled.span`
  font-weight: 600;
  color: #1d1d1f;
`;

const UserEmail = styled.span`
  font-size: 0.8rem;
  color: #6e6e73;
`;

export default function LoginButton() {
  const { 
    isAuthenticated, 
    loginWithRedirect, 
    logout, 
    user, 
    isLoading 
  } = useAuth0();

  if (isLoading) {
    return (
      <LoginButtonStyled disabled>
        Loading...
      </LoginButtonStyled>
    );
  }

  if (isAuthenticated) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        <UserInfo>
          {user?.picture && (
            <UserAvatar src={user.picture} alt={user.name} />
          )}
          <UserDetails>
            <UserName>{user?.name || 'User'}</UserName>
            <UserEmail>{user?.email}</UserEmail>
          </UserDetails>
        </UserInfo>
        <LogoutButtonStyled
          onClick={() => logout({ 
            logoutParams: { 
              returnTo: window.location.origin 
            } 
          })}
        >
          Logout
        </LogoutButtonStyled>
      </div>
    );
  }

  return (
    <LoginButtonStyled onClick={() => loginWithRedirect()}>
      Sign In
    </LoginButtonStyled>
  );
} 