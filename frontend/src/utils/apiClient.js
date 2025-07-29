import { useAuth0 } from '@auth0/auth0-react';
import { apiConfig } from '../auth/auth0-config';

// Custom hook for authenticated API calls
export const useApiClient = () => {
  const { getAccessTokenSilently, isAuthenticated } = useAuth0();

  const apiCall = async (endpoint, options = {}) => {
    try {
      let headers = {
        'Content-Type': 'application/json',
        ...options.headers,
      };

      // TEMPORARILY DISABLED AUTH - FOR TESTING ONLY
      // TODO: Re-enable authentication after testing
      /*
      // Add Auth0 token if authenticated
      if (isAuthenticated) {
        try {
          const token = await getAccessTokenSilently({
            audience: apiConfig.audience,
            scope: 'read:research write:research',
          });
          headers.Authorization = `Bearer ${token}`;
        } catch (tokenError) {
          console.error('Failed to get access token:', tokenError);
          // Continue without token for public endpoints
        }
      }
      */

      const response = await fetch(`${apiConfig.baseURL}${endpoint}`, {
        ...options,
        headers,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('API call failed:', error);
      throw error;
    }
  };

  return {
    get: (endpoint) => apiCall(endpoint, { method: 'GET' }),
    post: (endpoint, data) => apiCall(endpoint, {
      method: 'POST',
      body: JSON.stringify(data),
    }),
    put: (endpoint, data) => apiCall(endpoint, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),
    delete: (endpoint) => apiCall(endpoint, { method: 'DELETE' }),
  };
};

// Utility function for non-hook API calls
export const makeApiCall = async (endpoint, options = {}) => {
  const response = await fetch(`${apiConfig.baseURL}${endpoint}`, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
  }

  return await response.json();
}; 