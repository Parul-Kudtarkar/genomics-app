// Auth0 Configuration
export const auth0Config = {
  domain: process.env.REACT_APP_AUTH0_DOMAIN || 'your-domain.auth0.com',
  clientId: process.env.REACT_APP_AUTH0_CLIENT_ID || 'your-client-id',
  audience: process.env.REACT_APP_AUTH0_AUDIENCE || 'https://your-api-identifier',
  redirectUri: process.env.REACT_APP_AUTH0_REDIRECT_URI || window.location.origin,
  scope: 'openid profile email read:research write:research',
  cacheLocation: 'localstorage',
  useRefreshTokens: true,
};

// API Configuration
export const apiConfig = {
  baseURL: process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000',
  audience: process.env.REACT_APP_AUTH0_AUDIENCE || 'https://your-api-identifier',
};

// User roles and permissions
export const userRoles = {
  RESEARCHER: 'researcher',
  ADMIN: 'admin',
  GUEST: 'guest',
};

export const permissions = {
  READ_RESEARCH: 'read:research',
  WRITE_RESEARCH: 'write:research',
  ADMIN_ACCESS: 'admin:access',
}; 