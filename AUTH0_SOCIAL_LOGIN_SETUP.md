# 🔐 Auth0 Social Login Setup Guide

This guide will help you set up Auth0 with multiple social login providers (Google, GitHub, Email/Password) for your Diabetes Research Assistant.

## 🎯 **What We're Setting Up**

- **Google/Gmail Login** - Most popular option
- **GitHub Login** - Great for developers
- **Email/Password Login** - Traditional fallback
- **Unified User Management** - Same permissions for all providers
- **Domain-based Admin Access** - `@lit-koi.pankbase.org` users get admin

## 📋 **Prerequisites**

1. **Auth0 Account** - Sign up at [auth0.com](https://auth0.com)
2. **Domain** - Your application domain: `https://lit-koi.pankbase.org`
3. **Google Developer Account** - For Google OAuth
4. **GitHub Account** - For GitHub OAuth

## 🚀 **Step 1: Auth0 Application Setup**

### **1.1 Create Auth0 Application**

1. **Login to Auth0 Dashboard**
   - Go to [manage.auth0.com](https://manage.auth0.com)
   - Create a new account or sign in

2. **Create Application**
   - Click "Applications" → "Create Application"
   - Name: `Diabetes Research Assistant`
   - Type: **Single Page Application**
   - Click "Create"

3. **Configure Application Settings**
   ```
   Allowed Callback URLs: 
   - http://localhost:3000 (for development)
   - https://lit-koi.pankbase.org (for production)
   
   Allowed Logout URLs: 
   - http://localhost:3000 (for development)
   - https://lit-koi.pankbase.org (for production)
   
   Allowed Web Origins: 
   - http://localhost:3000 (for development)
   - https://lit-koi.pankbase.org (for production)
   ```

### **1.2 Create Auth0 API**

1. **Create API**
   - Go to "APIs" → "Create API"
   - Name: `Diabetes Research API`
   - Identifier: `https://lit-koi.pankbase.org/api`
   - Signing Algorithm: **RS256**

2. **Configure API Scopes**
   ```
   read:research - Read research data
   write:research - Write research data
   admin:access - Admin access
   ```

3. **Create API Permissions**
   - Go to "APIs" → Your API → "Permissions"
   - Add permissions:
     - `read:research`
     - `write:research`
     - `admin:access`

## 🚀 **Step 2: Enable Social Connections**

### **2.1 Enable Google Login**

1. **Create Google OAuth App**
   - Go to [Google Cloud Console](https://console.developers.google.com)
   - Create a new project or select existing
   - Enable **Google+ API**
   - Go to **"Credentials"** → **"Create Credentials"** → **"OAuth 2.0 Client IDs"**
   - Application type: **Web application**
   - Authorized redirect URIs:
     ```
     https://your-domain.auth0.com/login/callback
     ```

2. **Configure Auth0 Google Connection**
   - Go to Auth0 Dashboard → **"Authentication"** → **"Social"**
   - Click **"Google"**
   - Toggle **"Enabled"** = ON
   - Add your Google credentials:
     ```
     Client ID: [Your Google Client ID]
     Client Secret: [Your Google Client Secret]
     ```
   - Click **"Save"**

### **2.2 Enable GitHub Login**

1. **Create GitHub OAuth App**
   - Go to [GitHub Developer Settings](https://github.com/settings/developers)
   - Click **"New OAuth App"**
   - Fill in details:
     ```
     Application name: Diabetes Research Assistant
     Homepage URL: https://lit-koi.pankbase.org
     Authorization callback URL: https://your-domain.auth0.com/login/callback
     ```

2. **Configure Auth0 GitHub Connection**
   - Go to Auth0 Dashboard → **"Authentication"** → **"Social"**
   - Click **"GitHub"**
   - Toggle **"Enabled"** = ON
   - Add your GitHub credentials:
     ```
     Client ID: [Your GitHub Client ID]
     Client Secret: [Your GitHub Client Secret]
     ```
   - Click **"Save"**

### **2.3 Enable Email/Password Login**

1. **Configure Database Connection**
   - Go to Auth0 Dashboard → **"Authentication"** → **"Database"**
   - Click **"Username-Password-Authentication"**
   - Toggle **"Enabled"** = ON
   - Toggle **"Disable Sign Ups"** = OFF (allow registration)
   - Click **"Save"**

## 🚀 **Step 3: Configure Auth0 Actions**

### **3.1 Create Post Login Action**

1. **Go to Auth0 Dashboard**
   - Navigate to **"Actions"** → **"Triggers"** → **"Post Login"**

2. **Create New Action**
   - Click **"Create Action"**
   - Name: `Assign User Permissions`
   - Runtime: **Node.js 18**
   - Click **"Create"**

3. **Add the Action Code**
   ```javascript
   exports.onExecutePostLogin = async (event, api) => {
     const namespace = 'https://lit-koi.pankbase.org/api';
     
     // Get user email (works for all providers)
     const email = event.user.email;
     
     // Default permissions for all users
     let permissions = ['read:research'];
     let role = 'researcher';
     
     // Admin users (works for all providers)
     if (email.endsWith('@lit-koi.pankbase.org') || email === 'admin@example.com') {
       permissions.push('admin:access');
       role = 'admin';
     }
     
     // Add permissions to tokens
     api.idToken.setCustomClaim(`${namespace}/permissions`, permissions);
     api.accessToken.setCustomClaim(`${namespace}/permissions`, permissions);
     
     // Add role to tokens
     api.idToken.setCustomClaim(`${namespace}/role`, role);
     api.accessToken.setCustomClaim(`${namespace}/role`, role);
     
     // Add user info to tokens
     api.idToken.setCustomClaim(`${namespace}/user_id`, event.user.user_id);
     api.accessToken.setCustomClaim(`${namespace}/user_id`, event.user.user_id);
     api.idToken.setCustomClaim(`${namespace}/email`, email);
     api.accessToken.setCustomClaim(`${namespace}/email`, email);
     
     // Add provider info
     api.idToken.setCustomClaim(`${namespace}/provider`, event.connection);
     api.accessToken.setCustomClaim(`${namespace}/provider`, event.connection);
   };
   ```

### **3.2 Deploy and Enable Action**

1. **Deploy Action**
   - Click **"Deploy"**
   - Wait for deployment to complete

2. **Add to Login Flow**
   - Go to **"Actions"** → **"Triggers"** → **"Post Login"**
   - Drag your action to the flow
   - Click **"Apply"**

## 🚀 **Step 4: Environment Configuration**

### **4.1 Frontend Environment Variables**

Create `frontend/.env`:
```bash
# Auth0 Configuration
REACT_APP_AUTH0_DOMAIN=your-domain.auth0.com
REACT_APP_AUTH0_CLIENT_ID=your-client-id
REACT_APP_AUTH0_AUDIENCE=https://lit-koi.pankbase.org/api
REACT_APP_AUTH0_REDIRECT_URI=https://lit-koi.pankbase.org

# API Configuration
REACT_APP_API_BASE_URL=https://lit-koi.pankbase.org/api
```

### **4.2 Backend Environment Variables**

Create `.env`:
```bash
# API Keys
OPENAI_API_KEY=your_actual_openai_key_here
PINECONE_API_KEY=your_actual_pinecone_key_here
PINECONE_INDEX_NAME=genomics-publications

# Pinecone Configuration
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
EMBEDDING_DIMENSION=1536

# Production Settings
ENVIRONMENT=production
LOG_LEVEL=info
DEBUG=false
PORT=8000

# Enhanced Features
EXTRACT_PDF_METADATA=true
ENRICH_WITH_CROSSREF=true
CROSSREF_EMAIL=your-email@domain.com

# Performance
WORKERS=4
MAX_REQUESTS=1000

# Auth0 Configuration
AUTH0_DOMAIN=your-domain.auth0.com
AUTH0_AUDIENCE=https://lit-koi.pankbase.org/api
AUTH0_ISSUER=https://your-domain.auth0.com/
```

## 🚀 **Step 5: Test the Setup**

### **5.1 Test Each Login Method**

1. **Test Google Login**
   - Go to your application
   - Click "Sign In"
   - Choose "Continue with Google"
   - Verify login works

2. **Test GitHub Login**
   - Go to your application
   - Click "Sign In"
   - Choose "Continue with GitHub"
   - Verify login works

3. **Test Email/Password Login**
   - Go to your application
   - Click "Sign In"
   - Choose "Sign up" or "Log in"
   - Create account or login
   - Verify login works

### **5.2 Verify Permissions**

```bash
# Test API with authentication
curl -X POST https://lit-koi.pankbase.org/api/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{"query": "What is diabetes?", "model": "gpt-3.5-turbo", "top_k": 2}'
```

## 🚀 **Step 6: User Management**

### **6.1 Create Test Users**

1. **Manual Creation**
   - Go to Auth0 Dashboard → **"User Management"** → **"Users"**
   - Click **"Create User"**
   - Fill in details and create

2. **Using Script (Optional)**
   ```bash
   # Set environment variables
   export AUTH0_DOMAIN=your-domain.auth0.com
   export AUTH0_MANAGEMENT_CLIENT_ID=your-management-client-id
   export AUTH0_MANAGEMENT_CLIENT_SECRET=your-management-client-secret
   
   # Run the script
   python create_auth0_users.py
   ```

### **6.2 Test Different User Types**

1. **Regular User** (any email)
   - Should get `read:research` permissions
   - Can access search and query endpoints

2. **Admin User** (`@lit-koi.pankbase.org` email)
   - Should get `read:research` + `admin:access` permissions
   - Can access admin endpoints

## 🔧 **Step 7: Troubleshooting**

### **7.1 Common Issues**

1. **"Invalid redirect_uri" Error**
   - Check Auth0 application settings
   - Verify callback URLs are correct

2. **"Connection not found" Error**
   - Ensure social connections are enabled
   - Check OAuth credentials are correct

3. **"Permission denied" Error**
   - Check Auth0 Action is deployed
   - Verify permissions are assigned correctly

### **7.2 Debug Commands**

```bash
# Check Auth0 configuration
curl https://your-domain.auth0.com/.well-known/openid_configuration

# Test JWT token
jwt.io (online tool)

# Check backend logs
tail -f /home/ubuntu/genomics-app/logs/error.log

# Test API endpoints
curl https://lit-koi.pankbase.org/api/health
curl https://lit-koi.pankbase.org/api/docs
```

## ✅ **Verification Checklist**

- [ ] Auth0 application created and configured
- [ ] Auth0 API created with proper scopes
- [ ] Google OAuth app created and configured
- [ ] GitHub OAuth app created and configured
- [ ] Email/Password authentication enabled
- [ ] Auth0 Actions created and deployed
- [ ] Environment variables set correctly
- [ ] Frontend authentication working
- [ ] Backend token validation working
- [ ] All login methods tested
- [ ] Permissions assigned correctly
- [ ] Production SSL configured

## 🎉 **Your Social Login Setup is Complete!**

**Users can now login with:**
- ✅ **Google/Gmail** account
- ✅ **GitHub** account
- ✅ **Email/Password** account

**All users get:**
- ✅ **Unified permissions** based on email domain
- ✅ **Secure authentication** via Auth0
- ✅ **Role-based access** (researcher/admin)

## 📞 **Support**

If you encounter issues:
1. Check Auth0 logs in the dashboard
2. Review Action logs in Auth0 → Actions → Your Action → Logs
3. Verify OAuth credentials are correct
4. Test each login method individually
5. Check environment variables are set correctly

Your Auth0 social login setup is now production-ready! 🔒 