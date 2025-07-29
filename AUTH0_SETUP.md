# 🔐 Auth0 Authentication Setup Guide

This guide will help you set up Auth0 authentication for your Diabetes Research Assistant with secure user management and role-based access control using modern Auth0 Actions.

## 🎯 **What We're Building**

- **Secure Authentication** with Auth0
- **Role-based Access Control** (Researcher, Admin, Guest)
- **JWT Token Validation** on the backend
- **User Profile Management** with permissions
- **Rate Limiting** per user
- **Protected API Endpoints**

## 📋 **Prerequisites**

1. **Auth0 Account** - Sign up at [auth0.com](https://auth0.com)
2. **Domain** - Your application domain: `https://lit-koi.pankbase.org`
3. **Environment Variables** - Configured for both frontend and backend

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

## 🚀 **Step 2: Auth0 Actions (Modern Approach)**

### **2.1 Create Auth0 Action for User Permissions**

1. **Go to Auth0 Dashboard**
   - Navigate to **"Actions"** → **"Triggers"** → **"Login"**

2. **Create New Action**
   - Click **"Create Action"**
   - Name: `Assign User Permissions`
   - Runtime: **Node.js 18** (recommended)
   - Click **"Create"**

3. **Add the Action Code**
   ```javascript
   exports.onExecutePostLogin = async (event, api) => {
     const namespace = 'https://lit-koi.pankbase.org/api';
     
     // Get user email
     const email = event.user.email;
     
     // Default permissions for all users
     let permissions = ['read:research'];
     let role = 'researcher';
     
     // Admin users (customize this logic for your organization)
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
   };
   ```

### **2.2 Create Auth0 Action for User Metadata (Optional)**

1. **Create Another Action**
   - Go to **"Actions"** → **"Triggers"** → **"Login"**
   - Click **"Create Action"**
   - Name: `Set User Metadata`
   - Runtime: **Node.js 18**

2. **Add the Action Code**
   ```javascript
   exports.onExecutePostLogin = async (event, api) => {
     const namespace = 'https://lit-koi.pankbase.org/api';
     
     // Set user profile information
     const userProfile = {
       user_id: event.user.user_id,
       email: event.user.email,
       name: event.user.name || event.user.email,
       picture: event.user.picture,
       created_at: event.user.created_at,
       last_login: new Date().toISOString()
     };
     
     // Add user profile to tokens
     api.idToken.setCustomClaim(`${namespace}/profile`, userProfile);
     api.accessToken.setCustomClaim(`${namespace}/profile`, userProfile);
   };
   ```

### **2.3 Deploy and Enable Actions**

1. **Deploy Actions**
   - Click **"Deploy"** for each action
   - Wait for deployment to complete

2. **Add Actions to Login Trigger**
   - Go to **"Actions"** → **"Triggers"** → **"Login"**
   - You should see your deployed actions available
   - Click **"Add Action"** and select your actions
   - Order: "Assign User Permissions" first, then "Set User Metadata"
   - Click **"Save"**

## 🚀 **Step 3: Environment Configuration**

### **3.1 Frontend Environment Variables**

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

### **3.2 Backend Environment Variables**

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

## 🚀 **Step 4: Install Dependencies**

### **4.1 Frontend Dependencies**
```bash
cd /home/ubuntu/genomics-app/frontend
npm install @auth0/auth0-react react-router-dom
```

### **4.2 Backend Dependencies**
```bash
cd /home/ubuntu/genomics-app
source /home/ubuntu/venv/bin/activate
pip install python-jose[cryptography] authlib
```

## 🚀 **Step 5: Test the Setup**

### **5.1 Start Backend**
```bash
cd /home/ubuntu/genomics-app
./start_api.sh
```

### **5.2 Start Frontend (Development)**
```bash
cd /home/ubuntu/genomics-app/frontend
npm start
```

### **5.3 Test Authentication**
1. Open http://localhost:3000 (development) or https://lit-koi.pankbase.org (production)
2. Click "Sign In"
3. Complete Auth0 login
4. Verify you can access the research interface

## 🔧 **Step 6: User Management**

### **6.1 Create Test Users**

1. **Go to Auth0 Dashboard**
   - Navigate to "User Management" → "Users"

2. **Create Test Users**
   ```
   Email: researcher@example.com
   Password: SecurePassword123!
   Role: Researcher
   
   Email: admin@example.com
   Password: AdminPassword123!
   Role: Admin
   ```

### **6.2 Test Permission Assignment**

1. **Login with Different Users**
   - Login with `researcher@example.com` → Should get `read:research`
   - Login with `admin@example.com` → Should get `read:research` + `admin:access`
   - Login with `user@lit-koi.pankbase.org` → Should get `read:research` + `admin:access`

2. **Verify Permissions**
   ```bash
   # Check user profile endpoint
   curl -H "Authorization: Bearer YOUR_TOKEN" \
        https://lit-koi.pankbase.org/api/user/permissions
   ```

## 🔒 **Step 7: Security Features**

### **7.1 Rate Limiting**
- **Per User**: 100 requests per hour
- **Per Endpoint**: Configurable limits
- **Anonymous**: Limited access

### **7.2 Permission-Based Access**
- **Public Endpoints**: `/health`, `/models`, `/filters/options`
- **Authenticated Endpoints**: `/query`, `/search`
- **Admin Endpoints**: `/admin/*`

### **7.3 JWT Token Validation**
- **RS256 Algorithm**: Secure token signing
- **Audience Validation**: Ensures tokens are for your API
- **Expiration Handling**: Automatic token refresh

## 🧪 **Step 8: Testing Security**

### **8.1 Test Unauthenticated Access**
```bash
# Should fail
curl -X POST https://lit-koi.pankbase.org/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'
```

### **8.2 Test Authenticated Access**
```bash
# Get token from Auth0
TOKEN="your-jwt-token"

# Should succeed
curl -X POST https://lit-koi.pankbase.org/api/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"query": "test"}'
```

### **8.3 Test Admin Access**
```bash
# Should succeed for admin users
curl -X GET https://lit-koi.pankbase.org/api/admin/stats \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

## 🔧 **Step 9: Production Deployment**

### **9.1 Update Auth0 Settings for Production**
1. **Update Callback URLs**
   ```
   Allowed Callback URLs: 
   - https://lit-koi.pankbase.org
   - http://localhost:3000 (for development)
   
   Allowed Logout URLs: 
   - https://lit-koi.pankbase.org
   - http://localhost:3000 (for development)
   
   Allowed Web Origins: 
   - https://lit-koi.pankbase.org
   - http://localhost:3000 (for development)
   ```

2. **Update Environment Variables**
   ```bash
   # Frontend
   REACT_APP_AUTH0_REDIRECT_URI=https://lit-koi.pankbase.org
   REACT_APP_API_BASE_URL=https://lit-koi.pankbase.org/api
   
   # Backend
   AUTH0_AUDIENCE=https://lit-koi.pankbase.org/api
   ```

### **9.2 SSL Configuration**
- **HTTPS Required**: All production traffic
- **Secure Cookies**: Auth0 session management
- **CORS Configuration**: Restrict to your domain

## 🚨 **Security Best Practices**

### **9.1 Token Security**
- **Short Expiration**: 1 hour for access tokens
- **Refresh Tokens**: For seamless user experience
- **Token Storage**: Secure localStorage with encryption

### **9.2 API Security**
- **CORS**: Restrict to trusted domains
- **Rate Limiting**: Prevent abuse
- **Input Validation**: Sanitize all inputs

### **9.3 User Data Protection**
- **GDPR Compliance**: User data handling
- **Data Encryption**: At rest and in transit
- **Access Logging**: Audit trail for security

## 🔍 **Troubleshooting**

### **Common Issues**

1. **CORS Errors**
   - Check Auth0 callback URLs include both localhost and production domain
   - Verify CORS configuration in backend

2. **Token Validation Failures**
   - Check Auth0 domain and audience (`https://lit-koi.pankbase.org/api`)
   - Verify JWT signing algorithm

3. **Permission Denied**
   - Check user roles and permissions
   - Verify Auth0 Actions are deployed and enabled

4. **Actions Not Working**
   - Check Action deployment status
   - Verify Action is added to Login flow
   - Check Action logs in Auth0 dashboard

### **Debug Commands**
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

# Check Action logs in Auth0 dashboard
# Go to Auth0 → Actions → Your Action → Logs
```

## ✅ **Verification Checklist**

- [ ] Auth0 application created and configured
- [ ] Auth0 API created with proper scopes (`https://lit-koi.pankbase.org/api`)
- [ ] Auth0 Actions created and deployed (not deprecated Rules)
- [ ] Actions added to Login flow
- [ ] Environment variables set correctly for production domain
- [ ] Frontend authentication working on https://lit-koi.pankbase.org
- [ ] Backend token validation working
- [ ] User roles and permissions assigned via Actions
- [ ] Rate limiting configured
- [ ] Security headers implemented
- [ ] Production SSL configured
- [ ] Error handling implemented
- [ ] Both localhost and production URLs configured in Auth0

## 🚀 **Quick Production Test**

```bash
# Test the complete flow
echo "Testing production deployment..."

# 1. Test frontend loads
curl -I https://lit-koi.pankbase.org/

# 2. Test API health
curl https://lit-koi.pankbase.org/api/health

# 3. Test API docs
curl -I https://lit-koi.pankbase.org/api/docs

# 4. Test authentication endpoint (should redirect to Auth0)
curl -I https://lit-koi.pankbase.org/api/user/profile

echo "✅ Production deployment test complete!"
```

## 🔄 **Migration from Rules to Actions**

If you have existing Rules, here's how to migrate:

1. **Export Rule Logic**
   - Copy your existing Rule code
   - Convert to Action format (see examples above)

2. **Create New Action**
   - Use the Action code provided above
   - Deploy and test

3. **Disable Old Rule**
   - Go to Auth0 → Rules
   - Toggle off the old Rule

4. **Verify Functionality**
   - Test login flow
   - Check permissions are assigned correctly

Your Auth0 authentication is now fully configured with modern Actions and secure for production! 🎉

## 📞 **Support**

If you encounter issues:
1. Check Auth0 logs in the dashboard
2. Review Action logs in Auth0 → Actions → Your Action → Logs
3. Review backend error logs: `tail -f /home/ubuntu/genomics-app/logs/error.log`
4. Verify environment variables
5. Test with Auth0's test tokens
6. Check Nginx logs: `sudo tail -f /var/log/nginx/error.log`

The system is now production-ready with enterprise-grade security using modern Auth0 Actions! 🔒 