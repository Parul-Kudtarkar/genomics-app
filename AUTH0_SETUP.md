# 🔐 Auth0 Authentication Setup Guide

This guide will help you set up Auth0 authentication for your Diabetes Research Assistant with secure user management and role-based access control.

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

## 🚀 **Step 2: Auth0 Rules & Actions**

### **2.1 Create Auth0 Rule for User Roles**

1. **Go to Auth0 Dashboard**
   - Navigate to "Auth Pipeline" → "Rules"

2. **Create New Rule**
   ```javascript
   function (user, context, callback) {
     // Assign default role based on email domain
     const email = user.email;
     
     // Default to researcher role
     let permissions = ['read:research'];
     
     // Admin users (customize this logic for your organization)
     if (email.endsWith('@lit-koi.pankbase.org') || email === 'admin@example.com') {
       permissions.push('admin:access');
     }
     
     // Add permissions to the token
     const namespace = 'https://lit-koi.pankbase.org/api';
     context.idToken[namespace + '/permissions'] = permissions;
     context.accessToken[namespace + '/permissions'] = permissions;
     
     callback(null, user, context);
   }
   ```

### **2.2 Create Auth0 Action for User Metadata**

1. **Go to Auth0 Dashboard**
   - Navigate to "Auth Pipeline" → "Actions" → "Flows" → "Login"

2. **Create New Action**
   ```javascript
   exports.onExecutePostLogin = async (event, api) => {
     const namespace = 'https://lit-koi.pankbase.org/api';
     
     // Set default permissions
     api.idToken.setCustomClaim(`${namespace}/permissions`, ['read:research']);
     api.accessToken.setCustomClaim(`${namespace}/permissions`, ['read:research']);
     
     // Set user role
     api.idToken.setCustomClaim(`${namespace}/role`, 'researcher');
     api.accessToken.setCustomClaim(`${namespace}/role`, 'researcher');
   };
   ```

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

### **6.2 Assign User Roles**

1. **Create Auth0 Rule for Role Assignment**
   ```javascript
   function (user, context, callback) {
     const namespace = 'https://lit-koi.pankbase.org/api';
     
     // Assign roles based on email
     let permissions = ['read:research'];
     let role = 'researcher';
     
     if (user.email === 'admin@example.com' || user.email.endsWith('@lit-koi.pankbase.org')) {
       permissions.push('admin:access');
       role = 'admin';
     }
     
     context.idToken[namespace + '/permissions'] = permissions;
     context.accessToken[namespace + '/permissions'] = permissions;
     context.idToken[namespace + '/role'] = role;
     context.accessToken[namespace + '/role'] = role;
     
     callback(null, user, context);
   }
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
   - Verify Auth0 rules are active

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
```

## ✅ **Verification Checklist**

- [ ] Auth0 application created and configured
- [ ] Auth0 API created with proper scopes (`https://lit-koi.pankbase.org/api`)
- [ ] Environment variables set correctly for production domain
- [ ] Frontend authentication working on https://lit-koi.pankbase.org
- [ ] Backend token validation working
- [ ] User roles and permissions assigned
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

Your Auth0 authentication is now fully configured and secure for production! 🎉

## 📞 **Support**

If you encounter issues:
1. Check Auth0 logs in the dashboard
2. Review backend error logs: `tail -f /home/ubuntu/genomics-app/logs/error.log`
3. Verify environment variables
4. Test with Auth0's test tokens
5. Check Nginx logs: `sudo tail -f /var/log/nginx/error.log`

The system is now production-ready with enterprise-grade security! 🔒 