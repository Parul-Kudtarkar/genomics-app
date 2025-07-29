#!/usr/bin/env python3
"""
Auth0 User Creation Script
Creates test users for the Diabetes Research Assistant
"""

import os
import requests
import json
from typing import Dict, List

# Auth0 Configuration
AUTH0_DOMAIN = os.getenv('AUTH0_DOMAIN', 'your-domain.auth0.com')
MANAGEMENT_CLIENT_ID = os.getenv('AUTH0_MANAGEMENT_CLIENT_ID')
MANAGEMENT_CLIENT_SECRET = os.getenv('AUTH0_MANAGEMENT_CLIENT_SECRET')

def get_management_token() -> str:
    """Get Auth0 Management API token"""
    url = f"https://{AUTH0_DOMAIN}/oauth/token"
    payload = {
        "client_id": MANAGEMENT_CLIENT_ID,
        "client_secret": MANAGEMENT_CLIENT_SECRET,
        "audience": f"https://{AUTH0_DOMAIN}/api/v2/",
        "grant_type": "client_credentials"
    }
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()['access_token']
    else:
        raise Exception(f"Failed to get management token: {response.text}")

def create_user(email: str, password: str, name: str, role: str = 'researcher') -> Dict:
    """Create a new Auth0 user"""
    token = get_management_token()
    
    url = f"https://{AUTH0_DOMAIN}/api/v2/users"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "email": email,
        "password": password,
        "connection": "Username-Password-Authentication",
        "name": name,
        "user_metadata": {
            "role": role
        },
        "app_metadata": {
            "role": role
        }
    }
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 201:
        return response.json()
    else:
        raise Exception(f"Failed to create user: {response.text}")

def create_test_users():
    """Create test users for the application"""
    test_users = [
        {
            "email": "researcher@example.com",
            "password": "SecurePassword123!",
            "name": "Test Researcher",
            "role": "researcher"
        },
        {
            "email": "admin@example.com", 
            "password": "AdminPassword123!",
            "name": "Test Admin",
            "role": "admin"
        },
        {
            "email": "user@lit-koi.pankbase.org",
            "password": "DomainPassword123!",
            "name": "Domain User",
            "role": "admin"  # Domain users get admin access
        }
    ]
    
    print("🔐 Creating Auth0 test users...")
    
    for user in test_users:
        try:
            result = create_user(
                email=user["email"],
                password=user["password"], 
                name=user["name"],
                role=user["role"]
            )
            print(f"✅ Created user: {user['email']} ({user['role']})")
        except Exception as e:
            print(f"❌ Failed to create {user['email']}: {e}")
    
    print("\n🎉 User creation complete!")
    print("\n📋 Test Users:")
    for user in test_users:
        print(f"   Email: {user['email']}")
        print(f"   Password: {user['password']}")
        print(f"   Role: {user['role']}")
        print()

if __name__ == "__main__":
    # Check if environment variables are set
    if not all([AUTH0_DOMAIN, MANAGEMENT_CLIENT_ID, MANAGEMENT_CLIENT_SECRET]):
        print("❌ Missing Auth0 environment variables!")
        print("Please set:")
        print("  AUTH0_DOMAIN=your-domain.auth0.com")
        print("  AUTH0_MANAGEMENT_CLIENT_ID=your-management-client-id")
        print("  AUTH0_MANAGEMENT_CLIENT_SECRET=your-management-client-secret")
        exit(1)
    
    create_test_users() 