#!/bin/bash
echo "🩺 Testing Diabetes Research Assistant Production Deployment"
echo "=========================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

TESTS_PASSED=0
TESTS_FAILED=0

test_endpoint() {
    local name="$1"
    local url="$2"
    local expected_status="$3"
    
    echo -n "Testing $name... "
    status=$(curl -s -o /dev/null -w "%{http_code}" "$url")
    
    if [ "$status" = "$expected_status" ]; then
        echo -e "${GREEN}✅ PASS${NC} (Status: $status)"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}❌ FAIL${NC} (Status: $status, Expected: $expected_status)"
        ((TESTS_FAILED++))
    fi
}

echo -e "\n${YELLOW}🌐 Testing Frontend${NC}"
test_endpoint "React App" "https://lit-koi.pankbase.org/" "200"

echo -e "\n${YELLOW}🔌 Testing API${NC}"
test_endpoint "Health Check" "https://lit-koi.pankbase.org/api/health" "200"
test_endpoint "API Docs" "https://lit-koi.pankbase.org/api/docs" "200"

echo -e "\n${YELLOW}🤖 Testing AI Functionality${NC}"
echo -n "Testing Query Endpoint... "
response=$(curl -s -X POST "https://lit-koi.pankbase.org/api/query" \
    -H "Content-Type: application/json" \
    -d '{"query": "What is diabetes?", "model": "gpt-3.5-turbo", "top_k": 2}')

if echo "$response" | grep -q "query\|llm_response"; then
    echo -e "${GREEN}✅ PASS${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}❌ FAIL${NC}"
    ((TESTS_FAILED++))
fi

echo -e "\n${YELLOW}📊 Services Status${NC}"
if pgrep -f "gunicorn.*main:app" > /dev/null; then
    echo -e "${GREEN}✅ FastAPI Running${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}❌ FastAPI Not Running${NC}"
    ((TESTS_FAILED++))
fi

if systemctl is-active --quiet nginx; then
    echo -e "${GREEN}✅ Nginx Running${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}❌ Nginx Not Running${NC}"
    ((TESTS_FAILED++))
fi

echo -e "\n${YELLOW}📋 Test Summary${NC}"
echo "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
echo "Tests Failed: ${RED}$TESTS_FAILED${NC}"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "\n${GREEN}🎉 All tests passed! Production deployment ready!${NC}"
    
    # Get public IP
    PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "YOUR_SERVER_IP")
    echo -e "\n${YELLOW}🌐 Access your application:${NC}"
    echo "   Frontend: https://lit-koi.pankbase.org/"
    echo "   API Docs: https://lit-koi.pankbase.org/api/docs"
    echo "   Health:   https://lit-koi.pankbase.org/api/health"
else
    echo -e "\n${RED}❌ Some tests failed. Check logs and configuration.${NC}"
fi 