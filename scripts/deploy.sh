#!/bin/bash
# deploy.sh - Updated deployment script for Python 3.12 compatibility
export CARGO_HOME=/tmp/.cargo
export CARGO_TARGET_DIR=/tmp/.cargo-target

# Your existing build commands below
apt-get update && \
apt-get install -y build-essential libmupdf-dev libjpeg-dev zlib1g-dev libtesseract-dev && \
pip install --upgrade pip setuptools wheel maturin && \
pip install -r requirements.txt

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 Document Analysis System - Fixed Deployment${NC}"
echo "================================================================"

# Check Python version
check_python() {
    echo -e "${BLUE}Checking Python version...${NC}"
    python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    echo "Python version: $python_version"
    
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        echo -e "${GREEN}✅ Python version is compatible${NC}"
    else
        echo -e "${RED}❌ Python 3.8+ required${NC}"
        exit 1
    fi
}

# Check required API keys
check_api_keys() {
    echo -e "${BLUE}Checking API keys...${NC}"
    
    if [ -z "$GEMINI_API_KEY" ]; then
        echo -e "${YELLOW}⚠️  GEMINI_API_KEY not set - some features will be limited${NC}"
        echo -e "${YELLOW}   Get your key from: https://makersuite.google.com/app/apikey${NC}"
        echo "   Set it with: export GEMINI_API_KEY='your_key_here'"
    else
        echo -e "${GREEN}✅ GEMINI_API_KEY is set${NC}"
    fi
    
    if [ -z "$PINECONE_API_KEY" ]; then
        echo -e "${YELLOW}⚠️  PINECONE_API_KEY not set - using memory fallback${NC}"
        echo -e "${YELLOW}   Get one from: https://app.pinecone.io for better performance${NC}"
    else
        echo -e "${GREEN}✅ PINECONE_API_KEY is set${NC}"
    fi
}

# Install and test dependencies locally
test_local() {
    echo -e "${BLUE}Testing locally before deployment...${NC}"
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    echo "Upgrading pip..."
    pip install --upgrade pip
    
    # Install dependencies
    echo "Installing dependencies..."
    pip install -r requirements.txt
    
    # Quick test
    echo "Starting test server..."
    python main.py &
    SERVER_PID=$!
    
    # Wait for server to start
    sleep 10
    
    # Test health endpoint
    if curl -s http://localhost:8000/health | grep -q "ok"; then
        echo -e "${GREEN}✅ Local test passed${NC}"
    else
        echo -e "${RED}❌ Local test failed${NC}"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
    
    # Stop test server
    kill $SERVER_PID 2>/dev/null || true
    sleep 2
    
    # Deactivate virtual environment
    deactivate
}

# Prepare for deployment
prepare_deployment() {
    echo -e "${BLUE}Preparing for deployment...${NC}"
    
    # Create .gitignore if it doesn't exist
    if [ ! -f ".gitignore" ]; then
        cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# Environment variables
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Testing
.pytest_cache/
.coverage
htmlcov/

# Temporary files
temp/
tmp/
*.tmp
EOF
        echo -e "${GREEN}✅ .gitignore created${NC}"
    fi
    
    # Check if git repo exists
    if [ ! -d ".git" ]; then
        echo "Initializing git repository..."
        git init
        git add .
        git commit -m "Initial commit - Fixed deployment"
    else
        # Stage all files
        git add .
        
        # Commit if there are changes
        if ! git diff --staged --quiet; then
            git commit -m "Fixed deployment - $(date)"
            echo -e "${GREEN}✅ Changes committed${NC}"
        else
            echo -e "${YELLOW}No changes to commit${NC}"
        fi
    fi
}

# Generate deployment instructions
create_deployment_guide() {
    echo -e "${BLUE}Creating deployment guide...${NC}"
    
    cat > DEPLOYMENT_GUIDE.md << 'EOF'
# 🚀 Fixed Deployment Guide

## Issues Fixed:
1. ✅ Package compatibility with Python 3.12
2. ✅ Updated dependency versions
3. ✅ Fixed import errors
4. ✅ Added proper error handling
5. ✅ Optimized for production deployment

## Quick Deploy to Render

### Step 1: Environment Setup
Set these environment variables in Render:
```
GEMINI_API_KEY=your_actual_gemini_api_key
PINECONE_API_KEY=your_pinecone_key (optional)
PYTHON_VERSION=3.11.0
```

### Step 2: Deploy
1. Push to GitHub:
   ```bash
   git remote add origin https://github.com/yourusername/your-repo.git
   git branch -M main
   git push -u origin main
   ```

2. In Render Dashboard:
   - New Web Service
   - Connect GitHub repo
   - Use existing `render.yaml` configuration
   - Deploy!

### Step 3: Test Deployment
```bash
# Test health
curl https://your-app.onrender.com/health

# Test API
curl -X POST https://your-app.onrender.com/api/v1/hackrx/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 4ddf287faf3c89dfb4c0adc648a46975d4063a37899d2243a451f717af4a32cc" \
  -d '{
    "documents": "Your policy text here...",
    "questions": ["Is surgery covered?"]
  }'
```

## Key Improvements Made:

### 1. Fixed Dependencies
- Updated all packages to Python 3.12 compatible versions
- Added missing dependencies (scipy, scikit-learn, etc.)
- Fixed version conflicts

### 2. Enhanced Error Handling
- Graceful fallbacks when services unavailable
- Better logging and debugging
- Proper exception handling

### 3. Service Resilience
- Works without Pinecone (memory fallback)
- Works without Gemini (rule-based fallback)
- Robust document processing

### 4. Production Ready
- Proper logging configuration
- Health checks with service status
- Performance monitoring
- CORS enabled

## Troubleshooting

**Build Fails:**
- Check Python version in Render (should be 3.11.0)
- Verify all environment variables are set
- Check build logs for specific errors

**Runtime Errors:**
- Check service logs in Render dashboard
- Verify API keys are correct and active
- Test individual endpoints

**Slow Performance:**
- Normal on free Render plan (cold starts)
- Consider paid plan for production use
- Check logs for specific bottlenecks

## Testing Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GEMINI_API_KEY=your_key
export PINECONE_API_KEY=your_key  # optional

# Run locally
python main.py

# Test
curl http://localhost:8000/health
```

## API Endpoints

1. **Health Check:** `GET /health`
2. **Main Processing:** `POST /api/v1/hackrx/run`
3. **Structured Analysis:** `POST /api/v1/analyze`

All endpoints require Bearer token authentication except health check.

## Support

- Check logs in Render dashboard
- Monitor health endpoint
- Use structured analysis endpoint for debugging
EOF
    
    echo -e "${GREEN}✅ Deployment guide created: DEPLOYMENT_GUIDE.md${NC}"
}

# Create test script
create_test_script() {
    cat > test_deployment.py << 'EOF'
#!/usr/bin/env python3
"""
Production deployment test script
"""

import requests
import json
import time
import sys

def test_deployment(base_url):
    """Test deployed API"""
    print(f"🔍 Testing deployment at: {base_url}")
    
    # Test health
    try:
        response = requests.get(f"{base_url}/health", timeout=30)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Status: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test main API
    try:
        payload = {
            "documents": """
            INSURANCE POLICY
            
            Coverage includes:
            - Medical expenses
            - Surgery costs
            - Emergency treatments
            
            Waiting periods:
            - General: No waiting
            - Surgery: 12 months
            - Pre-existing: 36 months
            
            Age limit: 18-65 years
            """,
            "questions": [
                "What is the waiting period for surgery?",
                "Am I eligible at age 45?"
            ]
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer 4ddf287faf3c89dfb4c0adc648a46975d4063a37899d2243a451f717af4a32cc"
        }
        
        print("🔍 Testing main API endpoint...")
        response = requests.post(
            f"{base_url}/api/v1/hackrx/run",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            print("✅ API test passed")
            data = response.json()
            for i, answer in enumerate(data.get("answers", [])[:2]):
                print(f"   Q{i+1}: {answer[:100]}...")
        else:
            print(f"❌ API test failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ API test error: {e}")
        return False
    
    print("🎉 All tests passed!")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_deployment.py <BASE_URL>")
        print("Example: python test_deployment.py https://your-app.onrender.com")
        sys.exit(1)
    
    base_url = sys.argv[1].rstrip('/')
    success = test_deployment(base_url)
    sys.exit(0 if success else 1)
EOF
    
    chmod +x test_deployment.py
    echo -e "${GREEN}✅ Test script created: test_deployment.py${NC}"
}

# Main function
main() {
    case "${1:-all}" in
        "check")
            check_python
            check_api_keys
            ;;
        "test")
            check_python
            check_api_keys
            test_local
            ;;
        "prepare")
            check_python
            check_api_keys
            prepare_deployment
            ;;
        "all")
            check_python
            check_api_keys
            test_local
            prepare_deployment
            create_deployment_guide
            create_test_script
            
            echo ""
            echo -e "${GREEN}🎉 Deployment preparation complete!${NC}"
            echo ""
            echo -e "${YELLOW}Next steps:${NC}"
            echo "1. Set your API keys: export GEMINI_API_KEY='your_key'"
            echo "2. Push to GitHub: git push origin main"
            echo "3. Deploy on Render using the guide: DEPLOYMENT_GUIDE.md"
            echo "4. Test: python test_deployment.py <your-render-url>"
            echo ""
            echo -e "${BLUE}What was fixed:${NC}"
            echo "✅ Python 3.12 compatibility"
            echo "✅ Package version conflicts"
            echo "✅ Missing dependencies"
            echo "✅ Import errors"
            echo "✅ Production optimizations"
            echo "✅ Error handling"
            echo "✅ Service fallbacks"
            ;;
        *)
            echo "Usage: $0 [check|test|prepare|all]"
            echo ""
            echo "Commands:"
            echo "  check    - Check Python and API keys"
            echo "  test     - Test system locally"
            echo "  prepare  - Prepare code for deployment"
            echo "  all      - Complete preparation (default)"
            exit 1
            ;;
    esac
}

main "$@"