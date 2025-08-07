#!/bin/bash
# render_deploy.sh - Render-specific deployment script

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 Document Analysis System - Render Deployment${NC}"
echo "=================================================="

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
        echo -e "${RED}❌ GEMINI_API_KEY not set${NC}"
        echo -e "${YELLOW}Get your key from: https://makersuite.google.com/app/apikey${NC}"
        echo "Set it with: export GEMINI_API_KEY='your_key_here'"
        exit 1
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
    
    # Install dependencies
    echo "Installing dependencies..."
    pip install -r requirements.txt
    
    # Quick test
    echo "Starting test server..."
    python main.py &
    SERVER_PID=$!
    
    # Wait for server to start
    sleep 8
    
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
}

# Prepare for Render deployment
prepare_render() {
    echo -e "${BLUE}Preparing for Render deployment...${NC}"
    
    # Create/update requirements.txt with exact versions for stability
    cat > requirements.txt << 'EOF'
# Core Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6

# Document Processing
PyMuPDF==1.23.9
python-docx==0.8.11
beautifulsoup4==4.12.2
requests==2.31.0
aiohttp==3.9.0
certifi==2023.11.17

# Vector Database and Embeddings
pinecone-client==3.0.0
sentence-transformers==2.2.2
numpy==1.24.3

# Database (optional)
psycopg2-binary==2.9.9
asyncpg==0.29.0

# LLM Integration
google-generativeai==0.3.2

# Data Processing
pydantic==2.5.0

# Utilities
python-dotenv==1.0.0
typing-extensions==4.8.0
EOF
    
    echo -e "${GREEN}✅ requirements.txt updated for Render${NC}"
    
    # Check if git repo exists
    if [ ! -d ".git" ]; then
        echo "Initializing git repository..."
        git init
        echo "node_modules/" > .gitignore
        echo "__pycache__/" >> .gitignore
        echo ".env" >> .gitignore
        echo "*.pyc" >> .gitignore
    fi
    
    # Stage all files
    git add .
    
    # Commit if there are changes
    if ! git diff --staged --quiet; then
        git commit -m "Prepare for Render deployment - $(date)"
        echo -e "${GREEN}✅ Changes committed${NC}"
    else
        echo -e "${YELLOW}No changes to commit${NC}"
    fi
}

# Generate Render deployment instructions
render_instructions() {
    echo -e "${BLUE}Creating Render deployment guide...${NC}"
    
    cat > RENDER_DEPLOYMENT.md << 'EOF'
# 🚀 Render Deployment Instructions

## Step 1: Push to GitHub
```bash
# Add your GitHub repository (replace with your repo URL)
git remote add origin https://github.com/yourusername/document-analysis-system.git
git branch -M main
git push -u origin main
```

## Step 2: Deploy on Render

1. **Go to Render Dashboard**
   - Visit: https://render.com
   - Sign in with GitHub

2. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select your `document-analysis-system` repo

3. **Configure Service Settings**
   ```
   Name: document-analysis-api
   Environment: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: python main.py
   ```

4. **Set Environment Variables**
   In the Render dashboard, add these environment variables:
   ```
   GEMINI_API_KEY = your_gemini_api_key_here
   PINECONE_API_KEY = your_pinecone_key (optional)
   PINECONE_ENVIRONMENT = gcp-starter
   PYTHON_VERSION = 3.11.0
   PORT = 8000
   ```

5. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment (5-10 minutes)
   - Your API will be live at: `https://your-service-name.onrender.com`

## Step 3: Test Your Deployment

```bash
# Test health endpoint
curl https://your-service-name.onrender.com/health

# Test main API
curl -X POST https://your-service-name.onrender.com/api/v1/hackrx/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 4ddf287faf3c89dfb4c0adc648a46975d4063a37899d2243a451f717af4a32cc" \
  -d '{
    "documents": "Sample policy text here...",
    "questions": ["Test question?"]
  }'
```

## Render-Specific Notes

- **Free Plan Limitations**: 
  - Service sleeps after 15 minutes of inactivity
  - 750 build hours per month
  - First request after sleep takes ~30 seconds

- **Monitoring**: 
  - Check logs in Render dashboard
  - Monitor at: https://your-service.onrender.com/health

- **Updates**: 
  - Push to GitHub main branch
  - Render auto-deploys on git push

## Troubleshooting

**Build Failures:**
- Check logs in Render dashboard
- Verify requirements.txt format
- Ensure Python 3.11+ specified

**Runtime Errors:**
- Check environment variables are set
- Verify API keys are valid
- Monitor logs for specific errors

**Slow Cold Starts:**
- Use Render paid plan for always-on services
- Or implement health check pings every 10 minutes
EOF
    
    echo -e "${GREEN}✅ Deployment guide created: RENDER_DEPLOYMENT.md${NC}"
}

# Create optimized test script for Render
create_render_test() {
    cat > test_render_deployment.py << 'EOF'
#!/usr/bin/env python3
"""
Test script specifically for Render deployment
"""

import requests
import json
import time
import sys
from typing import Dict, Any

# Configuration
AUTH_TOKEN = "4ddf287faf3c89dfb4c0adc648a46975d4063a37899d2243a451f717af4a32cc"

class RenderTester:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AUTH_TOKEN}"
        }
        
    def test_health(self) -> bool:
        """Test health endpoint"""
        print("🔍 Testing health endpoint...")
        try:
            response = requests.get(f"{self.base_url}/health", timeout=30)
            if response.status_code == 200 and "ok" in response.text:
                print("✅ Health check passed")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_main_endpoint(self) -> bool:
        """Test main hackrx endpoint"""
        print("🔍 Testing main API endpoint...")
        
        payload = {
            "documents": """
            INSURANCE POLICY DOCUMENT
            
            This policy provides coverage for medical expenses including:
            - Hospitalization costs
            - Surgery expenses
            - Emergency treatments
            
            WAITING PERIODS:
            - General treatments: No waiting period
            - Pre-existing conditions: 36 months
            - Specific surgeries (knee, hip): 24 months
            
            AGE LIMITS: 18-65 years
            GEOGRAPHIC COVERAGE: Worldwide
            
            POLICY REQUIREMENTS:
            - Minimum 12 months for surgery coverage
            - Continuous premium payments required
            """,
            "questions": [
                "What is the waiting period for knee surgery?",
                "Is a 46-year-old eligible for coverage?",
                "46M, knee surgery, 3-month policy - covered?"
            ]
        }
        
        try:
            print("Sending request... (this may take 10-30 seconds)")
            response = requests.post(
                f"{self.base_url}/api/v1/hackrx/run",
                headers=self.headers,
                json=payload,
                timeout=60  # Longer timeout for Render cold start
            )
            
            if response.status_code == 200:
                data = response.json()
                print("✅ API endpoint test passed")
                print("\nSample Responses:")
                for i, answer in enumerate(data.get("answers", [])[:2]):
                    print(f"{i+1}. {answer[:100]}...")
                return True
            else:
                print(f"❌ API test failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ API test error: {e}")
            return False
    
    def test_cold_start_performance(self) -> Dict[str, Any]:
        """Test performance including cold start"""
        print("🔍 Testing performance (including cold start)...")
        
        times = []
        for i in range(3):
            print(f"Request {i+1}/3...", end=" ")
            start_time = time.time()
            
            try:
                response = requests.get(f"{self.base_url}/health", timeout=45)
                end_time = time.time()
                duration = end_time - start_time
                times.append(duration)
                print(f"{duration:.2f}s")
                
                if i == 0 and duration > 20:
                    print("⚠️  Cold start detected (normal for free Render plan)")
                    
            except Exception as e:
                print(f"Failed: {e}")
                times.append(None)
        
        valid_times = [t for t in times if t is not None]
        if valid_times:
            avg_time = sum(valid_times) / len(valid_times)
            print(f"📊 Average response time: {avg_time:.2f}s")
            if avg_time < 5:
                print("✅ Good performance")
            elif avg_time < 15:
                print("⚠️  Acceptable performance")
            else:
                print("⚠️  Slow performance (consider Render paid plan)")
        
        return {"times": times, "average": avg_time if valid_times else None}
    
    def run_comprehensive_test(self) -> bool:
        """Run all tests"""
        print(f"🚀 Testing Render deployment at: {self.base_url}")
        print("=" * 60)
        
        # Test 1: Health check
        health_ok = self.test_health()
        if not health_ok:
            return False
        
        # Test 2: Performance check
        perf_results = self.test_cold_start_performance()
        
        # Test 3: Main API
        api_ok = self.test_main_endpoint()
        
        print("\n" + "=" * 60)
        if health_ok and api_ok:
            print("🎉 All tests passed! Your Render deployment is working.")
            print(f"📡 API URL: {self.base_url}")
            print("🔗 Share this URL for the hackathon submission")
            return True
        else:
            print("❌ Some tests failed. Check the logs above.")
            return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python test_render_deployment.py <RENDER_URL>")
        print("Example: python test_render_deployment.py https://your-app.onrender.com")
        sys.exit(1)
    
    url = sys.argv[1]
    tester = RenderTester(url)
    success = tester.run_comprehensive_test()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
EOF
    
    chmod +x test_render_deployment.py
    echo -e "${GREEN}✅ Render test script created: test_render_deployment.py${NC}"
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
            test_local
            prepare_render
            ;;
        "all")
            check_python
            check_api_keys
            test_local
            prepare_render
            render_instructions
            create_render_test
            
            echo ""
            echo -e "${GREEN}🎉 Ready for Render deployment!${NC}"
            echo ""
            echo -e "${YELLOW}Next steps:${NC}"
            echo "1. Push to GitHub: git remote add origin <your-repo-url>"
            echo "2. Follow instructions in: RENDER_DEPLOYMENT.md"
            echo "3. Test deployment: python test_render_deployment.py <render-url>"
            echo ""
            echo -e "${BLUE}Quick deploy checklist:${NC}"
            echo "✅ Local tests passed"
            echo "✅ Code ready for Render"
            echo "✅ Dependencies optimized"
            echo "✅ Test scripts ready"
            ;;
        *)
            echo "Usage: $0 [check|test|prepare|all]"
            echo ""
            echo "Commands:"
            echo "  check    - Check Python and API keys"
            echo "  test     - Test system locally"
            echo "  prepare  - Prepare code for Render"
            echo "  all      - Complete preparation (default)"
            exit 1
            ;;
    esac
}

main "$@"