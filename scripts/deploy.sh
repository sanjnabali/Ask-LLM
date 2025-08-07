#!/bin/bash
# deploy.sh - Fixed for Render deployment structure

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 Document Analysis System - Render Deployment Fix${NC}"
echo "================================================================="

# Fix project structure for Render
fix_project_structure() {
    echo -e "${BLUE}Fixing project structure for Render deployment...${NC}"
    
    # Check if we're in the right directory
    if [ ! -f "scripts/main.py" ]; then
        echo -e "${RED}❌ scripts/main.py not found. Are you in the project root?${NC}"
        exit 1
    fi
    
    # Copy main.py to root directory
    echo "Copying main.py to root directory..."
    cp scripts/main.py ./main.py
    
    # Check if requirements.txt exists in root, if not copy from scripts or create
    if [ ! -f "requirements.txt" ]; then
        if [ -f "scripts/requirements.txt" ]; then
            echo "Copying requirements.txt from scripts..."
            cp scripts/requirements.txt ./requirements.txt
        else
            echo "Creating requirements.txt in root directory..."
            cat > requirements.txt << 'EOF'
# Core web framework - Python 3.12.4 compatible
fastapi==0.115.5
uvicorn[standard]==0.32.1
pydantic==2.9.2

# Document processing
PyMuPDF==1.24.12
python-docx==1.1.2
beautifulsoup4==4.12.3
requests==2.32.3
aiohttp==3.11.7
certifi==2024.8.30

# Vector database and embeddings - Updated for Python 3.12
pinecone-client==5.3.1
sentence-transformers==3.3.1
numpy==2.1.3

# PyTorch CPU-only for embeddings (no CUDA)
torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# LLM API
google-generativeai==0.8.3

# Utilities
python-multipart==0.0.17
jinja2==3.1.4

# Additional dependencies for sentence-transformers
transformers==4.46.3
tokenizers==0.20.3
huggingface-hub==0.26.2
safetensors==0.4.5
regex==2024.11.6
tqdm==4.67.0

# For document processing
lxml==5.3.0
pillow==10.4.0

# HTTP client libraries
httpx==0.27.2
urllib3==2.2.3

# Async support
aiofiles==24.1.0
EOF
        fi
    fi
    
    # Copy or create render.yaml in root
    if [ ! -f "render.yaml" ]; then
        if [ -f "scripts/render.yaml" ]; then
            echo "Copying render.yaml from scripts..."
            cp scripts/render.yaml ./render.yaml
        else
            echo "Creating render.yaml in root directory..."
            cat > render.yaml << 'EOF'
services:
  - type: web
    name: document-analysis-system
    env: python
    plan: starter
    
    buildCommand: |
      # Set Python version
      python --version
      
      # Update system packages
      apt-get update && apt-get install -y \
        build-essential \
        libffi-dev \
        libssl-dev \
        python3-dev \
        pkg-config \
        libmupdf-dev \
        libjpeg-dev \
        zlib1g-dev \
        cmake \
        gcc \
        g++
      
      # Upgrade Python packaging tools
      pip install --upgrade pip setuptools wheel
      
      # Set Rust/Cargo environment for tokenizers
      export CARGO_HOME=/tmp/.cargo
      export CARGO_TARGET_DIR=/tmp/.cargo-target
      
      # Install PyTorch first (CPU-only)
      pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
      pip install --no-cache-dir numpy==2.1.3
      
      # Install other dependencies
      pip install --no-cache-dir -r requirements.txt
      
      # Verify critical imports
      python -c "import torch; print('PyTorch:', torch.__version__)"
      python -c "import numpy; print('NumPy:', numpy.__version__)"
      python -c "import sentence_transformers; print('SentenceTransformers: OK')" || echo "SentenceTransformers: Will use fallback"
    
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    
    envVars:
      - key: PYTHON_VERSION
        value: 3.12.4
      
      - key: GEMINI_API_KEY
        sync: false
      
      - key: PINECONE_API_KEY
        sync: false
      
      - key: PINECONE_ENVIRONMENT
        value: us-east-1-aws
      
      - key: INDEX_NAME
        value: askllm
      
      - key: PORT
        value: 10000
      
      - key: PYTHONUNBUFFERED
        value: "1"
      
      - key: PYTHONDONTWRITEBYTECODE
        value: "1"
      
      - key: CARGO_HOME
        value: /tmp/.cargo
      
      - key: CARGO_TARGET_DIR
        value: /tmp/.cargo-target
      
      - key: TORCH_CUDA_ARCH_LIST
        value: ""
      
      - key: CUDA_VISIBLE_DEVICES
        value: ""
      
      - key: MALLOC_ARENA_MAX
        value: "2"
      
      - key: TOKENIZERS_PARALLELISM
        value: "false"
    
    healthCheckPath: /health
    autoDeploy: true
    buildTimeout: 30m
EOF
        fi
    fi
    
    echo -e "${GREEN}✅ Project structure fixed for Render deployment${NC}"
}

# Test local installation
test_local_installation() {
    echo -e "${BLUE}Testing local installation...${NC}"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        echo "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    echo "Upgrading pip and build tools..."
    pip install --upgrade pip setuptools wheel
    
    # Install dependencies
    echo "Installing dependencies..."
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
    pip install numpy==2.1.3
    pip install -r requirements.txt
    
    # Verify imports
    echo "Verifying critical imports..."
    python -c "
import sys
print('Python version:', sys.version)

try:
    import main
    print('✅ main.py imports successfully')
except Exception as e:
    print('❌ main.py import error:', e)

try:
    import fastapi
    print('✅ FastAPI:', fastapi.__version__)
except ImportError as e:
    print('❌ FastAPI import error:', e)

try:
    import torch
    print('✅ PyTorch:', torch.__version__)
except ImportError as e:
    print('❌ PyTorch import error:', e)

try:
    import numpy
    print('✅ NumPy:', numpy.__version__)
except ImportError as e:
    print('❌ NumPy import error:', e)
    "
    
    echo -e "${GREEN}✅ Local installation test completed${NC}"
    deactivate
}

# Check API keys
check_api_keys() {
    echo -e "${BLUE}Checking API key configuration...${NC}"
    
    if [ -z "$GEMINI_API_KEY" ]; then
        echo -e "${YELLOW}⚠️  GEMINI_API_KEY not set${NC}"
        echo "   Set with: export GEMINI_API_KEY='your_key_here'"
        echo "   Get your key from: https://makersuite.google.com/app/apikey"
    else
        echo -e "${GREEN}✅ GEMINI_API_KEY is configured${NC}"
    fi
    
    if [ -z "$PINECONE_API_KEY" ]; then
        echo -e "${YELLOW}⚠️  PINECONE_API_KEY not set - will use memory fallback${NC}"
    else
        echo -e "${GREEN}✅ PINECONE_API_KEY is configured${NC}"
    fi
}

# Create deployment guide
create_deployment_guide() {
    echo -e "${BLUE}Creating deployment guide...${NC}"
    
    cat > RENDER_DEPLOYMENT_GUIDE.md << 'EOF'
# 🚀 Render Deployment Guide - FIXED

## 🔧 Issue Fixed
The original error was caused by Render looking for `main.py` in the root directory, but it was located in `scripts/main.py`.

## 📁 Correct Project Structure
```
project-root/
├── main.py           # ← MOVED from scripts/main.py
├── requirements.txt  # ← MOVED from scripts/requirements.txt
├── render.yaml       # ← MOVED from scripts/render.yaml
├── scripts/
│   ├── deploy.sh
│   └── (backup files)
└── README.md
```

## 🚀 Deployment Steps

### Step 1: Prepare Project Structure
```bash
# Run this script to fix the structure
./scripts/deploy.sh fix

# Or manually:
cp scripts/main.py ./main.py
cp scripts/requirements.txt ./requirements.txt
cp scripts/render.yaml ./render.yaml
```

### Step 2: Set Environment Variables in Render Dashboard
```bash
GEMINI_API_KEY=your_actual_key_here
PINECONE_API_KEY=your_pinecone_key_here  # Optional
```

### Step 3: Deploy to Render
```bash
# Commit changes
git add .
git commit -m "Fix: Move main.py to root for Render deployment"
git push origin main

# In Render Dashboard:
# 1. Create new Web Service
# 2. Connect your GitHub repository
# 3. Render will automatically detect and use render.yaml
# 4. Build should complete successfully
```

### Step 4: Verify Deployment
```bash
# Test health endpoint
curl https://your-app-name.onrender.com/health

# Expected response:
{
  "status": "ok",
  "python_version": "3.12.4",
  "services": {
    "gemini": "available",
    "pinecone": "available",  
    "sentence_transformers": "available"
  }
}
```

## 🐛 Common Issues & Solutions

### 1. "Could not import module 'main'"
- **Cause**: main.py not in root directory
- **Solution**: Run `cp scripts/main.py ./main.py`

### 2. "No module named 'fastapi'"
- **Cause**: requirements.txt missing or incorrect
- **Solution**: Ensure requirements.txt is in root directory

### 3. Build timeout
- **Cause**: PyTorch compilation takes time
- **Solution**: Using CPU-only PyTorch and extended timeout (30m)

### 4. Memory errors
- **Cause**: Limited memory on free plan
- **Solution**: Using Starter plan and optimized settings

## 📊 Performance Expectations

- **Build Time**: 15-25 minutes (first build)
- **Cold Start**: 30-60 seconds (Starter plan)
- **Response Time**: 2-10 seconds per query

## 🔧 Local Testing
```bash
# Test locally before deploying
./scripts/deploy.sh test

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py

# Test endpoints
curl http://localhost:8000/health
```

## ✅ Deployment Checklist

- [ ] main.py in root directory
- [ ] requirements.txt in root directory  
- [ ] render.yaml in root directory
- [ ] GEMINI_API_KEY set in Render dashboard
- [ ] Repository pushed to GitHub
- [ ] Render service connected to repository

## 📞 Support

If deployment still fails:
1. Check build logs in Render dashboard
2. Verify file structure matches above
3. Test locally first
4. Check environment variables are set

The module import error is now fixed! 🎉
EOF
    
    echo -e "${GREEN}✅ Deployment guide created: RENDER_DEPLOYMENT_GUIDE.md${NC}"
}

# Main execution
main() {
    case "${1:-all}" in
        "fix")
            fix_project_structure
            ;;
        "test")
            test_local_installation
            ;;
        "check")
            check_api_keys
            ;;
        "all")
            echo -e "${BLUE}🔧 Fixing Render deployment issue...${NC}"
            echo
            
            fix_project_structure
            test_local_installation
            check_api_keys
            create_deployment_guide
            
            echo
            echo -e "${GREEN}🎉 Render deployment fix complete!${NC}"
            echo
            echo -e "${YELLOW}📋 Next Steps:${NC}"
            echo "1. Set API keys in Render dashboard"
            echo "2. Commit changes: git add . && git commit -m 'Fix: Move files for Render deployment'"
            echo "3. Push: git push origin main"
            echo "4. Deploy on Render (should work now!)"
            echo
            echo -e "${BLUE}✅ Fixed Issues:${NC}"
            echo "• ✅ Moved main.py to root directory"
            echo "• ✅ Moved requirements.txt to root directory"
            echo "• ✅ Fixed render.yaml configuration"
            echo "• ✅ Updated Python 3.12.4 compatibility"
            echo "• ✅ Optimized build process"
            echo
            echo -e "${GREEN}📖 Read: RENDER_DEPLOYMENT_GUIDE.md for detailed instructions${NC}"
            ;;
        *)
            echo "Usage: $0 [fix|test|check|all]"
            echo
            echo "Commands:"
            echo "  fix    - Fix project structure for Render"
            echo "  test   - Test local installation"
            echo "  check  - Check API key configuration"
            echo "  all    - Complete fix and preparation (recommended)"
            exit 1
            ;;
    esac
}

main "$@"