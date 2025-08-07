#!/bin/bash
# port_fix.sh - Fixes Render deployment port configuration issue

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔧 FIXING RENDER PORT CONFIGURATION ISSUE${NC}"
echo "================================================================="
echo
echo -e "${YELLOW}Issue Identified:${NC}"
echo "• Your app is listening on PORT 10000 (Render's automatic setting)"
echo "• But Render is scanning for PORT 8000 (from your envVars override)"
echo "• This creates a port binding mismatch"
echo

# Fix 1: Update render.yaml to remove PORT override
fix_render_yaml() {
    echo -e "${BLUE}Fix 1: Updating render.yaml...${NC}"
    
    # Check if render.yaml exists
    if [ -f "render.yaml" ]; then
        # Create backup
        cp render.yaml render.yaml.backup
        echo "✅ Backup created: render.yaml.backup"
    elif [ -f "scripts/render.yaml" ]; then
        # Copy from scripts and backup
        cp scripts/render.yaml render.yaml
        cp scripts/render.yaml render.yaml.backup
        echo "✅ Copied render.yaml from scripts/"
    else
        echo -e "${RED}❌ render.yaml not found. Creating new one...${NC}"
    fi
    
    # Update the fixed render.yaml
    cat > render.yaml << 'EOF'
services:
  - type: web
    name: document-analysis-system
    env: python
    plan: starter

    buildCommand: |
      echo "=== Build Phase Started ==="
      echo "Current directory: $(pwd)"
      echo "Python version: $(python --version)"

      # Install system dependencies
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

      # Set environment for tokenizers compilation
      export CARGO_HOME=/tmp/.cargo
      export CARGO_TARGET_DIR=/tmp/.cargo-target

      # Ensure main.py is in root (this was the original issue)
      if [ -f "scripts/main.py" ]; then
        echo "Moving main.py from scripts/ to root directory..."
        cp scripts/main.py ./main.py
      fi

      # Verify main.py exists
      if [ ! -f "main.py" ]; then
        echo "ERROR: main.py not found in root directory!"
        find . -name "*.py" -type f
        exit 1
      fi

      echo "✅ main.py verified in root directory"

      # Install dependencies
      echo "Installing PyTorch (CPU-only)..."
      pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
      pip install --no-cache-dir numpy==2.1.3

      echo "Installing remaining dependencies..."
      pip install --no-cache-dir -r requirements.txt

      # Verify critical imports
      echo "Verifying imports..."
      python -c "import torch; print('✅ PyTorch:', torch.__version__)" || exit 1
      python -c "import numpy; print('✅ NumPy:', numpy.__version__)" || exit 1
      python -c "import sentence_transformers; print('✅ SentenceTransformers: OK')" || echo "⚠️  SentenceTransformers: Fallback mode"
      python -c "import main; print('✅ main.py imports successfully')" || exit 1

      echo "=== Build Phase Completed Successfully ==="

    # CRITICAL FIX: Let Render set PORT automatically (don't override it)
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT --log-level info

    envVars:
      # REMOVED PORT override - let Render set it automatically
      
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

      - key: PYTHONUNBUFFERED
        value: "1"

      - key: PYTHONDONTWRITEBYTECODE
        value: "1"

      # Rust/Cargo settings
      - key: CARGO_HOME
        value: /tmp/.cargo

      - key: CARGO_TARGET_DIR
        value: /tmp/.cargo-target

      # PyTorch CPU-only settings
      - key: TORCH_CUDA_ARCH_LIST
        value: ""

      - key: CUDA_VISIBLE_DEVICES
        value: ""

      # Memory optimization for Render
      - key: MALLOC_ARENA_MAX
        value: "2"

      - key: TOKENIZERS_PARALLELISM
        value: "false"

      - key: WEB_CONCURRENCY
        value: "1"

    healthCheckPath: /health
    autoDeploy: true
    buildTimeout: 30m
EOF

    echo -e "${GREEN}✅ render.yaml updated with port fix${NC}"
}

# Fix 2: Ensure main.py is in root directory
fix_main_py_location() {
    echo -e "${BLUE}Fix 2: Ensuring main.py is in root directory...${NC}"
    
    if [ -f "scripts/main.py" ] && [ ! -f "main.py" ]; then
        cp scripts/main.py ./main.py
        echo "✅ main.py copied to root directory"
    elif [ -f "main.py" ]; then
        echo "✅ main.py already exists in root directory"
    else
        echo -e "${RED}❌ main.py not found in scripts/ or root directory${NC}"
        echo "Please check your project structure"
        exit 1
    fi
}

# Fix 3: Update requirements.txt if needed
fix_requirements() {
    echo -e "${BLUE}Fix 3: Checking requirements.txt...${NC}"
    
    if [ ! -f "requirements.txt" ]; then
        if [ -f "scripts/requirements.txt" ]; then
            cp scripts/requirements.txt ./requirements.txt
            echo "✅ requirements.txt copied to root directory"
        else
            echo -e "${YELLOW}⚠️  Creating requirements.txt in root directory${NC}"
            cat > requirements.txt << 'EOF'
# Core web framework
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

# Vector database and embeddings
pinecone-client==5.3.1
sentence-transformers==3.3.1
numpy==2.1.3

# PyTorch CPU-only
torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# LLM API
google-generativeai==0.8.3

# Additional dependencies
python-multipart==0.0.17
transformers==4.46.3
tokenizers==0.20.3
huggingface-hub==0.26.2
safetensors==0.4.5
regex==2024.11.6
tqdm==4.67.0
EOF
        fi
    else
        echo "✅ requirements.txt already exists"
    fi
}

# Test local configuration
test_local() {
    echo -e "${BLUE}Fix 4: Testing local configuration...${NC}"
    
    # Quick Python import test
    python3 -c "
import sys
print('Python version:', sys.version)
print('Checking if main.py can be imported...')

try:
    # Just check syntax without running
    import ast
    with open('main.py', 'r') as f:
        ast.parse(f.read())
    print('✅ main.py syntax is valid')
except Exception as e:
    print('❌ main.py syntax error:', e)
    sys.exit(1)
    "
}

# Create deployment summary
create_summary() {
    echo -e "${BLUE}Creating deployment summary...${NC}"
    
    cat > PORT_FIX_SUMMARY.md << 'EOF'
# 🔧 PORT CONFIGURATION FIX SUMMARY

## 🚨 Issue Fixed
**Root Cause**: Port binding mismatch between Render's automatic PORT setting and your envVars override.

### What Was Wrong:
```
❌ render.yaml had: PORT=8000 (in envVars)
❌ Render automatically sets: PORT=10000
❌ App listens on: PORT 10000 (from Render's $PORT)
❌ Render expects app on: PORT 8000 (from your override)
❌ Result: Port scan timeout
```

### What's Fixed:
```
✅ Removed PORT override from render.yaml envVars
✅ App will listen on: $PORT (whatever Render sets)
✅ Render will scan: same $PORT value
✅ Result: Perfect match!
```

## 🚀 Files Updated

### 1. `render.yaml` - CRITICAL CHANGES
- ✅ **REMOVED** `PORT=8000` from envVars
- ✅ Render now sets PORT automatically (usually 10000)
- ✅ App listens on `$PORT` (matches what Render expects)

### 2. `main.py` - Port Handling
- ✅ Uses `os.getenv("PORT", 8000)` (8000 for local dev only)
- ✅ Render will override with its own PORT value

### 3. Project Structure
- ✅ `main.py` moved to root directory
- ✅ `requirements.txt` in root directory
- ✅ `render.yaml` in root directory

## 🎯 Why This Fixes The Issue

**Before (Broken):**
```bash
# Render sets environment
export PORT=10000

# Your envVars override it
export PORT=8000

# App starts
uvicorn main:app --port $PORT  # Uses 10000 (from Render)

# Render health check
curl localhost:8000/health  # FAILS - nothing listening on 8000
```

**After (Fixed):**
```bash
# Render sets environment  
export PORT=10000

# No override in envVars - uses Render's value

# App starts
uvicorn main:app --port $PORT  # Uses 10000

# Render health check
curl localhost:10000/health  # SUCCESS - app listening on 10000
```

## 🚀 Deployment Steps

1. **Commit the fixes:**
```bash
git add .
git commit -m "Fix: Remove PORT override for Render deployment"
git push origin main
```

2. **Redeploy in Render:**
- Go to your Render dashboard
- Click "Manual Deploy" or wait for auto-deploy
- Watch build logs - should complete successfully
- Health check should pass

3. **Verify deployment:**
```bash
curl https://your-app.onrender.com/health
# Should return: {"status": "ok", "port": 10000, ...}
```

## 🎉 Expected Result

- ✅ Build completes successfully
- ✅ App starts on correct port
- ✅ Health check passes  
- ✅ No more "port scan timeout" errors
- ✅ API endpoints work correctly

The port mismatch is now FIXED! 🎯
EOF

    echo -e "${GREEN}✅ Summary created: PORT_FIX_SUMMARY.md${NC}"
}

# Main execution
main() {
    echo -e "${YELLOW}This script will fix your Render port configuration issue.${NC}"
    echo
    
    # Execute all fixes
    fix_render_yaml
    fix_main_py_location  
    fix_requirements
    test_local
    create_summary
    
    echo
    echo -e "${GREEN}🎉 PORT CONFIGURATION FIX COMPLETE!${NC}"
    echo
    echo -e "${YELLOW}📋 What Was Fixed:${NC}"
    echo "✅ Removed PORT=8000 override from render.yaml"
    echo "✅ App will now use Render's automatic PORT setting"  
    echo "✅ main.py moved to root directory"
    echo "✅ All configuration files updated"
    echo
    echo -e "${BLUE}🚀 Next Steps:${NC}"
    echo "1. git add ."
    echo "2. git commit -m 'Fix: Remove PORT override for Render deployment'"
    echo "3. git push origin main"
    echo "4. Redeploy on Render (should work now!)"
    echo
    echo -e "${GREEN}📖 Read PORT_FIX_SUMMARY.md for detailed explanation${NC}"
}

# Check if script is being run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi