#!/bin/bash

# Market Intelligence Agent - Setup and Run Script
# This script automates the setup and running of the Market Intelligence Agent application

# Text colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_message() {
  echo -e "${BLUE}[Market Intelligence Agent]${NC} $1"
}

print_success() {
  echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
  echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

# Check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check for required dependencies
check_dependencies() {
  print_message "Checking for required dependencies..."
  
  # Check for Python
  if ! command_exists python3; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
  fi
  
  # Check for pip
  if ! command_exists pip3; then
    print_error "pip3 is not installed. Please install pip3."
    exit 1
  fi
  
  # Check for Node.js
  if ! command_exists node; then
    print_error "Node.js is not installed. Please install Node.js 16 or higher."
    exit 1
  fi
  
  # Check for npm
  if ! command_exists npm; then
    print_error "npm is not installed. Please install npm."
    exit 1
  fi
  
  print_success "All required dependencies are installed."
}

# Extract the archive if it exists
extract_archive() {
  print_message "Checking for project archive..."
  
  if [ -f "market-intelligence-agent-enhanced.tar.gz" ]; then
    print_message "Extracting project archive..."
    tar -xzf market-intelligence-agent-enhanced.tar.gz
    if [ $? -ne 0 ]; then
      print_error "Failed to extract the archive."
      exit 1
    fi
    print_success "Archive extracted successfully."
  elif [ -d "market-intelligence-agent-enhanced" ]; then
    print_message "Project directory already exists. Skipping extraction."
  else
    print_error "Project archive not found. Please ensure market-intelligence-agent-enhanced.tar.gz is in the current directory."
    exit 1
  fi
}

# Setup Python virtual environment and install dependencies
setup_backend() {
  print_message "Setting up backend..."
  
  cd market-intelligence-agent-enhanced || exit 1
  
  # Create virtual environment
  print_message "Creating Python virtual environment..."
  python3 -m venv venv
  if [ $? -ne 0 ]; then
    print_error "Failed to create virtual environment."
    exit 1
  fi
  
  # Activate virtual environment
  print_message "Activating virtual environment..."
  source venv/bin/activate
  if [ $? -ne 0 ]; then
    print_error "Failed to activate virtual environment."
    exit 1
  fi
  
  # Install dependencies
  print_message "Installing Python dependencies (this may take a few minutes)..."
  pip install -r requirements.txt
  if [ $? -ne 0 ]; then
    print_error "Failed to install Python dependencies."
    exit 1
  fi
  
  print_success "Backend setup completed successfully."
}

# Setup frontend dependencies
setup_frontend() {
  print_message "Setting up frontend..."
  
  cd frontend/market-intel-ui || exit 1
  
  # Install dependencies with legacy-peer-deps to handle dependency conflicts
  print_message "Installing Node.js dependencies (this may take a few minutes)..."
  print_message "Using --legacy-peer-deps to handle dependency conflicts..."
  npm install --legacy-peer-deps
  
  # If the first attempt fails, try with --force
  if [ $? -ne 0 ]; then
    print_warning "First installation attempt failed. Trying with --force..."
    npm install --force
    
    if [ $? -ne 0 ]; then
      print_error "Failed to install Node.js dependencies after multiple attempts."
      print_message "You may need to manually install dependencies with: cd frontend/market-intel-ui && npm install --legacy-peer-deps"
      exit 1
    fi
  fi
  
  print_success "Frontend setup completed successfully."
  
  # Return to project root
  cd ../..
}

# Check if .env file exists, create if not
setup_env() {
  print_message "Checking for environment configuration..."
  
  if [ ! -f ".env" ]; then
    print_message "Creating .env file with default values..."
    cat > .env << EOL
SECRET_KEY=$(openssl rand -hex 32)
GOOGLE_API_KEY=your_google_api_key
NEWSAPI_KEY=your_newsapi_key
ALPHA_VANTAGE_KEY=your_alpha_vantage_key
TAVILY_API_KEY=your_tavily_api_key
EOL
    print_warning "Created .env file with placeholder API keys. Please edit the .env file with your actual API keys."
  else
    print_message ".env file already exists."
  fi
}

# Start the backend server
start_backend() {
  print_message "Starting backend server..."
  
  # Activate virtual environment if not already activated
  if [ -z "$VIRTUAL_ENV" ]; then
    source venv/bin/activate
  fi
  
  # Start the server in the background
  uvicorn app:app --host 0.0.0.0 --port 8000 &
  BACKEND_PID=$!
  
  # Check if server started successfully
  sleep 3
  if ps -p $BACKEND_PID > /dev/null; then
    print_success "Backend server started successfully (PID: $BACKEND_PID)."
  else
    print_error "Failed to start backend server."
    exit 1
  fi
}

# Start the frontend development server
start_frontend() {
  print_message "Starting frontend development server..."
  
  cd frontend/market-intel-ui || exit 1
  
  # Start the development server in the background
  npm run dev &
  FRONTEND_PID=$!
  
  # Check if server started successfully
  sleep 5
  if ps -p $FRONTEND_PID > /dev/null; then
    print_success "Frontend development server started successfully (PID: $FRONTEND_PID)."
  else
    print_error "Failed to start frontend development server."
    exit 1
  fi
  
  # Return to project root
  cd ../..
}

# Display application access information
display_access_info() {
  print_message "Market Intelligence Agent is now running!"
  echo -e "${GREEN}----------------------------------------${NC}"
  echo -e "${GREEN}Backend API:${NC} http://localhost:8000"
  echo -e "${GREEN}API Documentation:${NC} http://localhost:8000/docs"
  echo -e "${GREEN}Frontend UI:${NC} http://localhost:5173"
  echo -e "${GREEN}----------------------------------------${NC}"
  echo ""
  print_message "Press Ctrl+C to stop the application."
}

# Cleanup function to handle script termination
cleanup() {
  print_message "Stopping servers..."
  
  # Kill the backend server
  if [ ! -z "$BACKEND_PID" ]; then
    kill $BACKEND_PID 2>/dev/null
    print_message "Backend server stopped."
  fi
  
  # Kill the frontend server
  if [ ! -z "$FRONTEND_PID" ]; then
    kill $FRONTEND_PID 2>/dev/null
    print_message "Frontend server stopped."
  fi
  
  print_message "Cleanup completed. Thank you for using Market Intelligence Agent!"
  exit 0
}

# Register the cleanup function to run on script termination
trap cleanup SIGINT SIGTERM

# Main execution
print_message "Starting Market Intelligence Agent setup and run script..."

check_dependencies
extract_archive
setup_backend
setup_env
setup_frontend
start_backend
start_frontend
display_access_info

# Keep the script running to maintain the servers
while true; do
  sleep 1
done
