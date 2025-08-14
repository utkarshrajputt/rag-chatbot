#!/bin/bash

echo "=========================================="
echo "    RAG Chatbot - Flask Application"
echo "=========================================="
echo

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please run: python -m venv venv"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "WARNING: .env file not found!"
    echo "Please create .env file with your OpenRouter API key"
    echo
fi

# Download NLTK data if needed
echo "Checking NLTK data..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

echo
echo "Starting RAG Chatbot server..."
echo "Open http://localhost:5000 in your browser"
echo "Press Ctrl+C to stop the server"
echo

# Start Flask application
python app.py
