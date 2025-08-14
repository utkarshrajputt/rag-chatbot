@echo off
echo ==========================================
echo    RAG Chatbot - Flask Application
echo ==========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv venv
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if .env file exists
if not exist ".env" (
    echo WARNING: .env file not found!
    echo Please create .env file with your OpenRouter API key
    echo.
)

REM Download NLTK data if needed
echo Checking NLTK data...
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

echo.
echo Starting RAG Chatbot server...
echo Open http://localhost:5000 in your browser
echo Press Ctrl+C to stop the server
echo.

REM Start Flask application
python app.py

pause
