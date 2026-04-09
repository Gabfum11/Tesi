@echo off
py -3.10 -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
echo.
echo Virtual environment ready. To activate it later, run: venv\Scripts\activate
