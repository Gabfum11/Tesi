#!/bin/bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo ""
echo "Virtual environment ready. To activate it later, run: source venv/bin/activate"
