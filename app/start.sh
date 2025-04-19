#!/bin/bash
# start.sh
echo "Starting Streamlit frontend..."
streamlit run frontend/app.py --server.port 10000 --server.address 0.0.0.0 &

echo "Waiting for FastAPI to start..."
sleep 10

echo "Starting FastAPI backend..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 


