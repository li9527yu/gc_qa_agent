d043ae67-959
nohup python -m uvicorn app.api_server:app --host 0.0.0.0 --port 8001 > logs/$(date +%Y%m%d_%H%M%S)_myapp.log
  2>&1 &

  curl -s -X POST http://localhost:8001/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "介绍下easy_rag 系统"}'


  curl -s -X POST "http://localhost:8001/api/v1/query/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "它有什么有点？",
    "session_id": "4b1b4f03-2c1"
  }'


  curl -X POST "http://localhost:8001/api/v1/query/price"     -H "Content-Type: application/json"     -d '{
       "question": "从信息价查电缆的价格"
     }'
