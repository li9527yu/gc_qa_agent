d043ae67-959


  curl -s -X POST http://localhost:8001/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "介绍下easy_rag 系统"}'


  curl -s -X POST "http://localhost:8001/api/v1/query/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "它有什么有点？",
    "session_id": "4b1b4f03-2c1"
  }'