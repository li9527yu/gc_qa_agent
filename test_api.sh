#!/bin/bash

# Easy-RAG API 测试脚本
# 使用方法: ./test_api.sh [command]
# 命令: health | agent | mcp | price | dialogue | all

BASE_URL="http://localhost:8001"
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查jq是否安装
if ! command -v jq &> /dev/null; then
    echo "警告: jq未安装，JSON格式化将不可用"
    echo "macOS: brew install jq"
    echo "Ubuntu: sudo apt-get install jq"
    JQ_CMD="cat"
else
    JQ_CMD="jq"
fi

# 打印标题
print_title() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# 打印成功
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# 测试服务健康
test_health() {
    print_title "1. 服务健康检查"
    
    RESPONSE=$(curl -s "$BASE_URL/")
    echo "响应:"
    echo "$RESPONSE" | $JQ_CMD
    
    if echo "$RESPONSE" | grep -q "healthy"; then
        print_success "服务运行正常"
    else
        echo "错误: 服务可能未启动"
        exit 1
    fi
}

# 测试服务状态
test_service_status() {
    print_title "2. 服务状态检查"
    
    RESPONSE=$(curl -s "$BASE_URL/api/v1/service/status")
    echo "响应:"
    echo "$RESPONSE" | $JQ_CMD
}

# 测试Agent对话 - 完整流程
test_agent() {
    print_title "3. Agent对话测试"
    
    # 步骤1: 首次查询
    echo ""
    echo -e "${YELLOW}步骤1: 首次查询${NC}"
    RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/agent/chat" \
        -H "Content-Type: application/json" \
        -d '{"message": "查一下钢筋的价格"}')
    
    echo "响应:"
    echo "$RESPONSE" | $JQ_CMD
    
    SESSION_ID=$(echo "$RESPONSE" | grep -o '"session_id": "[^"]*"' | cut -d'"' -f4)
    STATE=$(echo "$RESPONSE" | grep -o '"state": "[^"]*"' | cut -d'"' -f4)
    
    echo ""
    echo "Session ID: $SESSION_ID"
    echo "State: $STATE"
    
    if [ -z "$SESSION_ID" ] || [ "$SESSION_ID" = "null" ]; then
        echo "错误: 无法获取session_id"
        return 1
    fi
    
    # 步骤2: 补充省份
    if [ "$STATE" = "collecting_entities" ] || [ "$STATE" = "confirming_candidate" ]; then
        echo ""
        echo -e "${YELLOW}步骤2: 补充省份${NC}"
        RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/agent/chat" \
            -H "Content-Type: application/json" \
            -d "{\"message\": \"广东省\", \"session_id\": \"$SESSION_ID\"}")
        
        echo "响应:"
        echo "$RESPONSE" | $JQ_CMD
        STATE=$(echo "$RESPONSE" | grep -o '"state": "[^"]*"' | cut -d'"' -f4)
    fi
    
    # 步骤3: 补充城市
    if [ "$STATE" = "collecting_entities" ]; then
        echo ""
        echo -e "${YELLOW}步骤3: 补充城市${NC}"
        RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/agent/chat" \
            -H "Content-Type: application/json" \
            -d "{\"message\": \"深圳市\", \"session_id\": \"$SESSION_ID\"}")
        
        echo "响应:"
        echo "$RESPONSE" | $JQ_CMD
        STATE=$(echo "$RESPONSE" | grep -o '"state": "[^"]*"' | cut -d'"' -f4)
    fi
    
    # 步骤4: 确认查询
    if [ "$STATE" = "ready_to_query" ]; then
        echo ""
        echo -e "${YELLOW}步骤4: 确认查询${NC}"
        RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/agent/chat" \
            -H "Content-Type: application/json" \
            -d "{\"message\": \"确认查询\", \"session_id\": \"$SESSION_ID\"}")
        
        echo "响应:"
        echo "$RESPONSE" | $JQ_CMD
        
        # 检查是否有价格数据
        if echo "$RESPONSE" | grep -q '"state": "completed"'; then
            print_success "Agent对话流程完成"
            TOTAL_COUNT=$(echo "$RESPONSE" | grep -o '"total_count": [0-9]*' | head -1 | cut -d' ' -f2)
            if [ -n "$TOTAL_COUNT" ]; then
                echo "查询到 $TOTAL_COUNT 条价格数据"
            fi
        fi
    fi
}

# 测试MCP工具
test_mcp() {
    print_title "4. MCP工具测试"
    
    # 4.1 工具列表
    echo ""
    echo -e "${YELLOW}4.1 工具列表${NC}"
    curl -s "$BASE_URL/api/v1/mcp/tools" | $JQ_CMD '.tools | map(.name)'
    
    # 4.2 提取实体
    echo ""
    echo -e "${YELLOW}4.2 提取实体${NC}"
    curl -s -X POST "$BASE_URL/api/v1/mcp/tools/call" \
        -H "Content-Type: application/json" \
        -d '{
            "tool_name": "extract_entities",
            "parameters": {"question": "查一下广东省深圳市HRB400钢筋的价格"}
        }' | $JQ_CMD '.result.data'
    
    # 4.3 识别渠道
    echo ""
    echo -e "${YELLOW}4.3 识别渠道${NC}"
    curl -s -X POST "$BASE_URL/api/v1/mcp/tools/call" \
        -H "Content-Type: application/json" \
        -d '{
            "tool_name": "identify_channel",
            "parameters": {"question": "从厂商报价查钢筋价格"}
        }' | $JQ_CMD '.result.data'
    
    # 4.4 快速搜索
    echo ""
    echo -e "${YELLOW}4.4 快速搜索材料${NC}"
    curl -s -X POST "$BASE_URL/api/v1/mcp/tools/call" \
        -H "Content-Type: application/json" \
        -d '{
            "tool_name": "quick_search_materials",
            "parameters": {"keyword": "混凝土", "limit": 3}
        }' | $JQ_CMD '.result.data.candidates'
    
    print_success "MCP工具测试完成"
}

# 测试标准价格查询
test_price() {
    print_title "5. 标准价格查询测试 (流式)"
    
    echo ""
    echo -e "${YELLOW}查询: 广东省深圳市钢筋价格${NC}"
    echo "响应: (流式输出)"
    
    curl -s -X POST "$BASE_URL/api/v1/query/price" \
        -H "Content-Type: application/json" \
        -d '{
            "question": "从信息价查广东省深圳市钢筋的价格"
        }'
    
    echo ""
    print_success "价格查询完成"
}

# 测试知识问答
test_qa() {
    print_title "6. 知识问答测试 (流式)"
    
    echo ""
    echo -e "${YELLOW}问题: 什么是工程造价${NC}"
    echo "响应: (流式输出)"
    
    curl -s -X POST "$BASE_URL/api/v1/query/stream" \
        -H "Content-Type: application/json" \
        -d '{
            "question": "什么是工程造价",
            "num_docs": 3
        }'
    
    echo ""
    print_success "知识问答完成"
}

# 测试文件列表
test_files() {
    print_title "7. 文件列表测试"
    
    curl -s "$BASE_URL/api/v1/files" | $JQ_CMD '{total_files, files: .files | map(.filename)}'
    
    print_success "文件列表获取完成"
}

# 性能测试
test_performance() {
    print_title "8. 性能测试"
    
    echo ""
    echo "测试1: Agent首次查询"
    time (curl -s -X POST "$BASE_URL/api/v1/agent/chat" \
        -H "Content-Type: application/json" \
        -d '{"message": "查一下钢筋的价格"}' > /dev/null)
    
    echo ""
    echo "测试2: MCP工具调用"
    time (curl -s -X POST "$BASE_URL/api/v1/mcp/tools/call" \
        -H "Content-Type: application/json" \
        -d '{"tool_name": "extract_entities", "parameters": {"question": "查钢筋价格"}}' > /dev/null)
    
    echo ""
    echo "测试3: 服务健康检查"
    time (curl -s "$BASE_URL/" > /dev/null)
}

# 运行所有测试
run_all_tests() {
    test_health
    test_service_status
    test_agent
    test_mcp
    test_price
    test_qa
    test_files
    
    echo ""
    print_title "所有测试完成！"
}

# 帮助信息
show_help() {
    echo "Easy-RAG API 测试脚本"
    echo ""
    echo "用法: ./test_api.sh [command]"
    echo ""
    echo "命令:"
    echo "  health      测试服务健康状态"
    echo "  status      测试服务状态"
    echo "  agent       测试Agent对话完整流程"
    echo "  mcp         测试MCP工具"
    echo "  price       测试标准价格查询(流式)"
    echo "  qa          测试知识问答(流式)"
    echo "  files       测试文件列表"
    echo "  perf        性能测试"
    echo "  all         运行所有测试"
    echo "  help        显示帮助信息"
    echo ""
    echo "示例:"
    echo "  ./test_api.sh health    # 快速检查服务是否正常"
    echo "  ./test_api.sh agent     # 测试Agent对话功能"
    echo "  ./test_api.sh all       # 运行完整测试套件"
}

# 主函数
main() {
    case "${1:-all}" in
        health)
            test_health
            ;;
        status)
            test_service_status
            ;;
        agent)
            test_agent
            ;;
        mcp)
            test_mcp
            ;;
        price)
            test_price
            ;;
        qa)
            test_qa
            ;;
        files)
            test_files
            ;;
        perf|performance)
            test_performance
            ;;
        all)
            run_all_tests
            ;;
        help|-h|--help)
            show_help
            ;;
        *)
            echo "未知命令: $1"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
