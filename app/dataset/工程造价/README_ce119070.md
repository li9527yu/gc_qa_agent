# Wild Horse Pose Game
![Wild Horse Pose Game Poster](./poster.jpg)
## 项目介绍 | Introduction
`Wild Horse Pose Game` 是一个基于人体姿态识别的第一人称驯马游戏原型，也是 `抖音 AI+游戏马拉松苏州站` 的参赛作品。在整个比赛中我和队友两个人使用Codex 的AI工具完成了全部的实现。玩家通过摄像头输入和身体动作与野马互动，结合姿态控制、即时反馈和大模型生成内容，体验 AI + 游戏结合的玩法探索。
`Wild Horse Pose Game` is a first-person horse-taming game prototype built around human pose recognition. It was created as an entry for the `Douyin AI + Game Marathon Suzhou` event. Players interact with wild horses through camera input and body movements, combining pose control, real-time feedback, and LLM-powered content generation in an AI game experiment.
## 使用前说明 | Before You Start
运行本项目前，你需要配置你自己的大模型 API Key。当前项目服务端默认接入 `OpenRouter`，请在本地创建并填写 `local-secrets.json`，不要将真实密钥提交到公开仓库。
Before running this project, you need to configure your own LLM API key. The current server integration uses `OpenRouter` by default. Create and fill in `local-secrets.json` locally, and do not commit real secrets to a public repository.
示例配置 | Example:
```json
{
"openRouterApiKey": "your-api-key",
"openRouterBaseUrl": "https://openrouter.ai/api/v1",
"openRouterModel": "deepseek/deepseek-v3.2"
}
```
也可以通过环境变量覆盖这些配置：
You can also override these settings with environment variables:
也可以通过环境变量覆盖这些配置：
You can also override these settings with environment variables:
- `OPENROUTER\_API\_KEY`
- `OPENROUTER\_BASE\_URL`
- `OPENROUTER\_MODEL`
## 功能特点 | Features
- 第一人称驯马玩法，强调动作反馈和沉浸感
- 基于 `MediaPipe Pose Landmarker` 的姿态交互
- 结合大模型生成的马匹行为或旁白内容
- 包含颜色、稀有度、性格差异的马匹收集系统
- 仓库中保留了调研、设计和素材生成流程
- First-person horse-taming gameplay with strong motion feedback
- Pose-based interaction powered by `MediaPipe Pose Landmarker`
- AI-driven horse behavior and narration
- Horse collection system with color, rarity, and temperament variation
- Research, design, and asset generation pipeline archived in the repo
## 技术栈 | Tech Stack
- 前端：`HTML`、`CSS`、`Vanilla JavaScript`
- 本地服务：`Node.js`
- 姿态识别：`MediaPipe Pose Landmarker`
- 大模型接入：`OpenRouter`
- 美术生成：`ByteDance Ark` 图像生成脚本
- Frontend: `HTML`, `CSS`, `Vanilla JavaScript`
- Local server: `Node.js`
- Pose detection: `MediaPipe Pose Landmarker`
- LLM integration: `OpenRouter`
- Asset generation: `ByteDance Ark` image generation scripts
## 仓库结构 | Repository Structure
仓库主要分为两个部分：
The repository can be understood as two main areas:
### 主游戏原型 | Main Game
以下文件构成当前可运行的游戏原型：
These files make up the current playable prototype:
- `index.html`: 页面结构 | page structure
- `style.css`: UI、HUD 与动画样式 | UI, HUD, and animation styling
- `app.js`: 核心玩法、姿态输入、战斗流程与收集逻辑 | core gameplay logic, pose input handling, battle flow, and collection system
- `server.js`: 本地静态服务与大模型请求处理 | local static server and LLM-related request handling
- `assets/`: 游戏图片、音频和马匹数据 | in-game images, audio, and horse catalog data
- `models/pose\_landmarker\_full.task`: MediaPipe 使用的姿态模型 | pose model used by MediaPipe
- `models/pose\_landmarker\_full.task`: MediaPipe 使用的姿态模型 | pose model used by MediaPipe
- `package.json`: 本地运行脚本 | local run scripts
### 调研与素材流水线 | Research And Asset Pipeline
[`black/`](./black) 目录保存了项目早期探索、文档与素材生成产物。
The [`black/`](./black) directory stores early exploration material, documentation, and asset-generation outputs.
- 产品文档：PRD 演进、技术方案、任务拆解
- 素材规划：提示词、风格说明、命名与检查清单
- 素材生成代码：马匹、背景、骑手状态等相关脚本与输出
- 产品文档：PRD 演进、技术方案、任务拆解
- 素材规划：提示词、风格说明、命名与检查清单
- 素材生成代码：马匹、背景、骑手状态等相关脚本与输出
- Product docs: PRD evolution, technical plans, and task breakdowns
- Asset planning: prompt libraries, art direction notes, naming, and checklist docs
- Asset generation code: scripts and outputs for horses, battle backgrounds, and rider states
简而言之：
In short:
- `main game`: 可游玩的原型 | playable prototype
- `black/`: 调研归档 + 素材流水线 | research archive + asset pipeline
## 本地运行 | Local Run
环境要求：
Requirements:
## 本地运行 | Local Run
环境要求：
Requirements:
- `Node.js 18+`
- 支持摄像头权限的桌面浏览器 | A desktop browser with camera access
- 仓库中存在 `models/pose\_landmarker\_full.task` | `models/pose\_landmarker\_full.task` present in the repo
启动本地服务：
Start the local server:
```bash
npm start
```
或者：
Or:
```bash
node server.js
```
打开：
Open:
```text
http://localhost:8080
```
## 本地密钥配置 | Local Secrets
请不要把真实 API Key 提交到公开仓库。项目当前默认读取根目录下的 `local-secrets.json`。
## 本地密钥配置 | Local Secrets
请不要把真实 API Key 提交到公开仓库。项目当前默认读取根目录下的 `local-secrets.json`。
Do not commit real API keys to a public repository. The project currently reads `local-secrets.json` from the repository root.
如果你需要启用大模型能力，请确保至少配置：
If you want to enable LLM features, make sure at least the following field is configured:
- `openRouterApiKey`
## 已实现内容 | What Is Already Implemented
基于当前代码与文档，项目已经包含：
Based on the current code and documents, the project already includes:
Based on the current code and documents, the project already includes:
- 浏览器内可运行的玩法原型 | Browser-based gameplay prototype
- 摄像头输入与姿态识别 | Camera input and pose landmark recognition
- 左拉、右拉、保持、下压等动作映射 | Left pull, right pull, hold, and press-down style action mapping
- 马匹稀有度、颜色、性格与收集数据 | Horse rarity, color, temperament, and collection data
- 音效反馈、HUD 与结算流程 | Audio feedback, HUD, and result flow
- 通过本地服务接入的大模型辅助叙事或行为输出 | LLM-assisted narration or behavior output through the local server
## 发布建议 | Publishing Notes
如果你计划将仓库公开，建议优先检查以下内容：
If you plan to publish this repository publicly, review the following first:
- 删除代码或本地配置中的真实 API Key | Remove any real API keys from code or local config files
- 确认 `local-secrets.json` 已被 `.gitignore` 排除 | Confirm `local-secrets.json` is ignored by `.gitignore`
- 检查 `black/` 中的生成产物是否都需要保留 | Check whether generated outputs under `black/` are worth keeping
- 补充真实截图、GIF 或演示素材 | Add real screenshots, GIFs, or demo materials
## 后续改进 | Next Improvements
- 将 `app.js` 拆分为更小的模块 | Split `app.js` into smaller modules
- 将玩法常量迁移到独立配置文件 | Move gameplay constants into dedicated config files
- 增加截图、GIF 和玩法流程图 | Add screenshots, GIFs, and a gameplay flow diagram
- 继续完善 `.gitignore` 与本地配置说明 | Refine `.gitignore` and local config documentation