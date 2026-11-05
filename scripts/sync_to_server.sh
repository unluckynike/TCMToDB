#!/bin/bash
# 本机 → 服务器 同步脚本（代码+数据全量，~20MB）
# 首次使用：把 SERVER 改成你的地址，然后 chmod +x scripts/sync_to_server.sh
# 之后每次我在本机更新文件，你跑一下这个脚本即可
set -e
SERVER="zhl@服务器IP"          # ← 改这里
DEST="/home/zhl/codeWorkSpace/tcmtox"

rsync -avz --exclude '.git' --exclude '__pycache__' \
  /Users/hailin/Project/TCMToDB/ "$SERVER:$DEST/"

echo "✓ 已同步到 $SERVER:$DEST"
echo "服务器端下一步：cd ~/codeWorkSpace/tcmtox && git status 不存在，直接用即可"
