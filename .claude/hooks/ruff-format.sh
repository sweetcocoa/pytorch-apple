#!/bin/bash
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

# .py 파일이 아니면 무시
if [[ ! "$FILE_PATH" =~ \.py$ ]]; then
  exit 0
fi

# ruff format 실행
cd "$CLAUDE_PROJECT_DIR"
uv run ruff format "$FILE_PATH" 2>/dev/null

exit 0
