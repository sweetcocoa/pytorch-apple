#!/bin/bash
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

# Skip non-Python files
if [[ ! "$FILE_PATH" =~ \.py$ ]]; then
  exit 0
fi

SKILL_FILE="$CLAUDE_PROJECT_DIR/.claude/skills/karpathy-guidelines/SKILL.md"
if [[ ! -f "$SKILL_FILE" ]]; then
  exit 0
fi

cat "$SKILL_FILE"

exit 0
