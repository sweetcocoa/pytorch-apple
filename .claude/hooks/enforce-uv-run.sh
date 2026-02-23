#!/bin/bash
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

# Extract the first token (the actual executable being called)
FIRST_TOKEN=$(echo "$COMMAND" | awk '{print $1}')

# Block raw python/python3 — must use "uv run python" instead
if [[ "$FIRST_TOKEN" == "python" || "$FIRST_TOKEN" == "python3" ]]; then
  echo "BLOCKED: raw '$FIRST_TOKEN' is not allowed. Use 'uv run $COMMAND' instead."
  echo "This ensures the correct virtualenv and dependencies are used."
  exit 2
fi

exit 0
