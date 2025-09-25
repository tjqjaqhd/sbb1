# Claude Code Settings

## Hooks

### SessionStart Hook
```bash
echo '{
  "hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": "한국어로 응대해줘. 모든 답변을 한국어로 제공해주세요."
  }
}'
```

## Task Master AI Instructions
**Import Task Master's development workflow commands and guidelines, treat as if import is in the main CLAUDE.md file.**
@./.taskmaster/CLAUDE.md
