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
# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.

## Task Master AI Instructions
**Import Task Master's development workflow commands and guidelines, treat as if import is in the main CLAUDE.md file.**
@./.taskmaster/CLAUDE.md
