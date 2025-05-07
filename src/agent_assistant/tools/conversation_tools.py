from agent_assistant.memory import MemoryManager
from agent_assistant.config import KNOWLEDGE_DIR

def get_new_conversation_entries():
    """Return all new conversation lines and the last processed line number."""
    mm = MemoryManager(knowledge_dir=KNOWLEDGE_DIR)
    lines, last_line = mm.get_unprocessed_conversation_lines()
    return lines, last_line

def mark_conversation_entries_processed(up_to_line):
    """Mark conversation lines up to up_to_line as processed."""
    mm = MemoryManager(knowledge_dir=KNOWLEDGE_DIR)
    mm.mark_conversation_lines_processed(up_to_line) 