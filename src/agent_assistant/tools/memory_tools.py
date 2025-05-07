from typing import Type
from pydantic import BaseModel
from crewai.tools import BaseTool
from agent_assistant.config import KNOWLEDGE_DIR
import os
import json
from datetime import datetime

class MemoryInput(BaseModel):
    content: str
    category: str = "general"

class MemoryTool(BaseTool):
    name: str = "memory_tool"
    description: str = (
        "Käytä tätä työkalua tallentamaan tärkeitä tietoja keskustelusta muistiin. "
        "Voit tallentaa muistioita, ideoita, mieltymyksiä ja muita tärkeitä tietoja. "
        "Anna sisältö (content) ja kategoria (category, esim. 'idea', 'preference', 'note')."
    )
    args_schema: Type[BaseModel] = MemoryInput

    def _run(self, content: str, category: str = "general") -> str:
        try:
            memory_file = os.path.join(KNOWLEDGE_DIR, "memories.json")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(memory_file), exist_ok=True)
            
            # Load existing memories or create new list
            if os.path.exists(memory_file):
                with open(memory_file, 'r', encoding='utf-8') as f:
                    memories = json.load(f)
            else:
                memories = []
            
            # Add new memory
            new_memory = {
                "content": content,
                "category": category,
                "timestamp": datetime.now().isoformat()
            }
            memories.append(new_memory)
            
            # Save updated memories
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(memories, f, indent=2, ensure_ascii=False)
            
            return f"Muisti tallennettu: {content}"
            
        except Exception as e:
            return f"Virhe muistin tallentamisessa: {str(e)}"
    
    def get_memories(self, category: str = None, limit: int = 10) -> list:
        """Get memories from storage, optionally filtered by category."""
        try:
            memory_file = os.path.join(KNOWLEDGE_DIR, "memories.json")
            if not os.path.exists(memory_file):
                return []
                
            with open(memory_file, 'r', encoding='utf-8') as f:
                memories = json.load(f)
            
            if category:
                memories = [m for m in memories if m["category"] == category]
            
            return memories[-limit:]  # Return most recent memories
            
        except Exception:
            return [] 