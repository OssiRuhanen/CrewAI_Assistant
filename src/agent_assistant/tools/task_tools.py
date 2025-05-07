from typing import Type, Optional
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from agent_assistant.task_manager import TaskManager
from agent_assistant.config import KNOWLEDGE_DIR
import os
import json
from datetime import datetime
import re

TASKS_FILE = os.path.join(KNOWLEDGE_DIR, "tasks.json")

def get_task_manager(storage_file=None):
    """Get task manager instance. If storage_file is provided, use that instead of default."""
    return TaskManager(storage_file=storage_file or TASKS_FILE)

class ConversationInput(BaseModel):
    count: int = 100

class ConversationTool(BaseTool):
    name: str = "conversation_tool"
    description: str = (
        "Käytä tätä työkalua lukemaan keskusteluhistoriaa. "
        "Voit määrittää kuinka monta viimeisintä viestiä haluat lukea (count). "
        "Suositeltu määrä on vähintään 100 viestiä kattavan keskusteluhistorian saamiseksi."
    )
    args_schema: Type[BaseModel] = ConversationInput
    conversation_file: str = None

    def __init__(self, knowledge_dir: str):
        super().__init__()
        self.conversation_file = os.path.join(knowledge_dir, "conversation_history.txt")
    
    def _run(self, count: int = 100) -> str:
        """Read recent conversations from the history file."""
        if not os.path.exists(self.conversation_file):
            return "Keskusteluhistoriaa ei löydy."
            
        conversations = []
        with open(self.conversation_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Skip header comments
        for line in lines:
            if line.startswith('#'):
                continue
            if line.strip():
                conversations.append(line.strip())
                
        # Return most recent conversations, ensure we get enough context
        recent = conversations[-max(count, 100):]  # Always get at least 100 messages
        return "\n".join(recent)

class AddTaskInput(BaseModel):
    description: str
    time_str: str = None
    repeat: str = None

class AddTaskTool(BaseTool):
    name: str = "add_task"
    description: str = (
        "Käytä tätä työkalua, kun käyttäjä pyytää muistutusta, hälytystä tai ilmoitusta, "
        "tai kun keskustelusta voidaan päätellä, että tehtävä tulisi lisätä. "
        "Anna tehtävän kuvaus (description), aika (time_str, jos tiedossa, muuten jätä tyhjäksi), ja toistuvuus (repeat, esim. 'daily', jos tarpeen)."
    )
    args_schema: Type[BaseModel] = AddTaskInput
    storage_file: Optional[str] = Field(default=None, description="Path to the tasks storage file")

    def _run(self, description: str, time_str: str = None, repeat: str = None) -> str:
        try:
            tm = get_task_manager(self.storage_file)
            # Ensure description is properly encoded
            description = description.encode('utf-8').decode('utf-8')
            
            # If no time is given, try to extract time from the description
            if not time_str:
                # Look for time patterns in the description
                time_patterns = [
                    r'kello\s+(\d{1,2}[.:]\d{2})',  # "kello 16:45"
                    r'(\d{1,2}[.:]\d{2})',          # "16:45"
                    r'(\d{1,2})\s+(\d{2})',         # "16 45"
                ]
                
                for pattern in time_patterns:
                    match = re.search(pattern, description)
                    if match:
                        if len(match.groups()) == 2:
                            # Handle space-separated time format
                            hour = match.group(1)
                            minute = match.group(2)
                            time_str = f"{hour}:{minute}"
                        else:
                            time_str = match.group(1)
                        break
            
            # Add task directly using task manager
            success, message = tm.add_task(description, time_str, repeat)
            return message
        except Exception as e:
            return f"Virhe tehtävän lisäämisessä: {str(e)}" 