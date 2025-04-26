from crewai.tools import BaseTool
from typing import Type, List, Optional
from pydantic import BaseModel, Field
from agent_assistant.memory import MemoryManager

class MemoryToolInput(BaseModel):
    """Input schema for MemoryTool."""
    action: str = Field(..., description="The action to perform: 'add_memory', 'add_idea', 'get_memories', 'get_ideas', or 'get_conversations'")
    topic: Optional[str] = Field(None, description="Topic or title for the memory or idea")
    content: Optional[str] = Field(None, description="Content of the memory or idea")
    count: Optional[int] = Field(5, description="Number of items to retrieve (for get actions)")

class MemoryTool(BaseTool):
    name: str = "Memory Tool"
    description: str = (
        "A tool for managing and accessing the agent's memory system. "
        "Can add new memories or ideas, and retrieve recent memories, ideas, or conversations. "
        "Use this tool to remember important information from conversations and access past interactions."
    )
    args_schema: Type[BaseModel] = MemoryToolInput
    
    def _run(self, action: str, topic: Optional[str] = None, content: Optional[str] = None, count: int = 5) -> str:
        """
        Run the memory tool with the specified action.
        
        Args:
            action: The action to perform
            topic: Topic or title for the memory or idea
            content: Content of the memory or idea
            count: Number of items to retrieve (for get actions)
            
        Returns:
            str: Result of the action
        """
        # Create a memory manager instance for this run
        memory_manager = MemoryManager()
        
        if action == "add_memory":
            if not topic or not content:
                return "Error: Topic and content are required for adding a memory."
            memory_manager.add_memory(topic, content)
            return f"Added memory: {topic} - {content}"
            
        elif action == "add_idea":
            if not topic or not content:
                return "Error: Title and description are required for adding an idea."
            memory_manager.add_idea(topic, content)
            return f"Added idea: {topic} - {content}"
            
        elif action == "get_memories":
            memories = memory_manager.get_recent_memories(count)
            if not memories:
                return "No memories found."
            
            # Format memories in a TTS-friendly way
            formatted_memories = []
            for memory in memories:
                # Extract date, topic, and content from the memory string
                # Format: [Date] - [Topic] - [Content]
                parts = memory.split(" - ", 2)
                if len(parts) == 3:
                    date, topic, content = parts
                    # Remove brackets from date
                    date = date.replace("[", "").replace("]", "")
                    formatted_memories.append(f"On {date}, about {topic}: {content}")
                else:
                    formatted_memories.append(memory)
            
            return "Here are your memories:\n" + "\n".join(formatted_memories)
            
        elif action == "get_ideas":
            ideas = memory_manager.get_recent_ideas(count)
            if not ideas:
                return "No ideas found."
            
            # Format ideas in a TTS-friendly way
            formatted_ideas = []
            for idea in ideas:
                # Extract date, title, and description from the idea string
                # Format: [Date] - [Title] - [Description]
                parts = idea.split(" - ", 2)
                if len(parts) == 3:
                    date, title, description = parts
                    # Remove brackets from date
                    date = date.replace("[", "").replace("]", "")
                    formatted_ideas.append(f"On {date}, idea: {title}. Description: {description}")
                else:
                    formatted_ideas.append(idea)
            
            return "Here are your ideas:\n" + "\n".join(formatted_ideas)
            
        elif action == "get_conversations":
            conversations = memory_manager.get_recent_conversations(count)
            if not conversations:
                return "No conversations found."
            
            # Format conversations in a TTS-friendly way
            formatted_conversations = []
            for conv in conversations:
                # Extract date, time, speaker, and message from the conversation string
                # Format: [Date Time] - [Speaker] - [Message]
                parts = conv.split(" - ", 2)
                if len(parts) == 3:
                    datetime_str, speaker, message = parts
                    # Remove brackets from datetime
                    datetime_str = datetime_str.replace("[", "").replace("]", "")
                    formatted_conversations.append(f"On {datetime_str}, {speaker} said: {message}")
                else:
                    formatted_conversations.append(conv)
            
            return "Here are your recent conversations:\n" + "\n".join(formatted_conversations)
            
        else:
            return f"Error: Unknown action '{action}'. Valid actions are 'add_memory', 'add_idea', 'get_memories', 'get_ideas', or 'get_conversations'." 