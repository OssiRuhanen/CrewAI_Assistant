import os
import datetime
from pathlib import Path
import traceback
from typing import List, Optional

class MemoryManager:
    """
    Manages the agent's memory system, including conversation history,
    important memories, and user ideas.
    """
    
    def __init__(self, knowledge_dir: str):
        """
        Initialize the memory manager with paths to memory files.
        
        Args:
            knowledge_dir (str): Directory containing knowledge files
        """
        if not isinstance(knowledge_dir, str):
            raise TypeError("knowledge_dir must be a string")
            
        if not os.path.isabs(knowledge_dir):
            import warnings
            warnings.warn(f"[MemoryManager] knowledge_dir is not absolute: {knowledge_dir}. This may cause files to be created in the wrong location.")
            
        self.knowledge_dir = knowledge_dir
        self.memories_file = os.path.join(knowledge_dir, "memories.txt")
        self.ideas_file = os.path.join(knowledge_dir, "ideas.txt")
        self.conversation_file = os.path.join(knowledge_dir, "conversation_history.txt")
        
        # Ensure the knowledge directory exists
        try:
            os.makedirs(knowledge_dir, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create knowledge directory: {e}")
        
        # Create files if they don't exist
        self._ensure_files_exist()
    
    def _ensure_files_exist(self) -> None:
        """Ensure all memory files exist, create them if they don't."""
        file_contents = {
            self.memories_file: "# Conversation Memories\n# This file stores important information from conversations with the user.\n# The agent will add new memories here as it learns from interactions.\n\n# Format: [Date] - [Topic] - [Key Information]\n",
            self.ideas_file: "# User Ideas\n# This file stores ideas and concepts shared by the user.\n# Add new ideas with a timestamp and brief description.\n\n# Format: [Date] - [Idea Title] - [Description]\n",
            self.conversation_file: "# Conversation History\n# This file stores a log of conversations with the user.\n# Each conversation is timestamped and includes both user and assistant messages.\n\n# Format: [Date Time] - [Speaker] - [Message]\n"
        }
        
        for file_path, content in file_contents.items():
            try:
                if not os.path.exists(file_path):
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
            except Exception as e:
                raise RuntimeError(f"Failed to create file {file_path}: {e}")
    
    def add_memory(self, topic: str, information: str) -> None:
        """
        Add a new memory to the memories file.
        
        Args:
            topic (str): The topic or category of the memory
            information (str): The information to remember
            
        Raises:
            TypeError: If topic or information is not a string
            RuntimeError: If file operation fails
        """
        if not isinstance(topic, str) or not isinstance(information, str):
            raise TypeError("topic and information must be strings")
            
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        memory_entry = f"[{timestamp}] - {topic} - {information}\n"
        
        try:
            with open(self.memories_file, 'a', encoding='utf-8') as f:
                f.write(memory_entry)
        except Exception as e:
            raise RuntimeError(f"Failed to write to memories file: {e}")
    
    def add_idea(self, title: str, description: str) -> None:
        """
        Add a new idea to the ideas file.
        
        Args:
            title (str): The title of the idea
            description (str): Description of the idea
            
        Raises:
            TypeError: If title or description is not a string
            RuntimeError: If file operation fails
        """
        if not isinstance(title, str) or not isinstance(description, str):
            raise TypeError("title and description must be strings")
            
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        idea_entry = f"[{timestamp}] - {title} - {description}\n"
        
        try:
            with open(self.ideas_file, 'a', encoding='utf-8') as f:
                f.write(idea_entry)
        except Exception as e:
            raise RuntimeError(f"Failed to write to ideas file: {e}")
    
    def log_conversation(self, speaker: str, message: str) -> None:
        """
        Log a conversation entry to the conversation history file.
        
        Args:
            speaker (str): Who is speaking (User or Assistant)
            message (str): The message content
            
        Raises:
            TypeError: If speaker or message is not a string
            RuntimeError: If file operation fails
        """
        if not isinstance(speaker, str) or not isinstance(message, str):
            raise TypeError("speaker and message must be strings")
            
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        conversation_entry = f"[{timestamp}] - {speaker} - {message}\n"
        
        try:
            with open(self.conversation_file, 'a', encoding='utf-8') as f:
                f.write(conversation_entry)
        except Exception as e:
            raise RuntimeError(f"Failed to write to conversation file: {e}")
    
    def get_recent_memories(self, count: int = 5) -> List[str]:
        """
        Get the most recent memories.
        
        Args:
            count (int): Number of memories to retrieve
            
        Returns:
            list: List of recent memories
            
        Raises:
            TypeError: If count is not an integer
            RuntimeError: If file operation fails
        """
        if not isinstance(count, int):
            raise TypeError("count must be an integer")
            
        memories = []
        try:
            with open(self.memories_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # Skip header comments
                for line in lines:
                    if line.startswith('#'):
                        continue
                    if line.strip():
                        memories.append(line.strip())
        except Exception as e:
            raise RuntimeError(f"Failed to read memories file: {e}")
        
        # Return the most recent memories
        return memories[-count:] if memories else []
    
    def get_recent_ideas(self, count: int = 5) -> List[str]:
        """
        Get the most recent ideas.
        
        Args:
            count (int): Number of ideas to retrieve
            
        Returns:
            list: List of recent ideas
            
        Raises:
            TypeError: If count is not an integer
            RuntimeError: If file operation fails
        """
        if not isinstance(count, int):
            raise TypeError("count must be an integer")
            
        ideas = []
        try:
            with open(self.ideas_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # Skip header comments
                for line in lines:
                    if line.startswith('#'):
                        continue
                    if line.strip():
                        ideas.append(line.strip())
        except Exception as e:
            raise RuntimeError(f"Failed to read ideas file: {e}")
        
        # Return the most recent ideas
        return ideas[-count:] if ideas else []
    
    def get_recent_conversations(self, count: int = 5) -> List[str]:
        """
        Get the most recent conversation entries.
        
        Args:
            count (int): Number of conversation entries to retrieve
            
        Returns:
            list: List of recent conversation entries
            
        Raises:
            TypeError: If count is not an integer
            RuntimeError: If file operation fails
        """
        if not isinstance(count, int):
            raise TypeError("count must be an integer")
            
        conversations = []
        try:
            with open(self.conversation_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # Skip header comments
                for line in lines:
                    if line.startswith('#'):
                        continue
                    if line.strip():
                        conversations.append(line.strip())
        except Exception as e:
            raise RuntimeError(f"Failed to read conversation file: {e}")
        
        # Return the most recent conversations
        return conversations[-count:] if conversations else []
    
    def extract_memory_from_conversation(self, user_message: str, assistant_response: str) -> None:
        """
        Analyze a conversation and extract important information to remember.
        This is a simple implementation that could be enhanced with NLP.
        
        Args:
            user_message (str): The user's message
            assistant_response (str): The assistant's response
            
        Raises:
            TypeError: If user_message or assistant_response is not a string
        """
        if not isinstance(user_message, str) or not isinstance(assistant_response, str):
            raise TypeError("user_message and assistant_response must be strings")
            
        # Simple heuristic: if the user asks about preferences or the assistant
        # provides information about the user, it might be worth remembering
        if "prefer" in user_message.lower() or "like" in user_message.lower():
            self.add_memory("User Preferences", f"User mentioned: {user_message}")
        
        # If the user shares an idea
        if "idea" in user_message.lower() or "concept" in user_message.lower():
            self.add_idea("User Idea", user_message)

    def get_unprocessed_conversation_lines(self):
        """Return all new lines from conversation_history.txt since last processed."""
        conversation_path = os.path.join(self.knowledge_dir, "conversation_history.txt")
        last_line = 0
        last_processed_line_file = os.path.join(self.knowledge_dir, "last_processed_line.txt")
        if os.path.exists(last_processed_line_file):
            with open(last_processed_line_file, "r", encoding="utf-8") as f:
                try:
                    last_line = int(f.read().strip())
                except Exception:
                    last_line = 0
        lines = []
        with open(conversation_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if i > last_line:
                    lines.append(line.rstrip())
        return lines, last_line

    def mark_conversation_lines_processed(self, up_to_line):
        last_processed_line_file = os.path.join(self.knowledge_dir, "last_processed_line.txt")
        with open(last_processed_line_file, "w", encoding="utf-8") as f:
            f.write(str(up_to_line)) 