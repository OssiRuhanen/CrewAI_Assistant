import os
import datetime
from pathlib import Path

class MemoryManager:
    """
    Manages the agent's memory system, including conversation history,
    important memories, and user ideas.
    """
    
    def __init__(self, knowledge_dir="knowledge"):
        """
        Initialize the memory manager with paths to memory files.
        
        Args:
            knowledge_dir (str): Directory containing knowledge files
        """
        self.knowledge_dir = knowledge_dir
        self.memories_file = os.path.join(knowledge_dir, "memories.txt")
        self.ideas_file = os.path.join(knowledge_dir, "ideas.txt")
        self.conversation_file = os.path.join(knowledge_dir, "conversation_history.txt")
        
        # Ensure the knowledge directory exists
        os.makedirs(knowledge_dir, exist_ok=True)
        
        # Create files if they don't exist
        self._ensure_files_exist()
    
    def _ensure_files_exist(self):
        """Ensure all memory files exist, create them if they don't."""
        for file_path in [self.memories_file, self.ideas_file, self.conversation_file]:
            if not os.path.exists(file_path):
                with open(file_path, 'w', encoding='utf-8') as f:
                    if file_path == self.memories_file:
                        f.write("# Conversation Memories\n# This file stores important information from conversations with the user.\n# The agent will add new memories here as it learns from interactions.\n\n# Format: [Date] - [Topic] - [Key Information]\n")
                    elif file_path == self.ideas_file:
                        f.write("# User Ideas\n# This file stores ideas and concepts shared by the user.\n# Add new ideas with a timestamp and brief description.\n\n# Format: [Date] - [Idea Title] - [Description]\n")
                    elif file_path == self.conversation_file:
                        f.write("# Conversation History\n# This file stores a log of conversations with the user.\n# Each conversation is timestamped and includes both user and assistant messages.\n\n# Format: [Date Time] - [Speaker] - [Message]\n")
    
    def add_memory(self, topic, information):
        """
        Add a new memory to the memories file.
        
        Args:
            topic (str): The topic or category of the memory
            information (str): The information to remember
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        memory_entry = f"[{timestamp}] - {topic} - {information}\n"
        
        with open(self.memories_file, 'a', encoding='utf-8') as f:
            f.write(memory_entry)
    
    def add_idea(self, title, description):
        """
        Add a new idea to the ideas file.
        
        Args:
            title (str): The title of the idea
            description (str): Description of the idea
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        idea_entry = f"[{timestamp}] - {title} - {description}\n"
        
        with open(self.ideas_file, 'a', encoding='utf-8') as f:
            f.write(idea_entry)
    
    def log_conversation(self, speaker, message):
        """
        Log a conversation entry to the conversation history file.
        
        Args:
            speaker (str): Who is speaking (User or Assistant)
            message (str): The message content
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        conversation_entry = f"[{timestamp}] - {speaker} - {message}\n"
        
        with open(self.conversation_file, 'a', encoding='utf-8') as f:
            f.write(conversation_entry)
    
    def get_recent_memories(self, count=5):
        """
        Get the most recent memories.
        
        Args:
            count (int): Number of memories to retrieve
            
        Returns:
            list: List of recent memories
        """
        memories = []
        with open(self.memories_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Skip header comments
            for line in lines:
                if line.startswith('#'):
                    continue
                if line.strip():
                    memories.append(line.strip())
        
        # Return the most recent memories
        return memories[-count:] if memories else []
    
    def get_recent_ideas(self, count=5):
        """
        Get the most recent ideas.
        
        Args:
            count (int): Number of ideas to retrieve
            
        Returns:
            list: List of recent ideas
        """
        ideas = []
        with open(self.ideas_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Skip header comments
            for line in lines:
                if line.startswith('#'):
                    continue
                if line.strip():
                    ideas.append(line.strip())
        
        # Return the most recent ideas
        return ideas[-count:] if ideas else []
    
    def get_recent_conversations(self, count=5):
        """
        Get the most recent conversation entries.
        
        Args:
            count (int): Number of conversation entries to retrieve
            
        Returns:
            list: List of recent conversation entries
        """
        conversations = []
        with open(self.conversation_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Skip header comments
            for line in lines:
                if line.startswith('#'):
                    continue
                if line.strip():
                    conversations.append(line.strip())
        
        # Return the most recent conversations
        return conversations[-count:] if conversations else []
    
    def extract_memory_from_conversation(self, user_message, assistant_response):
        """
        Analyze a conversation and extract important information to remember.
        This is a simple implementation that could be enhanced with NLP.
        
        Args:
            user_message (str): The user's message
            assistant_response (str): The assistant's response
        """
        # Simple heuristic: if the user asks about preferences or the assistant
        # provides information about the user, it might be worth remembering
        if "prefer" in user_message.lower() or "like" in user_message.lower():
            self.add_memory("User Preferences", f"User mentioned: {user_message}")
        
        # If the user shares an idea
        if "idea" in user_message.lower() or "concept" in user_message.lower():
            self.add_idea("User Idea", user_message) 