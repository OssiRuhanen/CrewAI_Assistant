import unittest
from datetime import datetime
from agent_assistant.task_manager import TaskManager
from agent_assistant.tools.task_tools import AddTaskTool, ConversationTool
from agent_assistant.tools.memory_tools import MemoryTool
import os
import tempfile
import json

class TestAgentFunctionality(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.tasks_file = os.path.join(self.test_dir, "tasks.json")
        self.conversation_file = os.path.join(self.test_dir, "conversation_history.txt")
        
        # Initialize task manager with test file
        self.task_manager = TaskManager(storage_file=self.tasks_file)
        
        # Initialize tools
        self.task_tool = AddTaskTool(storage_file=self.tasks_file)
        self.conversation_tool = ConversationTool(self.test_dir)
        self.memory_tool = MemoryTool()

    def tearDown(self):
        # Clean up test files
        if os.path.exists(self.tasks_file):
            os.remove(self.tasks_file)
        if os.path.exists(self.conversation_file):
            os.remove(self.conversation_file)
        os.rmdir(self.test_dir)

    def clear_tasks(self):
        """Helper method to clear all tasks."""
        # Clear tasks list
        self.task_manager.tasks = []
        # Save empty list to file
        with open(self.tasks_file, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        # Reload task manager to ensure it's using the cleared list
        self.task_manager = TaskManager(storage_file=self.tasks_file)
        # Update task tool to use the same file
        self.task_tool = AddTaskTool(storage_file=self.tasks_file)

    def test_add_task_duplicate(self):
        """Test that duplicate tasks are not added."""
        self.clear_tasks()
        
        # Add first task
        success, _ = self.task_manager.add_task("Test task", "10:00")
        self.assertTrue(success, "First task should be added successfully")
        
        # Try to add the same task again
        success, message = self.task_manager.add_task("Test task", "10:00")
        self.assertFalse(success, "Duplicate task should not be added")
        self.assertIn("on jo olemassa", message, "Error message should indicate duplicate task")
        
        # Verify only one task exists
        tasks = self.task_manager.get_all_pending_tasks()
        self.assertEqual(len(tasks), 1, "Should only have one task")

    def test_add_task_different_times(self):
        """Test that same task can be added with different times."""
        self.clear_tasks()
        
        # Add task with first time
        success, _ = self.task_manager.add_task("Test task", "10:00")
        self.assertTrue(success, "First task should be added successfully")
        
        # Add same task with different time
        success, message = self.task_manager.add_task("Test task", "11:00")
        self.assertTrue(success, "Task with different time should be added")
        self.assertIn("lisätty", message, "Success message should indicate task was added")
        
        # Verify both tasks exist
        tasks = self.task_manager.get_all_pending_tasks()
        self.assertEqual(len(tasks), 2, "Should have two tasks")

    def test_task_tool_extract_time(self):
        """Test that task tool correctly extracts time from description."""
        self.clear_tasks()
        
        # Test various time formats
        test_cases = [
            ("Punttisali kello 16:45", "16:45"),
            ("Ruokailu 18:00", "18:00"),
            ("Iltatoimet 22 00", "22:00"),
        ]
        
        for description, expected_time in test_cases:
            result = self.task_tool._run(description=description)
            self.assertIn(expected_time, result, f"Time {expected_time} not found in result")
            # Clear tasks after each test case
            self.clear_tasks()

    def test_memory_tool_categories(self):
        """Test memory tool with different categories."""
        test_cases = [
            ("Test reminder", "reminder"),
            ("Test idea", "idea"),
            ("Test preference", "preference"),
        ]
        
        for content, category in test_cases:
            result = self.memory_tool._run(content=content, category=category)
            self.assertIn(content, result, f"Content not found in result for category {category}")

if __name__ == '__main__':
    unittest.main() 