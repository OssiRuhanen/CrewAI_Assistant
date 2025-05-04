import json
import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional

class TaskManager:
    def __init__(self, storage_file: str = "tasks.json"):
        self.storage_file = storage_file
        self.tasks = self._load_tasks()

    def _load_tasks(self) -> List[Dict]:
        """Load tasks from storage file."""
        if os.path.exists(self.storage_file):
            with open(self.storage_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _save_tasks(self):
        """Save tasks to storage file."""
        with open(self.storage_file, 'w', encoding='utf-8') as f:
            json.dump(self.tasks, f, ensure_ascii=False, indent=2)

    def _parse_time(self, time_str: str) -> Optional[datetime]:
        """Parse time string into datetime object, handling various formats."""
        # Remove any whitespace
        time_str = time_str.strip()
        
        # Try different time formats
        formats = [
            "%H:%M",    # 16:25
            "%H.%M",    # 16.25
            "%H %M",    # 16 25
            "%H:%M:%S", # 16:25:00
            "%H.%M.%S", # 16.25.00
        ]
        
        for fmt in formats:
            try:
                # Try parsing with current format
                time_obj = datetime.strptime(time_str, fmt)
                return time_obj
            except ValueError:
                continue
        
        # If no format matched, try to extract numbers and create time
        numbers = re.findall(r'\d+', time_str)
        if len(numbers) >= 2:
            try:
                hour = int(numbers[0])
                minute = int(numbers[1])
                if 0 <= hour <= 23 and 0 <= minute <= 59:
                    return datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
            except ValueError:
                pass
        
        return None

    def add_task(self, description: str, time_str: str, repeat: Optional[str] = None) -> bool:
        """Add a new task with description and time."""
        try:
            # Parse the time string
            task_time = self._parse_time(time_str)
            if not task_time:
                return False
            
            now = datetime.now()
            task_time = task_time.replace(year=now.year, month=now.month, day=now.day)
            
            # If the time is in the past, assume it's for tomorrow
            if task_time < now:
                task_time = task_time.replace(day=now.day + 1)

            new_task = {
                "description": description,
                "time": task_time.strftime("%Y-%m-%d %H:%M"),
                "status": "pending"
            }
            if repeat:
                new_task["repeat"] = repeat
            
            self.tasks.append(new_task)
            self._save_tasks()
            return True
        except Exception as e:
            print(f"Virhe ajan jäsentämisessä: {e}")
            return False

    def get_upcoming_tasks(self, minutes_ahead: int = 30) -> List[Dict]:
        """Get tasks that are due within the next X minutes."""
        now = datetime.now()
        upcoming = []
        
        for task in self.tasks:
            task_time = datetime.strptime(task["time"], "%Y-%m-%d %H:%M")
            time_diff = (task_time - now).total_seconds() / 60
            
            if 0 <= time_diff <= minutes_ahead and task["status"] == "pending":
                upcoming.append(task)
        
        return upcoming

    def mark_task_done(self, description: str) -> bool:
        """Mark a task as done by its description."""
        for task in self.tasks:
            if task["description"] == description and task["status"] == "pending":
                if task.get("repeat") == "daily":
                    # Päivitä aika seuraavalle päivälle
                    task_time = datetime.strptime(task["time"], "%Y-%m-%d %H:%M")
                    next_time = task_time + timedelta(days=1)
                    task["time"] = next_time.strftime("%Y-%m-%d %H:%M")
                    # status pysyy 'pending'
                else:
                    task["status"] = "done"
                self._save_tasks()
                return True
        return False

    def get_all_pending_tasks(self) -> List[Dict]:
        """Get all pending tasks."""
        return [task for task in self.tasks if task["status"] == "pending"] 