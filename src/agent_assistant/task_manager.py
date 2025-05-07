import json
import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

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

    def add_task(self, description: str, time_str: str, repeat: Optional[str] = None) -> Tuple[bool, str]:
        """Add a new task with description and time. Returns (success, message)."""
        try:
            # Check for duplicate tasks
            for existing_task in self.tasks:
                if (existing_task["description"] == description and 
                    existing_task["status"] == "pending" and
                    existing_task.get("time", "").split()[1] == time_str):
                    return False, f"Tehtävä '{description}' kello {time_str} on jo olemassa."

            # Parse the time string
            task_time = self._parse_time(time_str)
            if not task_time:
                return False, "Virheellinen aika."
            
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
            return True, f"Tehtävä lisätty: {description} ({time_str})"
        except Exception as e:
            return False, f"Virhe tehtävän lisäämisessä: {str(e)}"

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

    def mark_task_done(self, description: str) -> Tuple[bool, str]:
        """Mark a task as done by its description. Returns (success, message)."""
        for task in self.tasks:
            if task["description"] == description and task["status"] == "pending":
                if task.get("repeat") == "daily":
                    # Päivitä aika seuraavalle päivälle
                    task_time = datetime.strptime(task["time"], "%Y-%m-%d %H:%M")
                    next_time = task_time + timedelta(days=1)
                    task["time"] = next_time.strftime("%Y-%m-%d %H:%M")
                    # status pysyy 'pending'
                    self._save_tasks()
                    return True, f"Tehtävä '{description}' päivitetty seuraavalle päivälle."
                else:
                    task["status"] = "done"
                    self._save_tasks()
                    return True, f"Tehtävä '{description}' merkitty tehdyksi."
        return False, f"Tehtävää '{description}' ei löydy tai se on jo tehty."

    def get_all_pending_tasks(self) -> List[Dict]:
        """Get all pending tasks."""
        return [task for task in self.tasks if task["status"] == "pending"]

    def add_task_if_keyword(self, description: str, time_str: str = None, repeat: Optional[str] = None) -> Tuple[bool, str]:
        """Add a new task. Use this tool when a reminder, alert, or notification should be added as a task."""
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
                        time_str = f"{match.group(1)}:{match.group(2)}"
                    else:
                        time_str = match.group(1)
                    break
            
            # If still no time found, use current time + 1 hour as default
            if not time_str:
                now = datetime.now()
                default_time = (now + timedelta(hours=1)).strftime("%H:%M")
                time_str = default_time
                
        return self.add_task(description, time_str, repeat) 