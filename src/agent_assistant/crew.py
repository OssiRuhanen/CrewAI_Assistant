from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.knowledge.knowledge import Knowledge
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from agent_assistant.tools.memory_tool import MemoryTool
import os
from agent_assistant.config import KNOWLEDGE_DIR as DEFAULT_KNOWLEDGE_DIR

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class AgentAssistant():
    """AgentAssistant crew"""

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def assistant(self) -> Agent:
        return Agent(
            config=self.agents_config['assistant'],
            tools=[MemoryTool()],
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def assist_task(self) -> Task:
        return Task(
            config=self.tasks_config['assist_task'],
        )

    @crew
    def crew(self, knowledge_dir: str = None) -> Crew:
        """Creates the AgentAssistant crew"""
        if knowledge_dir is None:
            knowledge_dir = DEFAULT_KNOWLEDGE_DIR
        txt_files = [f for f in os.listdir(knowledge_dir) if f.endswith(".txt")]
        knowledge = Knowledge(
            collection_name="default",
            sources=[TextFileKnowledgeSource(file_paths=txt_files)],
        )
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            knowledge=knowledge
        )
