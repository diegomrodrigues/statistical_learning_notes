from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import DirectoryReadTool, FileReadTool, FileWriterTool

@CrewBase
class StudyAgent():
	"""StudyAgent crew for processing and generating educational content"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	def __init__(self):
		super().__init__()
		self.llm = None

	@agent
	def cleanup_agent(self) -> Agent:
		"""Creates a cleanup agent to remove prompt artifacts"""
		return Agent(
			config=self.agents_config['cleanup_agent'],
			tools=[],
			verbose=True,
			allow_delegation=False,
			llm=self.llm
		)

	@agent
	def math_formatter(self) -> Agent:
		"""Creates a formatter agent to handle mathematical notation"""
		return Agent(
			config=self.agents_config['math_formatter'],
			tools=[],
			verbose=True,
			allow_delegation=False,
			llm=self.llm
		)

	@agent
	def diagram_creator(self) -> Agent:
		"""Creates a diagram agent to generate Mermaid visualizations"""
		return Agent(
			config=self.agents_config['diagram_creator'],
			tools=[],
			verbose=True,
			allow_delegation=False,
			llm=self.llm
		)

	@agent
	def examples_generator(self) -> Agent:
		"""Creates an examples agent to add numerical examples"""
		return Agent(
			config=self.agents_config['examples_generator'],
			tools=[],
			verbose=True,
			allow_delegation=False,
			llm=self.llm
		)

	@task
	def cleanup_task(self) -> Task:
		"""Cleans up prompt artifacts from generated content"""
		return Task(
			config=self.tasks_config['cleanup_task'],
			context=[]
		)

	@task
	def generate_examples_task(self) -> Task:
		"""Adds numerical examples to content"""
		return Task(
			config=self.tasks_config['generate_examples_task'],
			context=[self.cleanup_task()]  # Depends on cleaned content
		)

	@task
	def create_diagrams_task(self) -> Task:
		"""Creates Mermaid diagrams for visualization"""
		return Task(
			config=self.tasks_config['create_diagrams_task'],
			context=[self.generate_examples_task()]  # Depends on content with examples
		)

	@task
	def format_math_task(self) -> Task:
		"""Formats mathematical content using LaTeX notation"""
		return Task(
			config=self.tasks_config['format_math_task'],
			context=[self.create_diagrams_task()]  # Depends on content with diagrams
		)

	@crew
	def crew(self, llm=None, verbose=True) -> Crew:
		"""Creates the StudyAgent crew with sequential processing
		
		Args:
			llm: Optional language model to use
			verbose: Whether to enable verbose output (default: True)
		"""
		self.llm = llm
		
		# Create agents with the custom LLM
		agents = [
			self.cleanup_agent(),
			self.examples_generator(),
			self.diagram_creator(),
			self.math_formatter()
		]
		
		# Create tasks with the custom LLM
		tasks = [
			self.cleanup_task(),
			self.generate_examples_task(),
			self.create_diagrams_task(),
			self.format_math_task()
		]

		# Configure each agent and task to use the custom LLM
		for agent in agents:
			agent.llm = self.llm
			agent.verbose = verbose  # Set verbose for each agent
			
		return Crew(
			agents=agents,
			tasks=tasks,
			process=Process.sequential,
			verbose=verbose,  # Use the passed verbose parameter
			manager_llm=self.llm
		)
