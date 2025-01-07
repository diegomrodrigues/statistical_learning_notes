from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import DirectoryReadTool, FileReadTool, FileWriterTool

@CrewBase
class StudyAgent():
	"""StudyAgent crew for processing and generating educational content"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	@agent
	def math_formatter(self) -> Agent:
		"""Creates a formatter agent to handle mathematical notation"""
		return Agent(
			config=self.agents_config['math_formatter'],
			tools=[],
			verbose=True,
			allow_delegation=True,
			memory=False
		)

	@agent
	def diagram_creator(self) -> Agent:
		"""Creates a diagram agent to generate Mermaid visualizations"""
		return Agent(
			config=self.agents_config['diagram_creator'],
			tools=[],
			verbose=True,
			allow_delegation=True,
			memory=False
		)

	@agent
	def examples_generator(self) -> Agent:
		"""Creates an examples agent to add numerical examples"""
		return Agent(
			config=self.agents_config['examples_generator'],
			tools=[],
			verbose=True,
			allow_delegation=True,
			memory=False
		)

	@agent
	def content_writer(self) -> Agent:
		"""Creates a writer agent to generate final content"""
		return Agent(
			config=self.agents_config['content_writer'],
			tools=[],
			verbose=True,
			allow_delegation=True,
			memory=False
		)

	@agent
	def cleanup_agent(self) -> Agent:
		"""Creates a cleanup agent to remove prompt artifacts"""
		return Agent(
			config=self.agents_config['cleanup_agent'],
			tools=[],
			verbose=True,
			allow_delegation=True,
			memory=False,
			llm_config={
				"temperature": 0.3,
				"top_p": 0.95,
				"top_k": 40,
				"max_output_tokens": 8192,
			}
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

	@task
	def write_content_task(self) -> Task:
		"""Generates final content with all components"""
		return Task(
			config=self.tasks_config['write_content_task'],
			context=[self.format_math_task()],
			tools=[FileWriterTool()],
			output_file='{section_dir}/{topic_number}. {topic_name}.md'
		)

	@crew
	def crew(self, llm=None) -> Crew:
		"""Creates the StudyAgent crew with sequential processing"""
		return Crew(
			agents=[
				self.document_analyzer(),
				self.cleanup_agent(),
				self.examples_generator(),
				self.diagram_creator(),
				self.math_formatter(),
				self.content_writer()
			],
			tasks=[
				self.analyze_documents_task(),
				self.cleanup_task(),
				self.generate_examples_task(),
				self.create_diagrams_task(), 
				self.format_math_task(),
				self.write_content_task()
			],
			process=Process.sequential,
			verbose=True,
			llm=llm
		)
