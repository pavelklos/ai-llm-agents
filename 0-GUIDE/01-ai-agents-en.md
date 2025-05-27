# AI Agents Development Course Guide
## Building Autonomous AI Systems for Complex Task Automation

This comprehensive course covers the development of AI agents capable of autonomous task execution and complex workflow automation using Python and modern AI frameworks.

---

## 01. AI API and First Agent

Learn the fundamentals of Large Language Models (LLMs) and API integration for agent development.

### Key Concepts:
- LLM API providers (OpenAI, Anthropic, Ollama, HuggingFace)
- Model parameters and provider differences
- Tool-calling architecture and agent foundations

```python
import openai
from typing import List, Dict

class SimpleAgent:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        
    def call_model(self, prompt: str, tools: List[Dict] = None):
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
            tool_choice="auto" if tools else None
        )
        return response.choices[0].message

# Example tool definition
def get_weather(location: str) -> str:
    return f"Weather in {location}: 22°C, sunny"

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather information",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }
    }
}]
```

---

## 02. Databases for Agents

Integration of various database types with AI agents for knowledge storage and retrieval.

### Database Types:
- SQL (MS SQL Server)
- NoSQL (MongoDB) 
- Vector databases (Chroma, Elasticsearch)

```python
import chromadb
from pymongo import MongoClient
import pyodbc

class DatabaseAgent:
    def __init__(self):
        # Vector database for embeddings
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection("knowledge_base")
        
        # MongoDB for document storage
        self.mongo_client = MongoClient('mongodb://localhost:27017/')
        self.db = self.mongo_client['agent_db']
        
    def store_knowledge(self, text: str, metadata: dict):
        # Store in vector database
        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[metadata.get('id', 'doc_1')]
        )
        
    def query_knowledge(self, query: str, n_results: int = 5):
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results
```

---

## 03. Model Context Protocol (MCP)

Understanding MCP for interoperability between AI systems and external tools.

### MCP Components:
- Message structure standardization
- Tool and resource definitions
- Cross-platform compatibility

```python
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class MCPMessage:
    id: str
    method: str
    params: Optional[dict] = None
    
@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: dict
    
class MCPHandler:
    def __init__(self):
        self.tools = {}
        
    def register_tool(self, tool: MCPTool, handler_func):
        self.tools[tool.name] = {
            'tool': tool,
            'handler': handler_func
        }
        
    def handle_message(self, message: MCPMessage):
        if message.method == "tools/call":
            tool_name = message.params.get('name')
            if tool_name in self.tools:
                return self.tools[tool_name]['handler'](message.params.get('arguments', {}))
        return {"error": "Unknown method or tool"}
```

---

## 04. Automation and Workflow with n8n

Visual workflow automation and agent orchestration using n8n platform.

### Features:
- Visual process design
- Node-based workflow creation
- Database and API integrations
- LLM integration capabilities

```python
# n8n webhook integration example
import requests
import json

class N8NAgent:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        
    def trigger_workflow(self, data: dict):
        response = requests.post(
            self.webhook_url,
            json=data,
            headers={'Content-Type': 'application/json'}
        )
        return response.json()
        
    def create_ai_workflow(self, prompt: str, tools: list):
        workflow_data = {
            "prompt": prompt,
            "tools": tools,
            "timestamp": "2024-01-01T00:00:00Z"
        }
        return self.trigger_workflow(workflow_data)
```

---

## 05. Custom Agent Framework

Building agents from scratch with different architectural patterns.

### Agent Types:
- Autonomous agents
- Workflow-based agents
- ReAct (Reasoning + Acting) pattern

```python
from enum import Enum
from typing import List, Tuple

class AgentState(Enum):
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    FINISHED = "finished"

class ReActAgent:
    def __init__(self, llm_client, tools: dict):
        self.llm = llm_client
        self.tools = tools
        self.state = AgentState.THINKING
        self.memory = []
        
    def think(self, observation: str) -> str:
        prompt = f"""
        Previous actions: {self.memory}
        Current observation: {observation}
        
        Think step by step about what to do next.
        Available tools: {list(self.tools.keys())}
        """
        response = self.llm.call_model(prompt)
        return response.content
        
    def act(self, thought: str) -> Tuple[str, str]:
        # Determine which tool to use based on thought
        for tool_name, tool_func in self.tools.items():
            if tool_name.lower() in thought.lower():
                result = tool_func()
                self.memory.append(f"Used {tool_name}: {result}")
                return tool_name, result
        return "no_action", "No suitable tool found"
        
    def run(self, initial_prompt: str, max_iterations: int = 10):
        observation = initial_prompt
        
        for i in range(max_iterations):
            if self.state == AgentState.FINISHED:
                break
                
            thought = self.think(observation)
            action, result = self.act(thought)
            observation = result
            
            if "task completed" in result.lower():
                self.state = AgentState.FINISHED
                
        return self.memory
```

---

## 06. LangChain and LangGraph

Advanced agent orchestration using LangChain ecosystem.

### Components:
- Prompt templates and chains
- Retrieval-Augmented Generation (RAG)
- LangServe, LangSmith, LangGraph orchestration

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

class RAGAgent:
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(api_key=api_key)
        self.embeddings = OpenAIEmbeddings(api_key=api_key)
        self.vectorstore = None
        
    def setup_knowledge_base(self, documents: List[str]):
        self.vectorstore = Chroma.from_texts(
            documents, 
            self.embeddings
        )
        
    def create_rag_chain(self):
        prompt = ChatPromptTemplate.from_template("""
        Context: {context}
        Question: {question}
        
        Answer the question based on the context provided.
        """)
        
        retriever = self.vectorstore.as_retriever()
        
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
        )
        return chain
        
    def query(self, question: str):
        chain = self.create_rag_chain()
        return chain.invoke(question)

# LangGraph state management
from langgraph.graph import StateGraph
from typing import TypedDict

class AgentState(TypedDict):
    messages: List[str]
    current_task: str
    completed: bool

def create_langgraph_agent():
    workflow = StateGraph(AgentState)
    
    def process_task(state: AgentState):
        # Process the current task
        state["completed"] = True
        return state
        
    workflow.add_node("process", process_task)
    workflow.set_entry_point("process")
    workflow.set_finish_point("process")
    
    return workflow.compile()
```

---

## 07. Semantic Kernel and Autogen

Multi-agent systems and advanced agent collaboration frameworks.

### Features:
- Agent planning and memory systems
- Inter-agent communication
- Autogen Studio integration

```python
# Semantic Kernel example
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

class SemanticKernelAgent:
    def __init__(self, api_key: str):
        self.kernel = sk.Kernel()
        self.kernel.add_chat_service(
            "chat-gpt", 
            OpenAIChatCompletion("gpt-4", api_key)
        )
        
    def create_function(self, name: str, prompt: str):
        return self.kernel.create_semantic_function(
            prompt, 
            function_name=name,
            max_tokens=150
        )

# Autogen multi-agent system
from autogen import AssistantAgent, UserProxyAgent

class MultiAgentSystem:
    def __init__(self, openai_api_key: str):
        self.config = {
            "model": "gpt-4",
            "api_key": openai_api_key
        }
        
    def create_agents(self):
        assistant = AssistantAgent(
            name="assistant",
            llm_config=self.config,
            system_message="You are a helpful AI assistant."
        )
        
        user_proxy = UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            code_execution_config={"work_dir": "coding"}
        )
        
        return assistant, user_proxy
        
    def run_conversation(self, message: str):
        assistant, user_proxy = self.create_agents()
        user_proxy.initiate_chat(assistant, message=message)
```

---

## 08. AI Agent in Practice: OpenAI Operator Style

System control agents capable of operating computer interfaces.

```python
import pyautogui
import subprocess
from typing import Dict, Any

class SystemControlAgent:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.actions = {
            "click": self.click_screen,
            "type": self.type_text,
            "screenshot": self.take_screenshot,
            "run_command": self.run_system_command
        }
        
    def click_screen(self, x: int, y: int):
        pyautogui.click(x, y)
        return f"Clicked at ({x}, {y})"
        
    def type_text(self, text: str):
        pyautogui.typewrite(text)
        return f"Typed: {text}"
        
    def take_screenshot(self):
        screenshot = pyautogui.screenshot()
        screenshot.save("current_screen.png")
        return "Screenshot saved"
        
    def run_system_command(self, command: str):
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout or result.stderr
        
    def execute_task(self, task_description: str):
        # Use LLM to break down task into actions
        prompt = f"""
        Task: {task_description}
        Available actions: {list(self.actions.keys())}
        
        Provide step-by-step actions to complete this task.
        Format: action_name(parameters)
        """
        
        response = self.llm.call_model(prompt)
        # Parse and execute actions
        return self.parse_and_execute(response.content)
```

---

## 09. Introduction to Reinforcement Learning

RL-based agents for decision-making and game environments.

### Algorithms:
- Q-learning
- Actor-Critic methods
- Policy-based approaches

```python
import gymnasium as gym
import numpy as np
from collections import defaultdict

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state])
        
    def learn(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
            
        self.q_table[state][action] += self.lr * (target - current_q)
        
    def train(self, episodes=1000):
        for episode in range(episodes):
            state, _ = self.env.reset()
            done = False
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                self.learn(state, action, reward, next_state, done)
                state = next_state
                done = done or truncated

# Flappy Bird RL environment example
class FlappyBirdAgent(QLearningAgent):
    def __init__(self):
        # Custom environment setup for Flappy Bird
        self.game_state = {"bird_y": 0, "pipe_distance": 0, "pipe_height": 0}
        
    def get_state(self):
        # Discretize continuous state space
        bird_y_discrete = int(self.game_state["bird_y"] // 10)
        pipe_dist_discrete = int(self.game_state["pipe_distance"] // 20)
        pipe_height_discrete = int(self.game_state["pipe_height"] // 15)
        
        return (bird_y_discrete, pipe_dist_discrete, pipe_height_discrete)
```

---

## 10. RL Agent - Practical Project

Financial trading bot using reinforcement learning principles.

```python
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from gymnasium import Env, spaces

class TradingEnvironment(Env):
    def __init__(self, df: pd.DataFrame, initial_balance=10000):
        super().__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # -1: short, 0: neutral, 1: long
        
        # Action space: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: price features + portfolio state
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        
    def reset(self, seed=None):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        return self._get_observation(), {}
        
    def step(self, action):
        self._execute_action(action)
        self.current_step += 1
        
        reward = self._calculate_reward()
        done = self.current_step >= len(self.df) - 1
        
        return self._get_observation(), reward, done, False, {}
        
    def _get_observation(self):
        # Technical indicators and portfolio state
        current_price = self.df.iloc[self.current_step]['close']
        sma_5 = self.df.iloc[max(0, self.current_step-4):self.current_step+1]['close'].mean()
        sma_20 = self.df.iloc[max(0, self.current_step-19):self.current_step+1]['close'].mean()
        
        return np.array([
            current_price,
            sma_5,
            sma_20,
            self.balance,
            self.position,
            self.df.iloc[self.current_step]['volume'],
            self.df.iloc[self.current_step]['high'],
            self.df.iloc[self.current_step]['low'],
            self.df.iloc[self.current_step]['open'],
            self._get_portfolio_value()
        ], dtype=np.float32)
        
    def _execute_action(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        
        if action == 1 and self.position <= 0:  # Buy
            shares_to_buy = self.balance // current_price
            cost = shares_to_buy * current_price
            self.balance -= cost
            self.position += shares_to_buy
            
        elif action == 2 and self.position > 0:  # Sell
            revenue = self.position * current_price
            self.balance += revenue
            self.position = 0
            
    def _calculate_reward(self):
        current_value = self._get_portfolio_value()
        return (current_value - self.initial_balance) / self.initial_balance
        
    def _get_portfolio_value(self):
        current_price = self.df.iloc[self.current_step]['close']
        return self.balance + (self.position * current_price)

class TradingBot:
    def __init__(self, data: pd.DataFrame):
        self.env = TradingEnvironment(data)
        self.model = PPO("MlpPolicy", self.env, verbose=1)
        
    def train(self, timesteps=100000):
        self.model.learn(total_timesteps=timesteps)
        
    def predict(self, observation):
        action, _ = self.model.predict(observation)
        return action
        
    def backtest(self, test_data: pd.DataFrame):
        test_env = TradingEnvironment(test_data)
        obs, _ = test_env.reset()
        
        total_reward = 0
        while True:
            action = self.predict(obs)
            obs, reward, done, truncated, _ = test_env.step(action)
            total_reward += reward
            
            if done or truncated:
                break
                
        return total_reward, test_env._get_portfolio_value()
```

---

## 11. Summary and Discussion

### Key Takeaways:
- **Agent Architecture**: Understanding different patterns from simple ReAct to complex multi-agent systems
- **Framework Selection**: Choosing between LangChain, Autogen, Semantic Kernel based on use case
- **Integration Strategies**: Connecting agents with databases, APIs, and external systems
- **Production Considerations**: Deployment, monitoring, and scaling of agent systems

### Next Steps:
- Explore open-source agent frameworks
- Join AI agent development communities  
- Build production-ready agent applications
- Stay updated with emerging tools and techniques

### Practical Applications:
- Customer service automation
- Content generation and curation
- Financial analysis and trading
- System administration and monitoring
- Research and data analysis workflows

This course provides a comprehensive foundation for developing sophisticated AI agents capable of autonomous operation in complex environments.