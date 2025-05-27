# Průvodce kurzem Vývoj AI agentů
## Budování autonomních AI systémů pro automatizaci komplexních úloh

Tento komplexní kurz pokrývá vývoj AI agentů schopných autonomního vykonávání úloh a automatizace složitých pracovních postupů pomocí Pythonu a moderních AI frameworků.

---

## 01. AI API a první agent

Pochopení základů Large Language Modelů (LLM) a integrace API pro vývoj agentů.

### Klíčové koncepty:
- Poskytovatelé LLM API (OpenAI, Anthropic, Ollama, HuggingFace)
- Parametry modelů a rozdíly mezi poskytovateli
- Architektura tool-calling a základy agentů

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

# Příklad definice nástroje
def get_weather(location: str) -> str:
    return f"Počasí v {location}: 22°C, slunečno"

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Získání informací o počasí",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }
    }
}]
```

---

## 02. Databáze pro agenty

Integrace různých typů databází s AI agenty pro ukládání a získávání znalostí.

### Typy databází:
- SQL (MS SQL Server)
- NoSQL (MongoDB) 
- Vektorové databáze (Chroma, Elasticsearch)

```python
import chromadb
from pymongo import MongoClient
import pyodbc

class DatabaseAgent:
    def __init__(self):
        # Vektorová databáze pro embeddingy
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection("knowledge_base")
        
        # MongoDB pro ukládání dokumentů
        self.mongo_client = MongoClient('mongodb://localhost:27017/')
        self.db = self.mongo_client['agent_db']
        
    def store_knowledge(self, text: str, metadata: dict):
        # Uložení do vektorové databáze
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

Pochopení MCP pro interoperabilitu mezi AI systémy a externími nástroji.

### Komponenty MCP:
- Standardizace struktury zpráv
- Definice nástrojů a zdrojů
- Kompatibilita napříč platformami

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
        return {"error": "Neznámá metoda nebo nástroj"}
```

---

## 04. Automatizace a workflow s n8n

Vizuální automatizace workflow a orchestrace agentů pomocí platformy n8n.

### Funkce:
- Vizuální návrh procesů
- Tvorba workflow založená na uzlech
- Integrace databází a API
- Možnosti integrace LLM

```python
# Příklad integrace n8n webhooku
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

## 05. Vlastní framework pro agenty

Budování agentů od základu s různými architektonickými vzory.

### Typy agentů:
- Autonomní agenti
- Workflow-based agenti
- ReAct (Reasoning + Acting) vzor

```python
from enum import Enum
from typing import List, Tuple

class AgentState(Enum):
    THINKING = "přemýšlení"
    ACTING = "jednání"
    OBSERVING = "pozorování"
    FINISHED = "dokončeno"

class ReActAgent:
    def __init__(self, llm_client, tools: dict):
        self.llm = llm_client
        self.tools = tools
        self.state = AgentState.THINKING
        self.memory = []
        
    def think(self, observation: str) -> str:
        prompt = f"""
        Předchozí akce: {self.memory}
        Aktuální pozorování: {observation}
        
        Promysli si krok za krokem, co dělat dále.
        Dostupné nástroje: {list(self.tools.keys())}
        """
        response = self.llm.call_model(prompt)
        return response.content
        
    def act(self, thought: str) -> Tuple[str, str]:
        # Určení nástroje na základě úvahy
        for tool_name, tool_func in self.tools.items():
            if tool_name.lower() in thought.lower():
                result = tool_func()
                self.memory.append(f"Použil {tool_name}: {result}")
                return tool_name, result
        return "no_action", "Nebyl nalezen vhodný nástroj"
        
    def run(self, initial_prompt: str, max_iterations: int = 10):
        observation = initial_prompt
        
        for i in range(max_iterations):
            if self.state == AgentState.FINISHED:
                break
                
            thought = self.think(observation)
            action, result = self.act(thought)
            observation = result
            
            if "úkol dokončen" in result.lower():
                self.state = AgentState.FINISHED
                
        return self.memory
```

---

## 06. LangChain a LangGraph

Pokročilá orchestrace agentů pomocí ekosystému LangChain.

### Komponenty:
- Šablony promptů a řetězce
- Retrieval-Augmented Generation (RAG)
- Orchestrace LangServe, LangSmith, LangGraph

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
        Kontext: {context}
        Otázka: {question}
        
        Odpověz na otázku na základě poskytnutého kontextu.
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

# Správa stavu LangGraph
from langgraph.graph import StateGraph
from typing import TypedDict

class AgentState(TypedDict):
    messages: List[str]
    current_task: str
    completed: bool

def create_langgraph_agent():
    workflow = StateGraph(AgentState)
    
    def process_task(state: AgentState):
        # Zpracování aktuálního úkolu
        state["completed"] = True
        return state
        
    workflow.add_node("process", process_task)
    workflow.set_entry_point("process")
    workflow.set_finish_point("process")
    
    return workflow.compile()
```

---

## 07. Semantic Kernel a Autogen

Multi-agentní systémy a pokročilé frameworky pro spolupráci agentů.

### Funkce:
- Plánování agentů a systémy paměti
- Inter-agentní komunikace
- Integrace Autogen Studio

```python
# Příklad Semantic Kernel
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

# Multi-agentní systém Autogen
from autogen import AssistantAgent, UserProxyAgent

class MultiAgentSystem:
    def __init__(self, openai_api_key: str):
        self.config = {
            "model": "gpt-4",
            "api_key": openai_api_key
        }
        
    def create_agents(self):
        assistant = AssistantAgent(
            name="asistent",
            llm_config=self.config,
            system_message="Jsi užitečný AI asistent."
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

## 08. AI agent v praxi: OpenAI Operator styl

Agenti pro ovládání systému schopní operovat počítačová rozhraní.

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
        return f"Kliknuto na ({x}, {y})"
        
    def type_text(self, text: str):
        pyautogui.typewrite(text)
        return f"Napsáno: {text}"
        
    def take_screenshot(self):
        screenshot = pyautogui.screenshot()
        screenshot.save("current_screen.png")
        return "Screenshot uložen"
        
    def run_system_command(self, command: str):
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout or result.stderr
        
    def execute_task(self, task_description: str):
        # Použití LLM pro rozdělení úkolu na akce
        prompt = f"""
        Úkol: {task_description}
        Dostupné akce: {list(self.actions.keys())}
        
        Poskytni krok za krokem akce pro dokončení tohoto úkolu.
        Formát: název_akce(parametry)
        """
        
        response = self.llm.call_model(prompt)
        # Parsování a vykonání akcí
        return self.parse_and_execute(response.content)
```

---

## 09. Úvod do reinforcement learningu

RL-based agenti pro rozhodování a herní prostředí.

### Algoritmy:
- Q-learning
- Actor-Critic metody
- Policy-based přístupy

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

# Příklad RL prostředí pro Flappy Bird
class FlappyBirdAgent(QLearningAgent):
    def __init__(self):
        # Nastavení vlastního prostředí pro Flappy Bird
        self.game_state = {"bird_y": 0, "pipe_distance": 0, "pipe_height": 0}
        
    def get_state(self):
        # Diskretizace kontinuálního stavového prostoru
        bird_y_discrete = int(self.game_state["bird_y"] // 10)
        pipe_dist_discrete = int(self.game_state["pipe_distance"] // 20)
        pipe_height_discrete = int(self.game_state["pipe_height"] // 15)
        
        return (bird_y_discrete, pipe_dist_discrete, pipe_height_discrete)
```

---

## 10. RL agent – praktický projekt

Finanční trading bot využívající principy reinforcement learningu.

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
        self.position = 0  # -1: short, 0: neutrální, 1: long
        
        # Prostor akcí: 0=držet, 1=koupit, 2=prodat
        self.action_space = spaces.Discrete(3)
        
        # Prostor pozorování: cenové vlastnosti + stav portfolia
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
        # Technické indikátory a stav portfolia
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
        
        if action == 1 and self.position <= 0:  # Koupit
            shares_to_buy = self.balance // current_price
            cost = shares_to_buy * current_price
            self.balance -= cost
            self.position += shares_to_buy
            
        elif action == 2 and self.position > 0:  # Prodat
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

## 11. Shrnutí a diskuze

### Klíčové poznatky:
- **Architektura agentů**: Pochopení různých vzorů od jednoduchého ReAct po složité multi-agentní systémy
- **Výběr frameworku**: Volba mezi LangChain, Autogen, Semantic Kernel na základě případu užití
- **Strategie integrace**: Propojení agentů s databázemi, API a externími systémy
- **Produkční aspekty**: Nasazení, monitorování a škálování agentních systémů

### Další kroky:
- Prozkoumání open-source frameworků pro agenty
- Připojení ke komunitám vývojářů AI agentů
- Budování produkčně připravených agentních aplikací
- Sledování nových nástrojů a technik

### Praktické aplikace:
- Automatizace zákaznického servisu
- Generování a kurátorování obsahu
- Finanční analýza a trading
- Správa systémů a monitorování
- Výzkum a workflow analýzy dat

Tento kurz poskytuje komplexní základ pro vývoj sofistikovaných AI agentů schopných autonomní práce ve složitých prostředích.