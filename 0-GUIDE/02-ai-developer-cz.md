# Průvodce kurzem AI Developer
Ovládni celý vývojový cyklus AI agentů – od optimalizace výkonu přes práci s daty až po jejich nasazení do komplexních systémů.

---

## PŘEHLED KURZU
Nauč se porozumět vývoji LLM a metodám jejich trénování, ladění a optimalizace včetně prompt engineeringu, fine-tuningu a RAG. Získáš zkušenosti s přípravou trénovacích dat, hodnocením výkonu modelů a jejich integrací do AI systémů pomocí Pythonu s klíčovými frameworky jako PyTorch, HuggingFace a LangChain.

---

## 01. Úvod do neuronových sítí a generativní AI
Přehled architektur neuronových sítí, trénovacích procesů a aplikací generativní AI napříč textovými, obrazovými, zvukovými a video doménami.

```python
import torch
import torch.nn as nn

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads), 
            num_layers=6
        )
        
    def forward(self, x):
        embedded = self.embedding(x)
        return self.transformer(embedded)
```

## 02. Návrh promptů a hodnocení LLM
Systematický přístup k prompt engineeringu využívající techniky one-shot, few-shot a Chain of Thought s komplexními metodikami benchmarkingu.

```python
def evaluate_prompt_performance(model, prompts, test_cases):
    results = []
    for prompt in prompts:
        for test_input in test_cases:
            response = model.generate(f"{prompt}\n{test_input}")
            score = calculate_accuracy(response, expected_output)
            results.append({'prompt': prompt, 'score': score})
    return results
```

## 03. Příprava dat pro trénování
Strategie sběru, čištění a formátování dat pro doménově specifickou adaptaci modelů a optimalizaci výkonu.

```python
import pandas as pd
from datasets import Dataset

def prepare_training_data(raw_data, tokenizer):
    processed_data = []
    for item in raw_data:
        tokens = tokenizer.encode(item['text'])
        if len(tokens) <= 512:  # Filtrování podle délky
            processed_data.append({
                'input_ids': tokens,
                'labels': tokens,
                'attention_mask': [1] * len(tokens)
            })
    return Dataset.from_list(processed_data)
```

## 04. Modely OpenAI a fine-tuning
Komplexní využití OpenAI API včetně chat completions, embeddingů a custom fine-tuningu modelů pro specifické aplikace.

```python
import openai

def fine_tune_openai_model(training_file_id):
    response = openai.FineTuningJob.create(
        training_file=training_file_id,
        model="gpt-3.5-turbo",
        hyperparameters={
            "n_epochs": 3,
            "batch_size": 4,
            "learning_rate_multiplier": 0.1
        }
    )
    return response.id
```

## 05. Úvod do HuggingFace
Práce s HuggingFace ekosystémem včetně datasets, transformers, tokenizace a nasazování modelů přes endpoints.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Načtení předtrénovaného modelu a tokenizeru
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Načtení datasetu
dataset = load_dataset("squad", split="train[:1000]")
```

## 06. Pokročilý fine-tuning v HuggingFace
Implementace pokročilých technik ladění včetně LoRA, QLoRA, PPO a RLHF pro task-specifickou optimalizaci.

```python
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer

# Konfigurace LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1
)

# Aplikace LoRA na model
model = get_peft_model(model, lora_config)

# Konfigurace trénování
training_args = TrainingArguments(
    output_dir="./lora-model",
    num_train_epochs=3,
    per_device_train_batch_size=4
)
```

## 07. LangChain – Vývoj AI aplikací
Řetězení modelů, orchestrace promptů a implementace retrieval-augmented generation (RAG) pro produkční AI aplikace.

```python
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Nastavení RAG
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(documents, embeddings)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)
```

## 08. LangGraph (AI agenti)
Orchestrace více agentů a jejich dohled s integrovanými RAG schopnostmi pro správu komplexních AI pipeline.

```python
from langgraph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    messages: list
    current_agent: str

def create_agent_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("researcher", research_agent)
    workflow.add_node("writer", writing_agent)
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", END)
    return workflow.compile()
```

## 09. Semantic Kernel (AI agenti)
Vývoj sémantických funkcí a AI plánování s vestavěnými funkcemi a plugin architekturou.

```python
import semantic_kernel as sk

kernel = sk.Kernel()

# Definice sémantické funkce
summarize_function = kernel.create_semantic_function(
    "Shrň následující text: {{$input}}",
    max_tokens=150,
    temperature=0.7
)

# Spuštění funkce
result = await summarize_function("Váš text zde...")
```

## 10. Autogen (Pokročilý framework pro AI agenty)
Orchestrace agentů v Autogen s řízením workflow v AutogenStudio a integrací externích nástrojů.

```python
import autogen

config_list = [{"model": "gpt-4", "api_key": "your-key"}]

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={"config_list": config_list}
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "coding"}
)

# Multi-agent konverzace
user_proxy.initiate_chat(assistant, message="Vyřeš tento problém...")
```

## 11. Vývoj AI agentů – praktický workshop
Praktická implementace AI agentů pomocí LangGraph a OpenAI s automatizovanými interakcemi modelů a testováním.

```python
def deploy_ai_agent(agent_config):
    agent = create_agent(agent_config)
    
    # Test funkcionality agenta
    test_cases = load_test_scenarios()
    for test in test_cases:
        result = agent.execute(test['input'])
        assert validate_output(result, test['expected'])
    
    # Nasazení do produkce
    return deploy_to_kubernetes(agent)
```

## 12. Shrnutí a budoucí směry
Integrace konceptů kurzu s novými trendy včetně multimodálních LLM, Mixture of Experts (MoE) a CrewAI frameworků pro pokračující specializaci.