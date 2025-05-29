# AI LLM Agents (Artificial Intelligence - Large Language Models)

- [SAMPLE CODE](#sample-code)
- [AGENTS (generated code)](#agents-generated-code)
- [RAG (generated code)](#rag-generated-code)
- [AI LLM Agents](#ai-llm-agents)
- [Gradio UI](#gradio-ui)
- [Streamlit UI](#streamlit-ui)
- [Streamlit UI (2)](#streamlit-ui-2)
- [30 Days of Streamlit](#30-days-of-streamlit)
- [GUIDE (generated code)](#0-guide-generated-by-claude-llm)

# SAMPLE CODE

- **AG2 (formerly AutoGen)**
  - **Quick Start** [101-ag2-quick-start.ipynb](100-ag2/101-ag2-quick-start.ipynb)✅
  - **Basic Concepts**
    - **Structured outputs** [102-ag2-basics-1.ipynb](100-ag2/102-ag2-basics-1.ipynb)✅
    - **ConversableAgent, Human in the loop** [103-ag2-basics-2.ipynb](100-ag2/103-ag2-basics-2.ipynb)✅
    - **Orchestrating agents**
      - **Sequential chat** [104-ag2-basics-3.ipynb](100-ag2/104-ag2-basics-3.ipynb)✅
      - **Nested chat** [105-ag2-basics-4.ipynb](100-ag2/105-ag2-basics-4.ipynb)✅
      - **Swarm** [106-ag2-basics-5.ipynb](100-ag2/106-ag2-basics-5.ipynb)✅
      - **Swarm Deep-dive** [TODO](https://docs.ag2.ai/docs/user-guide/advanced-concepts/swarm/deep-dive)⚠️
      - **Ending a chat** [107-ag2-basics-6.ipynb](100-ag2/107-ag2-basics-6.ipynb)✅ TRY⚠️
    - **Tools**
      - **Tools Basics** [108-ag2-basics-7.ipynb](100-ag2/108-ag2-basics-7.ipynb)✅
      - **Tools with Secrets** [109-ag2-basics-8.ipynb](100-ag2/109-ag2-basics-8.ipynb)✅
      - **Tools with Dependency Injection** [TODO](https://docs.ag2.ai/docs/use-cases/notebooks/notebooks/tools_dependency_injection)⚠️
      - **Interoperability**
        - **LangChain Tools Integration** [110-ag2-basics-9.ipynb](100-ag2/110-ag2-basics-9.ipynb)✅
        - **CrewAI Tools Integration** [111-ag2-basics-10.ipynb](100-ag2/111-ag2-basics-10.ipynb)✅
        - **PydanticAI Tools Integration** [112-ag2-basics-11.ipynb](100-ag2/112-ag2-basics-11.ipynb)✅
  - **Advanced Concepts**
    - **RAG** [113-ag2-advanced-1.ipynb](100-ag2/113-ag2-advanced-1.ipynb)✅
      - **Using Neo4j’s graph database with AG2 agents for Q&A notebook** [TODO](https://docs.ag2.ai/docs/use-cases/notebooks/notebooks/agentchat_graph_rag_neo4j)⚠️
      - **Trip planning with a FalkorDB GraphRAG agent using a Swarm notebook** [TODO](https://docs.ag2.ai/docs/use-cases/notebooks/notebooks/agentchat_swarm_graphrag_trip_planner)⚠️
    - **GroupChat**
      - **Overview** [114-ag2-advanced-2.ipynb](100-ag2/114-ag2-advanced-2.ipynb)✅
      - **Group Chat with Tools** [115-ag2-advanced-3.ipynb](100-ag2/115-ag2-advanced-3.ipynb)✅
      - **Customized GroupChat flows** [116-ag2-advanced-4.ipynb](100-ag2/116-ag2-advanced-4.ipynb)✅
        - **More GroupChat examples**
          - **GroupChat with Customized Speaker Selection Method** [TODO](https://docs.ag2.ai/docs/use-cases/notebooks/notebooks/agentchat_groupchat_customized)⚠️
          - **GroupChat with Coder and Visualization Critic** [TODO](https://docs.ag2.ai/docs/use-cases/notebooks/notebooks/agentchat_groupchat_vis)⚠️
          - **GroupChat with Retrieval-Augmented Generation** [TODO](https://docs.ag2.ai/docs/use-cases/notebooks/notebooks/agentchat_groupchat_RAG)⚠️
          - **Implementing Swarm with a GroupChat** [TODO](https://docs.ag2.ai/docs/use-cases/notebooks/notebooks/agentchat_swarm_w_groupchat_legacy)⚠️
      - **Resuming a Group Chat** [117-ag2-advanced-5.ipynb](100-ag2/117-ag2-advanced-5.ipynb)✅ TRY⚠️
    - **Swarm**
      - **Deep-dive** [118-ag2-advanced-6.ipynb](100-ag2/118-ag2-advanced-6.ipynb)✅ TRY⚠️
      - **Nested chats** [119-ag2-advanced-7.ipynb](100-ag2/119-ag2-advanced-7.ipynb)✅ TRY⚠️
      - **Concepts Code** [120-ag2-advanced-8.ipynb](100-ag2/120-ag2-advanced-8.ipynb)✅
      - **Use Case example** [121-ag2-advanced-9.ipynb](100-ag2/121-ag2-advanced-9.ipynb)✅✅✅
      - **More Swarm examples**
        - **Introduction to Swarm notebook** [TODO](https://docs.ag2.ai/docs/use-cases/notebooks/notebooks/agentchat_swarm)⚠️
        - **Swarm with GraphRAG notebook** [TODO](https://docs.ag2.ai/docs/use-cases/notebooks/notebooks/agentchat_swarm_graphrag_trip_planner)⚠️
    - **RealtimeAgent** [TODO](https://docs.ag2.ai/docs/user-guide/advanced-concepts/realtime-agent/index)⚠️
    - **Code Execution** [TODO](https://docs.ag2.ai/docs/user-guide/advanced-concepts/code-execution)⚠️
    - **Conversation Patterns Deep-dive** [TODO](https://docs.ag2.ai/docs/user-guide/advanced-concepts/conversation-patterns-deep-dive)⚠️
    - **LLM Config Deep-dive** [TODO](https://docs.ag2.ai/docs/user-guide/advanced-concepts/llm-configuration-deep-dive)⚠️
    - **Pattern Cookbook** [TODO](https://docs.ag2.ai/docs/user-guide/advanced-concepts/pattern-cookbook/overview)⚠️
  - **Model Providers** [TODO](https://docs.ag2.ai/docs/user-guide/models)⚠️
  - **Reference Agents** [TODO](https://docs.ag2.ai/docs/user-guide/reference-agents/index)⚠️
  - **Reference Tools** [TODO](https://docs.ag2.ai/docs/user-guide/reference-tools/index)⚠️
- **CrewAI**
  - **Quick Start** [201_crewai_quickstart.ipynb](200-crewai/201_crewai_quickstart.ipynb)✅
- **LangGraph**
  - **Quick Start** [301_langgraph_quickstart.ipynb](300-langgraph/301_langgraph_quickstart.ipynb)✅
  - **Tutorials**
    - **Agentic RAG** [302_langgraph_agentic_rag.ipynb](300-langgraph/302_langgraph_agentic_rag.ipynb)✅
    - **Code Assistant** [303_langgraph_code_assistant.ipynb](300-langgraph/303_langgraph_code_assistant.ipynb)✅
- **LlamaIndex**
  - **Quick Start** [401_llamaindex_quickstart.ipynb](400-llamaindex/401_llamaindex_quickstart.ipynb)✅
  - **Starter Examples**
    - **Starter Tutorial (Using OpenAI)** [402_llamaindex_starter_openai.ipynb](400-llamaindex/402_llamaindex_starter_openai.ipynb)✅
    - **Starter Tutorial (Using Local LLMs)** [403_llamaindex_starter_local_llms.ipynb](400-llamaindex/403_llamaindex_starter_local_llms.ipynb)✅ TRY(>32GB)⚠️
  - **Examples**
    - **AgentWorkflow Basic Introduction** [404_llamaindex_agent_workflow.ipynb](400-llamaindex/404_llamaindex_agent_workflow.ipynb)✅
  - **Learn**
    - **Building agents (5 parts)** [405_llamaindex_building_agents.ipynb](400-llamaindex/405_llamaindex_building_agents.ipynb)✅
      - (tools, state, streaming, loop, multi-agent)⚠️TRY(multi-agent, result to dict)
    - **Building Workflows** [TODO](https://docs.llamaindex.ai/en/stable/understanding/workflows/)⚠️
    - **Building a RAG pipeline** [406_llamaindex_building_rag.ipynb](400-llamaindex/406_llamaindex_building_rag.ipynb)✅
    - **Structured Data Extraction** [407_llamaindex_structured_data.ipynb](400-llamaindex/407_llamaindex_structured_data.ipynb)✅
    - **Tracing and Debugging** [TODO](https://docs.llamaindex.ai/en/stable/understanding/tracing_and_debugging/tracing_and_debugging/)⚠️
    - **Evaluating** [TODO](https://docs.llamaindex.ai/en/stable/understanding/evaluating/evaluating/)⚠️
    - **Full-Stack Web Application** [TODO](https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/apps/)⚠️
      - **LlamaIndex Starter Pack** [TODO](https://github.com/logan-markewich/llama_index_starter_pack)⚠️⚠️⚠️
  - **Component Guides (Agents)** [TODO](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/)⚠️
    - **Examples/Module Guides** [TODO](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/modules/)⚠️
  - **Use Cases (Question-Answering (RAG))** [TODO](https://docs.llamaindex.ai/en/stable/use_cases/q_and_a/)⚠️
    - **RAG over Unstructured Docs (Semantic search, Summarization)** [410_llamaindex_unstructured_docs.ipynb](400-llamaindex/410_llamaindex_unstructured_docs.ipynb)✅
    - **QA over Structured Data (Text-to-SQL)** [408_llamaindex_text-to-sql.ipynb](400-llamaindex/408_llamaindex_text-to-sql.ipynb)✅
    - **QA over Structured Data (Text-to-Pandas)** [409_llamaindex_text-to-pandas.ipynb](400-llamaindex/409_llamaindex_text-to-pandas.ipynb)✅

<br>

# AGENTS (generated code)

- **001: Autogen AI Agent, with 2 different agents**
  - by **GitHub Copilot** [001-1-github-copilot.ipynb](001/001-1-github-copilot.ipynb)
  - by **DeepSeek** [001-2-deepseek.ipynb](001/001-2-deepseek.ipynb)
  - by **ChatGPT** [001-3-chatgpt.ipynb](001/001-3-chatgpt.ipynb)
  - by **Claude** [001-4-claude.ipynb](001/001-4-claude.ipynb)
  - by **Cursor** [001-5-cursor-1.ipynb](001/001-5-cursor-1.ipynb)
  - by **Cursor** [001-5-cursor-2.ipynb](001/001-5-cursor-2.ipynb)
- **002: CrewAI AI Agent, with 2 different agents**
  - by **DeepSeek** [002-2-deepseek.ipynb](002/002-2-deepseek.ipynb)
  - by **ChatGPT** [002-3-chatgpt.ipynb](002/002-3-chatgpt.ipynb)✅
  - by **Claude** [002-4-claude.ipynb](002/002-4-claude.ipynb)
  - by **Cursor** [002-5-cursor.ipynb](002/002-5-cursor.ipynb)✅

<br>

# RAG (generated code)
- **Pinecone (vector store)**
  - (quickstart) [rag-000.ipynb](rag/rag-000.ipynb)✅
  - (with LangChain) [rag-001.ipynb](rag/rag-001.ipynb)✅ + data (3 files) (TXT, MD, PDF)
  - (with LangChain) [rag-002.ipynb](rag/rag-002.ipynb)✅ + data (6 files) (MD, IPYNB, PDF, DOCX, XLSX, HTML)
    - ⚠️ IMPROVE LOAD, CHUNK, METADATA FOR ALL FILE TYPES

<br>

# AI LLM Agents

- [LLM Agents](https://www.promptingguide.ai/research/llm-agents) *by Prompt Engineering Guide*
- **AutoGen (Microsoft)**
  - [PyPI](https://pypi.org/user/AutoGenDevs/) autogen-core, autogen-agentchat, autogen-ext
  - [GitHub](https://github.com/microsoft/autogen)
  - [AutoGen](https://microsoft.github.io/autogen/stable/index.html)
  - [AgentChat](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/index.html)
- **AG2**
  - [PyPI](https://pypi.org/user/autogen-ai/) autogen, ag2
  - [GitHub](https://github.com/ag2ai/ag2)
  - [AG2](https://ag2.ai/)
- **CrewAI**
  - [PyPI](https://pypi.org/project/crewai/) crewai
  - [GitHub](https://github.com/crewAIInc/crewAI)
  - [CrewAI](https://www.crewai.com/)
- **LangChain**
  - [Agents](https://python.langchain.com/docs/how_to/#agents)
  - [Build an Agent](https://python.langchain.com/docs/tutorials/agents/)
  - [LangGraph](https://langchain-ai.github.io/langgraph/)
- **LlamaIndex**
  - [Agents (use cases)](https://docs.llamaindex.ai/en/stable/use_cases/agents/)
  - [Agents (module guides)](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/)

<br>

# Gradio UI
- **Sample code with OpenAI** [app.py](gradio/app.py)✅

<br>

# Streamlit UI
- **AG2** (GroupChat, GroupChatManager) [ag2-groupchat.ipynb](streamlit/ag2-groupchat.ipynb)✅
- **AG2** (Swarm) [ag2-swarm-1.ipynb](streamlit/ag2-swarm-1.ipynb)✅
- **AG2** (Swarm) [ag2-swarm-2.ipynb](streamlit/ag2-swarm-2.ipynb)✅
- **AG2** (Swarm + Streamlit) [ag2-swarm-2-streamlit.py](streamlit/ag2-swarm-2-streamlit.py)✅ [Recipe Creator Swarm] [PDF](streamlit/ag2-swarm-2-streamlit.pdf) [MD](streamlit/recipe_italian.md)
- **AG2** (Swarm + Streamlit) [main.py](streamlit/main.py)✅ [AI Game Design Agent Team] [PDF](streamlit/main.pdf)
- **CrewAI** (Agents + Streamlit) [crewai-1-streamlit.py](streamlit/crewai-1-streamlit.py)✅ [Travel Planning Crew] [PDF](streamlit/crewai-1-streamlit.pdf)

<br>

# Streamlit UI (2)

- **Basic Streamlit by GitHub Copilot** (controls, data, chart, file uploader, csv) [000-streamlit-basic-copilot.py](streamlit-2/000-streamlit-basic-copilot.py)✅
- **Re-write your text** (openai, prompt template) [001-streamlit-redaction-improver.py](streamlit-2/001-streamlit-redaction-improver.py)✅
- **Blog Post Generator** (openai, prompt template) [002-streamlit-blog-post-generator.py](streamlit-2/002-streamlit-blog-post-generator.py)✅
- **AI Long Text Summarizer** (openai, upload file, recursive character text splitter, summarize chain) [003-streamlit-split-and-summarize.py](streamlit-2/003-streamlit-split-and-summarize.py)✅
- **Writing Text Summarization** (openai, character text splitter, summarize chain) [004-streamlit-text-summarization.py](streamlit-2/004-streamlit-text-summarization.py)✅
- **Extract Key Information from Product Reviews** (openai, prompt template, format output)<br>`005-streamlit-extract-json-from-review.py`
- **Evaluate a RAG App** (openai, upload file, rag with faiss, qa eval chain)<br>`009-streamlit-evaluate-QandA-from-long-document.py`

## 30 Days of Streamlit
- [Get started with Streamlit](https://docs.streamlit.io/get-started)<br>
  [30 Days of Streamlit](https://30days.streamlit.app/)

- **Install conda**<br>
  [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main)
- **Terminal**
  ```
  conda create -n stenv python=3.9
  conda activate stenv
  pip install streamlit
  streamlit hello
  ```
  ```
  conda create --name myenv
  conda create --name myenv python=3.8 numpy
  conda activate myenv
  conda install --file requirements.txt
  conda deactivate

  conda info --envs
  conda env list
  conda env remove --name myenv
  ```
  ```
  python -m venv venv
  .\venv\Scripts\activate  # Windows
  source venv/bin/activate  # macOS
  pip install -r requirements.txt
  ```
- `streamlit_app.py`<br>`streamlit_app_layout.py`<br>`streamlit_app_dashboard.py`<br>`streamlit_app_shap.py`<br>`streamlit_app_zero_shot_classifier.py`<br>`streamlit_app_art.py`
  ```
  streamlit run streamlit_app.py
  ```

## 0. GUIDE (*Generated by Claude LLM*)

- **1. AI Agents Development Course Guide** 📗 [01-ai-agents-en.md](0-GUIDE/01-ai-agents-en.md) <img src="https://flagcdn.com/w40/cz.png" width="15" alt="Czech Flag"> [01-ai-agents-cz.md](0-GUIDE/01-ai-agents-cz.md)
  01. **AI API and First Agent** 📗 [01-01a.md](0-GUIDE/01-01a.md)
  02. **Databases for Agents** 📗 [01-02a.md](0-GUIDE/01-02a.md)
  03. **Model Context Protocol (MCP)** 📗 [01-03a.md](0-GUIDE/01-03a.md)
  04. **Automation and Workflow with n8n** 📗 [01-04a.md](0-GUIDE/01-04a.md)
  05. **Custom Agent Framework** 📗 [01-05a.md](0-GUIDE/01-05a.md)
  06. **LangChain and LangGraph** 📗 [01-06a.md](0-GUIDE/01-06a.md)
  07. **Semantic Kernel and Autogen** 📗 [01-07a.md](0-GUIDE/01-07a.md)
  08. **AI Agent in Practice: OpenAI Operator Style** 📗 [01-08a.md](0-GUIDE/01-08a.md)
  09. **Introduction to Reinforcement Learning** 📗 [01-09a.md](0-GUIDE/01-09a.md)
  10. **RL Agent - Practical Project** 📗 [01-10a.md](0-GUIDE/01-10a.md)
  11. **Summary and Discussion** 📗 [01-11a.md](0-GUIDE/01-11a.md)

- **2. AI Developer Course Guide** 📗 [02-ai-developer-en.md](0-GUIDE/02-ai-developer-en.md) <img src="https://flagcdn.com/w40/cz.png" width="15" alt="Czech Flag"> [02-ai-developer-cz.md](0-GUIDE/02-ai-developer-cz.md)
  01. **Introduction to Neural Networks and Generative AI**
  02. **Prompt Design and LLM Evaluation**
  03. **Training Data Preparation**
  04. **OpenAI Models and Fine-tuning**
  05. **HuggingFace Introduction**
  06. **Advanced Fine-tuning with HuggingFace**
  07. **LangChain – AI Application Development**
  08. **LangGraph (AI Agents)**
  09. **Semantic Kernel (AI Agents)**
  10. **Autogen (Advanced AI Agent Framework)**
  11. **AI Agent Development Workshop**
  12. **Summary and Future Directions**

- **3. AI Chatbot Development Course Guide** 📗 [03-ai-chatbots-en.md](0-GUIDE/03-ai-chatbots-en.md) <img src="https://flagcdn.com/w40/cz.png" width="15" alt="Czech Flag"> [03-ai-chatbots-cz.md](0-GUIDE/03-ai-chatbots-cz.md)
  01. **Introduction to AI Assistants and Creating Your First GPT Assistant**
  02. **Capabilities and Limitations of GPT Assistants**
  03. **Vector Databases and Their Applications**
  04. **Multi-Agent Orchestration with LangGraph**
  05. **Advanced API Integration for Dynamic Responses**
  06. **Monitoring and Performance Optimization**
  07. **Code Integration in GPT Assistant Responses**
  08. **Customer Assistant Design and Configuration**
  09. **Testing and Optimization of Customer Assistants**
  10. **Building Emotional Intelligence and Digital Twins**
  11. **Future Development Planning and Advanced Applications**