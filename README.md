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
- [PROJECTS (generated code)](#0-projects-generated-by-claude-sonnet-4)

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

- **Basic Streamlit by GitHub Copilot** (controls, data, chart, file uploader, csv) [000-streamlit-basic-copilot.py](streamlit-2/000-streamlit-basic-copilot.py)✅ [PDF](streamlit-2/streamlit-output/000%20Basic%20Streamlit%20by%20GitHub%20Copilot.pdf)
- **Re-write your text** (openai, prompt template) [001-streamlit-redaction-improver.py](streamlit-2/001-streamlit-redaction-improver.py)✅ [PDF](streamlit-2/streamlit-output/001%20Re-write%20your%20text.pdf)
- **Blog Post Generator** (openai, prompt template) [002-streamlit-blog-post-generator.py](streamlit-2/002-streamlit-blog-post-generator.py)✅ [PDF](streamlit-2/streamlit-output/002%20Blog%20Post%20Generator.pdf)
- **AI Long Text Summarizer** (openai, upload file, recursive character text splitter, summarize chain) [003-streamlit-split-and-summarize.py](streamlit-2/003-streamlit-split-and-summarize.py)✅ [PDF](streamlit-2/streamlit-output/003%20AI%20Long%20Text%20Summarizer%20(date-cz).pdf), [PDF](streamlit-2/streamlit-output/003%20AI%20Long%20Text%20Summarizer%20(david-lynch).pdf)
- **Writing Text Summarization** (openai, character text splitter, summarize chain) [004-streamlit-text-summarization.py](streamlit-2/004-streamlit-text-summarization.py)✅ [PDF](streamlit-2/streamlit-output/004%20Writing%20Text%20Summarization.pdf)
- **Extract Key Information from Product Reviews** (openai, prompt template, format output) [005-streamlit-extract-json-from-review.py](streamlit-2/005-streamlit-extract-json-from-review.py)✅ [PDF](streamlit-2/streamlit-output/005%20Extract%20Key%20Information%20from%20Product%20Reviews.pdf)
- **Evaluate a RAG App** (openai, upload file, rag with faiss, qa eval chain) [009-streamlit-evaluate-QandA-from-long-document.py](streamlit-2/009-streamlit-evaluate-QandA-from-long-document.py)✅ [PDF](streamlit-2/streamlit-output/009%20Evaluate%20a%20RAG%20App%20-%20CORRECT.pdf), [PDF](streamlit-2/streamlit-output/009%20Evaluate%20a%20RAG%20App%20-%20INCORRECT.pdf)

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

## 0. GUIDE <small>(*Generated by Claude LLM*)</small>
*<small>Claude Sonnet 4, Claude 3.7 Sonnet Thinking, Claude web</small>*

- **1. AI Agents Development Course Guide** 📗 [01-ai-agents-en.md](0-GUIDE/01-ai-agents-en.md) <img src="https://flagcdn.com/w40/cz.png" width="15" alt="Czech Flag"> [01-ai-agents-cz.md](0-GUIDE/01-ai-agents-cz.md)
  01. **AI API and First Agent** 📗 [01-01-cs4a.md](0-GUIDE/01-01-cs4a.md) 📗 [01-01a.md](0-GUIDE/01-01a.md) 📗 [01-01-cwa.md](0-GUIDE/01-01-cwa.md)
  02. **Databases for Agents** 📗 [01-02-cs4a.md](0-GUIDE/01-02-cs4a.md) 📗 [01-02a.md](0-GUIDE/01-02a.md) 📗 [01-02-cwa.md](0-GUIDE/01-02-cwa.md)
  03. **Model Context Protocol (MCP)** 📗 [01-03-cs4a.md](0-GUIDE/01-03-cs4a.md) 📗 [01-03a.md](0-GUIDE/01-03a.md) 📗 [01-03-cwa.md](0-GUIDE/01-03-cwa.md)
  04. **Automation and Workflow with n8n** 📗 [01-04-cs4a.md](0-GUIDE/01-04-cs4a.md) 📗 [01-04a.md](0-GUIDE/01-04a.md) 📗 [01-04-cwa.md](0-GUIDE/01-04-cwa.md)
  05. **Custom Agent Framework** 📗 [01-05-cs4a.md](0-GUIDE/01-05-cs4a.md) 📗 [01-05a.md](0-GUIDE/01-05a.md) 📗 [01-05-cwa.md](0-GUIDE/01-05-cwa.md)
  06. **LangChain and LangGraph** 📗 [01-06-cs4a.md](0-GUIDE/01-06-cs4a.md) 📗 [01-06a.md](0-GUIDE/01-06a.md) 📗 [01-06-cwa.md](0-GUIDE/01-06-cwa.md)
  07. **Semantic Kernel and Autogen** 📗 [01-07-cs4a.md](0-GUIDE/01-07-cs4a.md) 📗 [01-07a.md](0-GUIDE/01-07a.md) 📗 [01-07-cwa.md](0-GUIDE/01-07-cwa.md)
  08. **AI Agent in Practice: OpenAI Operator Style** 📗 [01-08-cs4a.md](0-GUIDE/01-08-cs4a.md) 📗 [01-08a.md](0-GUIDE/01-08a.md) 📗 [01-08-cwa.md](0-GUIDE/01-08-cwa.md)
  09. **Introduction to Reinforcement Learning** 📗 [01-09-cs4a.md](0-GUIDE/01-09-cs4a.md) 📗 [01-09a.md](0-GUIDE/01-09a.md) 📗 [01-09-cwa.md](0-GUIDE/01-09-cwa.md)
  10. **RL Agent - Practical Project** 📗 [01-10-cs4a.md](0-GUIDE/01-10-cs4a.md) 📗 [01-10a.md](0-GUIDE/01-10a.md) 📗 [01-10-cwa.md](0-GUIDE/01-10-cwa.md)
  11. **Summary and Discussion** 📗 [01-11-cs4a.md](0-GUIDE/01-11-cs4a.md) 📗 [01-11a.md](0-GUIDE/01-11a.md) 📗 [01-11-cwa.md](0-GUIDE/01-11-cwa.md)

- **2. AI Developer Course Guide** 📗 [02-ai-developer-en.md](0-GUIDE/02-ai-developer-en.md) <img src="https://flagcdn.com/w40/cz.png" width="15" alt="Czech Flag"> [02-ai-developer-cz.md](0-GUIDE/02-ai-developer-cz.md)
  01. **Introduction to Neural Networks and Generative AI** 📗 [02-01-cs4a.md](0-GUIDE/02-01-cs4a.md) 📗 [02-01a.md](0-GUIDE/02-01a.md), [02-01b.md](0-GUIDE/02-01b.md) 📗 [02-01-cwa.md](0-GUIDE/02-01-cwa.md)
  02. **Prompt Design and LLM Evaluation** 📗 [02-02-cs4a.md](0-GUIDE/02-02-cs4a.md) 📗 [02-02a.md](0-GUIDE/02-02a.md) 📗 [02-02-cwa.md](0-GUIDE/02-02-cwa.md)
  03. **Training Data Preparation** 📗 [02-03-cs4a.md](0-GUIDE/02-03-cs4a.md) 📗 [02-03a.md](0-GUIDE/02-03a.md) 📗 [02-03-cwa.md](0-GUIDE/02-03-cwa.md)
  04. **OpenAI Models and Fine-tuning** 📗 [02-04-cs4a.md](0-GUIDE/02-04-cs4a.md) 📗 [02-04a.md](0-GUIDE/02-04a.md) 📗 [02-04-cwa.md](0-GUIDE/02-04-cwa.md)
  05. **HuggingFace Introduction** 📗 [02-05-cs4a.md](0-GUIDE/02-05-cs4a.md) 📗 [02-05a.md](0-GUIDE/02-05a.md) 📗 [02-05-cwa.md](0-GUIDE/02-05-cwa.md)
  06. **Advanced Fine-tuning with HuggingFace** 📗 [02-06-cs4a.md](0-GUIDE/02-06-cs4a.md) 📗 [02-06a.md](0-GUIDE/02-06a.md) 📗 [02-06-cwa.md](0-GUIDE/02-06-cwa.md)
  07. **LangChain – AI Application Development** 📗 [02-07-cs4a.md](0-GUIDE/02-07-cs4a.md) 📗 [02-07a.md](0-GUIDE/02-07a.md) 📗 [02-07-cwa.md](0-GUIDE/02-07-cwa.md)
  08. **LangGraph (AI Agents)** 📗 [02-08-cs4a.md](0-GUIDE/02-08-cs4a.md) 📗 [02-08a.md](0-GUIDE/02-08a.md) 📗 [02-08-cwa.md](0-GUIDE/02-08-cwa.md)
  09. **Semantic Kernel (AI Agents)** 📗 [02-09-cs4a.md](0-GUIDE/02-09-cs4a.md) 📗 [02-09a.md](0-GUIDE/02-09a.md) 📗 [02-09-cwa.md](0-GUIDE/02-09-cwa.md)
  10. **Autogen (Advanced AI Agent Framework)** 📗 [02-10-cs4a.md](0-GUIDE/02-10-cs4a.md) 📗 [02-10a.md](0-GUIDE/02-10a.md) 📗 [02-10-cwa.md](0-GUIDE/02-10-cwa.md)
  11. **AI Agent Development Workshop** 📗 [02-11-cs4a.md](0-GUIDE/02-11-cs4a.md) 📗 [02-11a.md](0-GUIDE/02-11a.md) 📗 [02-11-cwa.md](0-GUIDE/02-11-cwa.md)
  12. **Summary and Future Directions** 📗 [02-12-cs4a.md](0-GUIDE/02-12-cs4a.md) 📗 [02-12a.md](0-GUIDE/02-12a.md) 📗 [02-12-cwa.md](0-GUIDE/02-12-cwa.md)

- **3. AI Chatbot Development Course Guide** 📗 [03-ai-chatbots-en.md](0-GUIDE/03-ai-chatbots-en.md) <img src="https://flagcdn.com/w40/cz.png" width="15" alt="Czech Flag"> [03-ai-chatbots-cz.md](0-GUIDE/03-ai-chatbots-cz.md)
  01. **Introduction to AI Assistants and Creating Your First GPT Assistant** 📗 [03-01-cs4a.md](0-GUIDE/03-01-cs4a.md) 📗 [03-01a.md](0-GUIDE/03-01a.md), [03-01b.md](0-GUIDE/03-01b.md) 📗 [03-01-cwa.md](0-GUIDE/03-01-cwa.md)
  02. **Capabilities and Limitations of GPT Assistants** 📗 [03-02-cs4a.md](0-GUIDE/03-02-cs4a.md) 📗 [03-02a.md](0-GUIDE/03-02a.md) 📗 [03-02-cwa.md](0-GUIDE/03-02-cwa.md)
  03. **Vector Databases and Their Applications** 📗 [03-03-cs4a.md](0-GUIDE/03-03-cs4a.md) 📗 [03-03a.md](0-GUIDE/03-03a.md), [03-03b.md](0-GUIDE/03-03b.md) 📗 [03-03-cwa.md](0-GUIDE/03-03-cwa.md)
  04. **Multi-Agent Orchestration with LangGraph** 📗 [03-04-cs4a.md](0-GUIDE/03-04-cs4a.md) 📗 [03-04a.md](0-GUIDE/03-04a.md) 📗 [03-04-cwa.md](0-GUIDE/03-04-cwa.md)
  05. **Advanced API Integration for Dynamic Responses** 📗 [03-05-cs4a.md](0-GUIDE/03-05-cs4a.md) 📗 [03-05a.md](0-GUIDE/03-05a.md) 📗 [03-05-cwa.md](0-GUIDE/03-05-cwa.md)
  06. **Monitoring and Performance Optimization** 📗 [03-06-cs4a.md](0-GUIDE/03-06-cs4a.md) 📗 [03-06a.md](0-GUIDE/03-06a.md) 📗 [03-06-cwa.md](0-GUIDE/03-06-cwa.md)
  07. **Code Integration in GPT Assistant Responses** 📗 [03-07-cs4a.md](0-GUIDE/03-07-cs4a.md) 📗 [03-07a.md](0-GUIDE/03-07a.md) 📗 [03-07-cwa.md](0-GUIDE/03-07-cwa.md)
  08. **Customer Assistant Design and Configuration** 📗 [03-08-cs4a.md](0-GUIDE/03-08-cs4a.md) 📗 [03-08a.md](0-GUIDE/03-08a.md) 📗 [03-08-cwa.md](0-GUIDE/03-08-cwa.md)
  09. **Testing and Optimization of Customer Assistants** 📗 [03-09-cs4a.md](0-GUIDE/03-09-cs4a.md) 📗 [03-09a.md](0-GUIDE/03-09a.md) 📗 [03-09-cwa.md](0-GUIDE/03-09-cwa.md)
  10. **Building Emotional Intelligence and Digital Twins** 📗 [03-10-cs4a.md](0-GUIDE/03-10-cs4a.md) 📗 [03-10a.md](0-GUIDE/03-10a.md) 📗 [03-10-cwa.md](0-GUIDE/03-10-cwa.md)
  11. **Future Development Planning and Advanced Applications** 📗 [03-11-cs4a.md](0-GUIDE/03-11-cs4a.md) 📗 [03-11a.md](0-GUIDE/03-11a.md) 📗 [03-11-cwa.md](0-GUIDE/03-11-cwa.md)

## 0. PROJECTS <small>(*Generated by [Claude Sonnet 4](https://www.anthropic.com/news/claude-4)*)</small>

- AI-LLM **MCP** *(Model Context Protocol)*
  
  - <small>Project selection by [Claude](https://claude.ai/) (01-01)</small>
    - **Smart Code Assistant** 📗 [01-01-01a.md](0-PROJECTS/01-01-01a.md), [01-01-01b.md](0-PROJECTS/01-01-01b.md), [01-01-01c.md](0-PROJECTS/01-01-01c.md)  
      (MCP, IDE integration, code analysis, Git repositories, documentation generation, TypeScript/Python)  
    - **Personal Knowledge Manager** 📗 [01-01-02.md](0-PROJECTS/01-01-02.md)  
      (MCP, vector databases, RAG, document indexing, semantic search, note-taking, Obsidian/Notion integration)  
    - **Enterprise Data Analytics Dashboard** 📗 [01-01-03.md](0-PROJECTS/01-01-03.md)  
      (MCP, SQL databases, business intelligence, data visualization, real-time reporting, PostgreSQL/MySQL)  
    - **Multi-Agent Customer Support System** 📗 [01-01-04.md](0-PROJECTS/01-01-04.md)  
      (MCP, agent orchestration, ticket management, knowledge base integration, CRM systems, Zendesk/Salesforce)  
    - **Content Creation & SEO Optimizer** 📗 [01-01-05.md](0-PROJECTS/01-01-05.md)  
      (MCP, web scraping, keyword analysis, content planning, social media APIs, WordPress/Shopify)  
    - **Financial Trading Assistant** 📗 [01-01-06.md](0-PROJECTS/01-01-06.md)  
      (MCP, market data APIs, portfolio management, risk analysis, trading strategies, Bloomberg/Alpha Vantage)  
    - **Healthcare Documentation System** 📗 [01-01-07.md](0-PROJECTS/01-01-07.md)  
      (MCP, medical records, FHIR standards, patient data analysis, compliance monitoring, HIPAA)  
    - **IoT Device Management Platform** 📗 [01-01-08.md](0-PROJECTS/01-01-08.md)  
      (MCP, sensor networks, real-time monitoring, device control, MQTT, edge computing, AWS IoT)  
    - **Educational Learning Companion** 📗 [01-01-09.md](0-PROJECTS/01-01-09.md)  
      (MCP, adaptive learning, progress tracking, curriculum management, LMS integration, personalized tutoring)  
    - **Automated DevOps Pipeline Manager** 📗 [01-01-10.md](0-PROJECTS/01-01-10.md)  
      (MCP, CI/CD orchestration, infrastructure monitoring, deployment automation, Docker/Kubernetes, GitHub Actions)
    ---
    - **Smart City Traffic Optimization** 📗 [01-01-11.md](0-PROJECTS/01-01-11.md)  
      (MCP, traffic sensors, route planning, congestion prediction, smart signals, urban mobility, Google Maps API)
    - **AI-Powered Legal Research Assistant** 📗 [01-01-12.md](0-PROJECTS/01-01-12.md)  
      (MCP, case law databases, contract analysis, legal precedents, document review, LexisNexis/Westlaw)
    - **Virtual Reality Training Simulator** 📗 [01-01-13.md](0-PROJECTS/01-01-13.md)  
      (MCP, VR environments, skill assessment, immersive learning, Unity/Unreal Engine, haptic feedback)
    - **Blockchain Smart Contract Auditor** 📗 [01-01-14.md](0-PROJECTS/01-01-14.md)  
      (MCP, Solidity analysis, vulnerability detection, gas optimization, Web3 integration, Ethereum/Polygon)
    - **Mental Health Monitoring Platform** 📗 [01-01-15.md](0-PROJECTS/01-01-15.md)  
      (MCP, mood tracking, behavioral analysis, therapy recommendations, crisis intervention, wearable devices)
    - **Agricultural Crop Management System** 📗 [01-01-16.md](0-PROJECTS/01-01-16.md)  
      (MCP, satellite imagery, soil sensors, weather data, yield prediction, precision farming, John Deere APIs)
    - **Voice-Activated Home Automation** 📗 [01-01-17.md](0-PROJECTS/01-01-17.md)  
      (MCP, natural language processing, IoT device control, voice recognition, smart home protocols, Alexa/Google Assistant)
    - **Cybersecurity Threat Intelligence Hub** 📗 [01-01-18.md](0-PROJECTS/01-01-18.md)  
      (MCP, threat detection, vulnerability scanning, incident response, SIEM integration, malware analysis)
    - **Digital Art Generation Studio** 📗 [01-01-19.md](0-PROJECTS/01-01-19.md)  
      (MCP, style transfer, image synthesis, creative collaboration, NFT marketplace, Stable Diffusion/DALL-E)
    - **Supply Chain Optimization Engine** 📗 [01-01-20.md](0-PROJECTS/01-01-20.md)  
      (MCP, logistics tracking, inventory management, demand forecasting, supplier networks, SAP/Oracle integration)
    - **Scientific Research Data Analyzer** 📗 [01-01-21.md](0-PROJECTS/01-01-21.md)  
      (MCP, research papers, hypothesis generation, experiment design, statistical analysis, academic databases)
    - **Real Estate Market Predictor** 📗 [01-01-22.md](0-PROJECTS/01-01-22.md)  
      (MCP, property valuation, market trends, demographic analysis, investment opportunities, Zillow/Redfin APIs)
    - **Gaming AI Companion System** 📗 [01-01-23.md](0-PROJECTS/01-01-23.md)  
      (MCP, player behavior analysis, dynamic difficulty adjustment, personalized content, gaming APIs, Steam/Epic Games)
    - **Environmental Monitoring Network** 📗 [01-01-24.md](0-PROJECTS/01-01-24.md)  
      (MCP, air quality sensors, climate data, pollution tracking, environmental compliance, EPA databases)
    - **Language Learning Immersion Platform** 📗 [01-01-25.md](0-PROJECTS/01-01-25.md)  
      (MCP, speech recognition, pronunciation coaching, cultural context, adaptive curriculum, Duolingo/Babbel)
    - **3D Modeling Automation Tool** 📗 [01-01-26.md](0-PROJECTS/01-01-26.md)  
      (MCP, CAD integration, parametric design, manufacturing optimization, Blender/AutoCAD, 3D printing)
    - **Music Composition Assistant** 📗 [01-01-27.md](0-PROJECTS/01-01-27.md)  
      (MCP, audio analysis, melody generation, chord progressions, genre classification, Spotify/Apple Music APIs)
    - **Drone Fleet Management System** 📗 [01-01-28.md](0-PROJECTS/01-01-28.md)  
      (MCP, flight path optimization, autonomous navigation, payload delivery, airspace compliance, DJI/Parrot SDKs)
    - **Social Media Sentiment Analyzer** 📗 [01-01-29.md](0-PROJECTS/01-01-29.md)  
      (MCP, brand monitoring, trend analysis, influencer identification, crisis management, Twitter/Instagram APIs)
    - **Quantum Computing Simulator** 📗 [01-01-30.md](0-PROJECTS/01-01-30.md)  
      (MCP, quantum algorithms, circuit optimization, error correction, quantum machine learning, IBM Qiskit/Google Cirq)
  
  - <small>Project selection by [GPT](https://chatgpt.com/) (01-02)</small>
    - **Legal Document Assistant** 📗 [01-02-01.md](0-PROJECTS/01-02-01.md)  
      *(MCP, LangChain, Pinecone, Retrieval-Augmented Generation, PDF Parsing, GPT-4o, Memory, Agents)*  
      → Upload legal documents (PDF), extract clauses, answer legal queries using structured context.
    - **Enterprise Email Summarizer + Action Planner** 📗 [01-02-02.md](0-PROJECTS/01-02-02.md)  
      *(MCP, OpenAI GPT-4o, Context Windows, Task Memory, Prompt Engineering, LLM Agents)*  
      → Summarizes company emails, extracts action items, and plans tasks using MCP-based multi-turn context injection.
    - **Medical Diagnosis and Treatment Advisor** 📗 [01-02-03.md](0-PROJECTS/01-02-03.md)  
      *(MCP, MedPrompt, LangGraph, OpenAI GPT-4o, Patient History Memory, Retrieval, JSON Templates)*  
      → Ingests structured patient data, retrieves similar cases, and generates diagnosis with justification.
    - **Codebase Companion for Developers** 📗 [01-02-04.md](0-PROJECTS/01-02-04.md)  
      *(MCP, LangChain, GPT-4-turbo, VS Code Plugin, RAG, Embedding Index, Git Integration)*  
      → Interacts with local/global codebase using embeddings, tracks dev context with MCP layers for smarter Q&A.
    - **Multi-Document Research Agent** 📗 [01-02-05.md](0-PROJECTS/01-02-05.md)  
      *(MCP, LangChain Agents, Multi-Modal RAG, Context Pruning, Pinecone, OpenAI Functions)*  
      → Allows uploading and querying across books, PDFs, articles, and websites with persistent context memory.
    - **Customer Support Copilot (Multi-Turn Memory)** 📗 [01-02-06.md](0-PROJECTS/01-02-06.md)  
      *(MCP, LangGraph, GPT-4o, CRM Integration, Multi-Turn Chat Memory, Tools, Retrieval)*  
      → Handles long support sessions while maintaining context over past interactions using structured memory.
    - **Financial Forecasting Chatbot** 📗 [01-02-07.md](0-PROJECTS/01-02-07.md)  
      *(MCP, Time-Series Data, LangChain, LLM + Tools, CSV Ingestion, Pandas Agent, GPT-4o)*  
      → Uses historical data, injects context using MCP for accurate financial forecasting and Q&A.
    - **Multi-Agent Game Master (RPG)** 📗 [01-02-08.md](0-PROJECTS/01-02-08.md)  
      *(MCP, CrewAI, LangGraph, Character Memory, World Context, Function Calling, JSON Tools)*  
      → Game master for text-based RPGs using persistent character backstories and evolving world state.
    - **Scientific Paper Translator & Explainer** 📗 [01-02-09.md](0-PROJECTS/01-02-09.md)  
      *(MCP, Arxiv API, LangChain, Translator Tool, Citation Retriever, Summary Memory)*  
      → Parses scientific papers and translates/explains them using retrieved context, citations, and summary layers.
    - **Interactive Education Tutor (All Subjects)** 📗 [01-02-10.md](0-PROJECTS/01-02-10.md)  
      *(MCP, LangChain, Long-Term Memory, Concept Tracking, OpenAI GPT-4o, Gradio UI)*  
      → Personalized tutor using MCP to track concepts learned, context windows per subject, and goal tracking.
  
  - <small>Project selection by [DeepSeek](https://chat.deepseek.com/) (01-03)</small>
    - **MCP-Powered Dynamic Chatbot Framework** 📗 [01-03-01.md](0-PROJECTS/01-03-01.md)  
      (*MCP, Conversational AI, Context-Aware Dialogs, RAG, Fine-Tuning, LangChain*)
    - **Automated Legal Document Analyzer with MCP** 📗 [01-03-02.md](0-PROJECTS/01-03-02.md)  
      (*MCP, NLP, Contract Review, Semantic Search, Zero-Shot Classification, Haystack*)
    - **Personalized Learning Assistant Using MCP** 📗 [01-03-03.md](0-PROJECTS/01-03-03.md)  
      (*MCP, Adaptive Education, Knowledge Graphs, GPT-4 Integration, Feedback Loops*)
    - **MCP-Based Multi-Agent Simulation Environment** 📗 [01-03-04.md](0-PROJECTS/01-03-04.md)  
      (*MCP, Autonomous Agents, Game Theory, Emergent Behavior, LLM Orchestration*)
    - **Enterprise Knowledge Management with MCP** 📗 [01-03-05.md](0-PROJECTS/01-03-05.md)  
      (*MCP, Vector Databases, Document Embeddings, Retrieval-Augmented Generation, LlamaIndex*)
    - **AI-Powered Research Paper Summarization (MCP-Driven)** 📗 [01-03-06.md](0-PROJECTS/01-03-06.md)  
      (*MCP, Academic NLP, Abstractive Summarization, Citation Graphs, BERTopic*)
    - **Context-Aware Code Generation & Debugging Tool** 📗 [01-03-07.md](0-PROJECTS/01-03-07.md)  
      (*MCP, GitHub Copilot Alternative, AST Parsing, Error Correction, MCP-Guided Prompts*)
    - **MCP-Enhanced Multilingual Translation System** 📗 [01-03-08.md](0-PROJECTS/01-03-08.md)  
      (*MCP, Low-Resource Languages, Context Retention, LLM Alignment, OPUS Models*)
    - **Dynamic Storytelling & Interactive Fiction Platform** 📗 [01-03-09.md](0-PROJECTS/01-03-09.md)  
      (*MCP, Narrative Generation, User-Driven Plotlines, MCP Memory Layers, Stable Diffusion*)
    - **MCP-Optimized Fraud Detection for Financial Transactions** 📗 [01-03-10a.md](0-PROJECTS/01-03-10a.md), [01-03-10b.md](0-PROJECTS/01-03-10b.md), [01-03-10c.md](0-PROJECTS/01-03-10c.md)  
      (*MCP, Anomaly Detection, Graph Neural Networks, Explainable AI, Synthetic Data*)

- AI-LLM **Multi-Agent Systems**

  - <small>Project selection by [Claude](https://claude.ai/) (02-01)</small>
    - **Autonomous Trading Floor Simulation** 📗 [02-01-01.md](0-PROJECTS/02-01-01.md)  
      (Financial Markets, Risk Management, Portfolio Optimization, Real-time Decision Making, Market Analysis Agents)
    - **Smart City Traffic Management System** 📗 [02-01-02.md](0-PROJECTS/02-01-02.md)  
      (IoT Integration, Traffic Flow Optimization, Emergency Response Coordination, Predictive Analytics, Urban Planning Agents)
    - **Collaborative Research Assistant Network** 📗 [02-01-03.md](0-PROJECTS/02-01-03.md)  
      (Knowledge Discovery, Literature Review, Hypothesis Generation, Peer Review, Academic Research Coordination)
    - **Distributed Customer Service Ecosystem** 📗 [02-01-04.md](0-PROJECTS/02-01-04.md)  
      (Natural Language Processing, Sentiment Analysis, Ticket Routing, Escalation Management, Multi-channel Support)
    - **Supply Chain Optimization Platform** 📗 [02-01-05.md](0-PROJECTS/02-01-05.md)  
      (Logistics Coordination, Inventory Management, Demand Forecasting, Supplier Negotiation, Risk Assessment Agents)
    - **Personalized Learning Management System** 📗 [02-01-06.md](0-PROJECTS/02-01-06.md)  
      (Adaptive Learning, Content Curation, Student Assessment, Progress Tracking, Educational Resource Coordination)
    - **Healthcare Diagnosis and Treatment Planning** 📗 [02-01-07.md](0-PROJECTS/02-01-07.md)  
      (Medical Knowledge Integration, Symptom Analysis, Treatment Recommendation, Drug Interaction Checking, Clinical Decision Support)
    - **Real Estate Market Analysis and Recommendation Engine** 📗 [02-01-08.md](0-PROJECTS/02-01-08.md)  
      (Property Valuation, Market Trend Analysis, Investment Advisory, Neighborhood Assessment, Client Matching)
    - **Cybersecurity Threat Detection and Response Network** 📗 [02-01-09.md](0-PROJECTS/02-01-09.md)  
      (Anomaly Detection, Threat Intelligence, Incident Response, Vulnerability Assessment, Security Policy Enforcement)
    - **Content Creation and Marketing Automation Hub** 📗 [02-01-10.md](0-PROJECTS/02-01-10.md)  
      (Content Strategy, SEO Optimization, Social Media Management, Campaign Performance Analysis, Brand Voice Consistency)
    ---
    - **Autonomous Game Development Studio** 📗 [02-01-11.md](0-PROJECTS/02-01-11.md)  
      (Multi-Agent Systems, Procedural Content Generation, Game Balance Testing, Player Behavior Analysis, Narrative Generation, Art Asset Creation)
    - **Smart Agriculture Ecosystem** 📗 [02-01-12.md](0-PROJECTS/02-01-12.md)  
      (Multi-Agent Systems, Crop Monitoring, Weather Prediction, Pest Detection, Irrigation Optimization, Harvest Timing Coordination)
    - **Virtual Event Planning and Management Platform** 📗 [02-01-13.md](0-PROJECTS/02-01-13.md)  
      (Multi-Agent Systems, Venue Selection, Speaker Coordination, Attendee Engagement, Schedule Optimization, Technical Support Automation)
    - **Decentralized News Verification Network** 📗 [02-01-14.md](0-PROJECTS/02-01-14.md)  
      (Multi-Agent Systems, Fact-checking, Source Verification, Bias Detection, Information Credibility Scoring, Misinformation Flagging)
    - **Autonomous Legal Document Review System** 📗 [02-01-15.md](0-PROJECTS/02-01-15.md)  
      (Multi-Agent Systems, Contract Analysis, Compliance Checking, Legal Precedent Research, Risk Assessment, Document Generation)
    - **Smart Home Energy Management Collective** 📗 [02-01-16.md](0-PROJECTS/02-01-16.md)  
      (Multi-Agent Systems, Appliance Coordination, Energy Usage Optimization, Grid Integration, Renewable Energy Scheduling, Cost Minimization)
    - **Multi-Language Translation and Localization Hub** 📗 [02-01-17.md](0-PROJECTS/02-01-17.md)  
      (Multi-Agent Systems, Cultural Adaptation, Context Preservation, Quality Assurance, Domain Specialization, Real-time Collaboration)
    - **Autonomous Software Testing and QA Framework** 📗 [02-01-18.md](0-PROJECTS/02-01-18.md)  
      (Multi-Agent Systems, Test Case Generation, Bug Detection, Performance Testing, Security Vulnerability Scanning, Regression Testing)
    - **Virtual Personal Stylist Network** 📗 [02-01-19.md](0-PROJECTS/02-01-19.md)  
      (Multi-Agent Systems, Fashion Trend Analysis, Body Type Assessment, Color Coordination, Budget Optimization, Wardrobe Planning)
    - **Collaborative Scientific Experiment Design Platform** 📗 [02-01-20.md](0-PROJECTS/02-01-20.md)  
      (Multi-Agent Systems, Hypothesis Formation, Methodology Validation, Data Collection Planning, Statistical Analysis, Result Interpretation)
    - **Smart Manufacturing Quality Control System** 📗 [02-01-21.md](0-PROJECTS/02-01-21.md)  
      (Multi-Agent Systems, Defect Detection, Process Optimization, Predictive Maintenance, Resource Allocation, Production Scheduling)
    - **Autonomous Music Composition and Production Studio** 📗 [02-01-22.md](0-PROJECTS/02-01-22.md)  
      (Multi-Agent Systems, Melody Generation, Arrangement Creation, Audio Mixing, Genre Adaptation, Artist Collaboration)
    - **Virtual Mental Health Support Network**  
      (Multi-Agent Systems, Mood Tracking, Therapeutic Intervention, Crisis Detection, Resource Recommendation, Progress Monitoring)
    - **Decentralized Skill-Based Matchmaking Platform**  
      (Multi-Agent Systems, Competency Assessment, Project Matching, Team Formation, Performance Evaluation, Career Development)
    - **Smart Waste Management and Recycling System**  
      (Multi-Agent Systems, Collection Route Optimization, Material Sorting, Environmental Impact Assessment, Resource Recovery, Sustainability Tracking)
    - **Autonomous Film and Video Production Assistant**  
      (Multi-Agent Systems, Script Analysis, Scene Planning, Casting Suggestions, Post-production Coordination, Distribution Strategy)
    - **Multi-Agent Food Safety and Nutrition Platform**  
      (Multi-Agent Systems, Ingredient Analysis, Allergen Detection, Nutritional Optimization, Recipe Generation, Dietary Compliance)
    - **Virtual Reality Training Simulation Coordinator**  
      (Multi-Agent Systems, Scenario Generation, Performance Assessment, Skill Gap Analysis, Training Path Optimization, Certification Management)
    - **Autonomous Patent Research and Innovation Hub**  
      (Multi-Agent Systems, Prior Art Discovery, Invention Analysis, Patent Landscape Mapping, Innovation Opportunity Identification, IP Strategy Development)
    - **Smart Tourism and Travel Planning Ecosystem**  
      (Multi-Agent Systems, Destination Recommendation, Itinerary Optimization, Cultural Experience Curation, Budget Management, Real-time Travel Assistance)

  - <small>Project selection by [GPT](https://chatgpt.com/) (02-02)</small>
    - **Autonomous Research Team Assistant** 📗 [02-02-01.md](0-PROJECTS/02-02-01.md)  
      *(RAG, LangChain Agents, Pinecone, OpenAI GPT-4o, Task Decomposition, Memory, Tool Use, Web Search)*  
      → Multi-agent system that autonomously performs academic research, dividing tasks among agents (e.g., summarizer, fact-checker, citation retriever).
    - **AI Legal Document Analyzer** 📗 [02-02-02.md](0-PROJECTS/02-02-02.md)  
      *(LLM Agents, Document QA, Retrieval, Legal NLP, Memory Graphs, Semantic Search, Claude Sonnet, LangGraph)*  
      → Team of legal-AI agents parsing legal documents to extract clauses, detect risks, and recommend actions.
    - **Financial Market Analysis and Decision Support** 📗 [02-02-03.md](0-PROJECTS/02-02-03.md)  
      *(Multi-Agent Collaboration, Real-Time Data, Forecasting, News Scraping, Trading APIs, Agent Communication, OpenAI + Anthropic)*  
      → Agents specialize in trends, risk analysis, news sentiment, and portfolio balancing for human financial advisors.
    - **AI Game Master for Text-Based RPGs** 📗 [02-02-04.md](0-PROJECTS/02-02-04.md)  
      *(LLM Agents, Role-Based Reasoning, Memory, Game State Engine, Prompt Engineering, React Framework, Text-to-Action)*  
      → Game master and multiple character agents co-create an evolving narrative in an interactive RPG.
    - **Customer Support AI Team for E-Commerce** 📗 [02-02-05.md](0-PROJECTS/02-02-05.md)  
      *(LangChain, Vectorstore Retrieval, Agent Hand-off, LLM Tool Use, E-Commerce API, Multi-LLM Setup)*  
      → Agents for order tracking, returns, product info, and escalation working in coordination to resolve customer queries.
    - **Scientific Paper Reviewer Committee** 📗 [02-02-06.md](0-PROJECTS/02-02-06.md)  
      *(Autonomous Agents, Role-Playing Reviewers, LLM+Tool Use, Citation Verification, Markdown Reports, Critique Chains)*  
      → Reviewer agents simulate peer review with feedback, scoring, and revision suggestions for academic submissions.
    - **Smart City Infrastructure Management** 📗 [02-02-07.md](0-PROJECTS/02-02-07.md)  
      *(Sensor Data Simulation, AI Agents, Planning, Multi-Agent Communication, UMAP/GraphQL, Decision Making)*  
      → Agents oversee traffic, energy, and safety systems—collaborating to optimize city-wide resources and alerts.
    - **AI Debate Team** 📗 [02-02-08.md](0-PROJECTS/02-02-08.md)  
      *(Multi-Agent Prompt Chaining, Role-Specific LLMs, Memory, Persuasion Logic, GPT-4o, Claude, Open Debate APIs)*  
      → Pro and con agents prepare and argue on a topic in structured debate format with live audience Q&A moderation.
    - **AI Software Engineering Team** 📗 [02-02-09.md](0-PROJECTS/02-02-09.md)  
      *(LLM Agents for Code Writing, Code Review, Debugging, Test Generation, GitHub API, LangGraph, Code Interpreter)*  
      → Agents emulate developer roles (coder, reviewer, tester) working on a collaborative software project.
    - **Multi-Agent Travel Planner** 📗 [02-02-10.md](0-PROJECTS/02-02-10.md)  
      *(Agents for Flight, Hotel, Activity, Budget Planning, External API Calls, Natural Language Queries, User Preference Memory)*  
      → A team of AI agents that negotiate and compile optimal travel plans based on user goals and constraints.

  - <small>Project selection by [DeepSeek](https://chat.deepseek.com/) (02-03)</small>
    - **Autonomous Research Team** 📗 [02-03-01.md](0-PROJECTS/02-03-01.md)  
      (*LLM agents, collaborative research, semantic search, knowledge synthesis, Python, LangChain*)  
      A multi-agent system where AI researchers autonomously gather, analyze, and summarize academic papers.
    - **Smart Contract Auditor Squad** 📗 [02-03-02.md](0-PROJECTS/02-03-02.md)  
      (*Blockchain, Solidity, LLM agents, vulnerability detection, collaborative reporting*)  
      AI agents simulate a security team auditing smart contracts, identifying bugs and suggesting fixes.
    - **AI-Powered Debate Platform** 📗 [02-03-03.md](0-PROJECTS/02-03-03.md)  
      (*Multi-agent debate, argument synthesis, truth discovery, reinforcement learning*)  
      LLM agents with opposing viewpoints debate complex topics to uncover balanced conclusions.
    - **Dynamic Customer Support Swarm** 📗 [02-03-04.md](0-PROJECTS/02-03-04.md)  
      (*Multi-agent helpdesk, intent routing, live collaboration, CRM integration*)  
      AI agents work together to handle customer queries, escalating and sharing context as needed.
    - **Generative Game Design Collective** 📗 [02-03-05.md](0-PROJECTS/02-03-05.md)  
      (*Procedural content generation, LLM agents, Unity/Unreal, collaborative creativity*)  
      Autonomous AI designers, writers, and testers collaborate to create game narratives and mechanics.
    - **AI Venture Capital Simulator** 📗 [02-03-06.md](0-PROJECTS/02-03-06.md)  
      (*Market analysis, startup evaluation, multi-agent decision-making, financial modeling*)  
      A simulated VC firm where AI agents scout, evaluate, and negotiate startup investments.
    - **Epidemic Response Coordinator** 📗 [02-03-07.md](0-PROJECTS/02-03-07.md)  
      (*Public health, data analysis, policy simulation, multi-agent cooperation*)  
      AI agents model disease spread, evaluate interventions, and optimize resource allocation.
    - **Legal Case Strategizer** 📗 [02-03-08.md](0-PROJECTS/02-03-08.md)  
      (*Legal AI, precedent analysis, argument generation, multi-perspective evaluation*)  
      A virtual law firm where AI agents develop case strategies by analyzing past rulings.
    - **Personalized Education Collective** 📗 [02-03-09.md](0-PROJECTS/02-03-09.md)  
      (*Adaptive learning, LLM tutors, multi-agent curriculum design, knowledge assessment*)  
      AI tutors with different specialties collaborate to create personalized learning paths.
    - **AI Film Production Studio** 📗 [02-03-10.md](0-PROJECTS/02-03-10.md)  
      (*Multi-modal agents, script generation, storyboarding, virtual production*)  
      Autonomous AI teams handle screenwriting, casting, and editing for generated short films.

- AI-LLM **RAG** *(Retrieval-Augmented Generation)*

  - <small>Project selection by [Claude](https://claude.ai/) (03-01)</small>
    - **Intelligent Document Assistant** 📗 [03-01-01.md](0-PROJECTS/03-01-01.md)  
      (RAG, Document Processing, PDF Parsing, Vector Embeddings, Semantic Search, OpenAI GPT, Langchain, FAISS)
    - **Enterprise Knowledge Base Chatbot** 📗 [03-01-02.md](0-PROJECTS/03-01-02.md)  
      (RAG, Internal Documentation, Employee Support, Confluence Integration, Azure OpenAI, Pinecone, Slack Bot API)
    - **Legal Research and Case Analysis Platform** 📗 [03-01-03.md](0-PROJECTS/03-01-03.md)  
      (RAG, Legal Documents, Case Law, Regulatory Compliance, Elasticsearch, Claude API, Citation Extraction, NER)
    - **Medical Literature Review System** 📗 [03-01-04.md](0-PROJECTS/03-01-04.md)  
      (RAG, PubMed Integration, Clinical Papers, Drug Information, BioBERT, Chroma DB, Medical NLP, HIPAA Compliance)
    - **Code Documentation and Bug Resolution Assistant** 📗 [03-01-05.md](0-PROJECTS/03-01-05.md)  
      (RAG, GitHub Integration, Stack Overflow, Technical Documentation, Code Embeddings, Weaviate, GitHub Copilot API)
    - **Financial Research and Investment Analysis Tool** 📗 [03-01-06.md](0-PROJECTS/03-01-06.md)  
      (RAG, SEC Filings, Financial Reports, Market Data, Time-series Analysis, Qdrant, Bloomberg API, Risk Assessment)
    - **E-learning Content Recommendation Engine** 📗 [03-01-07.md](0-PROJECTS/03-01-07.md)  
      (RAG, Course Materials, Educational Videos, Learning Paths, Content Similarity, Milvus, YouTube API, Adaptive Learning)
    - **Customer Support Knowledge Assistant** 📗 [03-01-08.md](0-PROJECTS/03-01-08.md)  
      (RAG, FAQ Database, Ticket History, Product Manuals, Sentiment Analysis, Redis Search, Zendesk Integration, Multi-language)
    - **Scientific Research Paper Discovery Platform** 📗 [03-01-09.md](0-PROJECTS/03-01-09.md)  
      (RAG, ArXiv Papers, Research Abstracts, Citation Networks, SciBERT, Neo4j, Graph RAG, Collaborative Filtering)
    - **Real Estate Property Intelligence System** 📗 [03-01-10.md](0-PROJECTS/03-01-10.md)  
      (RAG, Property Listings, Market Reports, Neighborhood Data, Geospatial Search, PostGIS, Zillow API, Location Embeddings)
    ---
    11. **Smart Recipe and Nutrition Advisor**
    (RAG, Food Database, Dietary Restrictions, Meal Planning, Nutritional Analysis, Spoonacular API, Allergen Detection, Health Tracking)
    12. **Travel Planning and Destination Guide**
    (RAG, Trip Itineraries, Local Attractions, Weather Data, Cultural Information, Google Places API, Booking Integration, Multi-modal Search)
    13. **Personal Fashion and Style Consultant**
    (RAG, Clothing Catalogs, Style Trends, Body Type Analysis, Color Matching, Pinterest API, Image Recognition, Wardrobe Management)
    14. **Automotive Diagnostic and Repair Assistant**
    (RAG, Vehicle Manuals, Error Codes, Parts Catalogs, Maintenance Schedules, OBD-II Integration, 3D Model Visualization, Repair Videos)
    15. **Gaming Strategy and Walkthrough Helper**
    (RAG, Game Guides, Achievement Lists, Player Statistics, Twitch Integration, Steam API, Community Forums, Real-time Game Data)
    16. **Mental Health and Wellness Companion**
    (RAG, Therapy Techniques, Mindfulness Content, Mood Tracking, Crisis Resources, Psychology Research, SAMHSA Guidelines, Privacy Protection)
    17. **Home Improvement and DIY Project Guide**
    (RAG, Construction Tutorials, Tool Recommendations, Safety Guidelines, Material Cost Estimation, Home Depot API, Project Planning, Skill Assessment)
    18. **Language Learning Conversation Partner**
    (RAG, Grammar Rules, Cultural Context, Pronunciation Guides, Translation Memory, Google Translate API, Speech Recognition, Progress Tracking)
    19. **Pet Care and Veterinary Information System**
    (RAG, Animal Health Records, Breed Information, Vaccination Schedules, Emergency Care, Veterinary Literature, Pet Insurance, Behavioral Analysis)
    20. **Agricultural Crop Management Advisor**
    (RAG, Weather Patterns, Soil Data, Pest Control, Harvest Timing, Satellite Imagery, USDA Database, IoT Sensor Integration, Market Prices)
    21. **Music Discovery and Analysis Platform**
    (RAG, Artist Biographies, Album Reviews, Concert Information, Spotify API, Last.fm Data, Genre Classification, Lyric Analysis)
    22. **Social Media Content Strategy Assistant**
    (RAG, Trending Topics, Engagement Analytics, Content Templates, Hashtag Research, Instagram API, Influencer Data, Brand Guidelines)
    23. **Environmental Impact Assessment Tool**
    (RAG, Carbon Footprint Data, Sustainability Reports, Regulatory Standards, Climate Models, EPA Database, Green Technology, Impact Visualization)
    24. **Sports Analytics and Performance Tracker**
    (RAG, Player Statistics, Game Footage, Training Regimens, Injury Prevention, ESPN API, Biomechanical Analysis, Team Strategy)
    25. **Art History and Museum Guide**
    (RAG, Artwork Information, Artist Biographies, Exhibition Catalogs, Cultural Context, Museum APIs, Image Analysis, Virtual Tours)
    26. **Genealogy and Family History Research**
    (RAG, Historical Records, Census Data, DNA Analysis, Family Trees, Ancestry API, Geographic Migration, Cultural Heritage)
    27. **Cryptocurrency and Blockchain Intelligence**
    (RAG, Market Analysis, White Papers, Trading Patterns, DeFi Protocols, CoinGecko API, Smart Contract Analysis, Risk Assessment)
    28. **Event Planning and Venue Management**
    (RAG, Vendor Directories, Budget Planning, Guest Lists, Venue Specifications, Eventbrite API, Catering Menus, Timeline Management)
    29. **Podcast Content Discovery and Analysis**
    (RAG, Episode Transcripts, Host Information, Topic Categorization, Listener Reviews, Spotify Podcast API, Audio Processing, Recommendation Engine)
    30. **Urban Planning and Smart City Solutions**
    (RAG, Population Data, Infrastructure Reports, Traffic Patterns, Zoning Laws, City Planning Documents, GIS Integration, Public Transportation)

  - <small>Project selection by [GPT](https://chatgpt.com/) (03-02)</small>
    - **Legal Document Assistant** 📗 [03-02-01.md](0-PROJECTS/03-02-01.md)  
      *(RAG, LangChain, OpenAI GPT-4o, Pinecone, PDF Parsing, Prompt Engineering, Semantic Search)*  
      → AI assistant for answering legal questions from a repository of contracts and legal documents.
    - **Medical Research Summarizer** 📗 [03-02-02.md](0-PROJECTS/03-02-02.md)  
      *(RAG, LlamaIndex, PubMed API, HuggingFace Transformers, BioBERT, Vector Databases, Summarization)*  
      → Summarizes and answers questions using biomedical research papers and studies.
    - **Enterprise Knowledge Chatbot** 📗 [03-02-03.md](0-PROJECTS/03-02-03.md)  
      *(RAG, LangChain, Azure OpenAI, SharePoint, FAISS, Authentication, Multi-user Chat)*  
      → Internal chatbot that helps employees access company documents and SOPs securely.
    - **Academic Tutor for Students** 📗 [03-02-04.md](0-PROJECTS/03-02-04.md)  
      *(RAG, Claude Haiku, PDF + DOCX ingestion, Milvus, Streamlit, Flashcards, Quiz Generator)*  
      → Personalized study assistant that extracts content from textbooks and generates quizzes.
    - **Codebase Q&A Bot** 📗 [03-02-05.md](0-PROJECTS/03-02-05.md)  
      *(RAG, OpenAI GPT-4o, GitHub API, Docstring Parsing, Embeddings, VSCode Extension)*  
      → AI bot that answers technical questions from your codebase and documentation files.
    - **Multilingual Travel Assistant** 📗 [03-02-06.md](0-PROJECTS/03-02-06.md)  
      *(RAG, GPT-4 Turbo, Translation APIs, Pinecone, CSV/JSON data, Location-based RAG)*  
      → Answers tourist queries using local attraction guides and multilingual content.
    - **Historical Archive Explorer** 📗 [03-02-07.md](0-PROJECTS/03-02-07.md)  
      *(RAG, Llama 3, OCR, Newspapers Archive, Vector Search, Timeline Generation)*  
      → Allows users to explore and query historical texts, scanned papers, and rare books.
    - **Customer Support Ticket Analyzer** 📗 [03-02-08.md](0-PROJECTS/03-02-08.md)  
      *(RAG, LangChain, ElasticSearch, Email Parsing, OpenAI Functions, Retrieval Filters)*  
      → AI that assists support teams by answering and classifying tickets using historical data.
    - **Scientific Paper Explorer for Engineers** 📗 [03-02-09.md](0-PROJECTS/03-02-09.md)  
      *(RAG, LlamaIndex, ArXiv API, Embedding Comparison, Visual Search, Streamlit UI)*  
      → Engineers can query technical papers and extract specific methods or evaluations.
    - **Legal & Regulatory Compliance Checker** 📗 [03-02-10.md](0-PROJECTS/03-02-10.md)  
      *(RAG, Claude Sonnet, Regulatory PDFs, Pinecone Hybrid Search, LangChain Agents)*  
      → Helps companies verify if internal policies comply with external laws and standards.

  - <small>Project selection by [DeepSeek](https://chat.deepseek.com/) (03-03)</small>
    - **Enterprise Knowledge Assistant** 📗 [03-03-01.md](0-PROJECTS/03-03-01.md)  
      (*RAG, Internal docs, FAISS, LangChain, GPT-4, Slack integration*)  
      A chatbot that retrieves answers from company documents (PDFs, wikis) and provides context-aware responses via Slack.
    - **Legal Document Analyzer** 📗 [03-03-02.md](0-PROJECTS/03-03-02.md)  
      (*RAG, Case law retrieval, BERT embeddings, Weaviate, Mistral-7B*)  
      RAG system to search through legal precedents and generate summaries or arguments for lawyers.
    - **Medical Diagnosis Support Tool** 📗 [03-03-03.md](0-PROJECTS/03-03-03.md)  
      (*RAG, PubMed/clinical papers, BioBERT, Pinecone, Llama-3*)  
      Retrieves relevant medical research based on symptoms and suggests potential diagnoses with citations.
    - **Academic Research Synthesizer** 📗 [03-03-04.md](0-PROJECTS/03-03-04.md)  
      (*RAG, ArXiv/Semantic Scholar, Specter embeddings, ChromaDB, GPT-4o*)  
      Aggregates and explains research papers based on user queries with auto-generated literature reviews.
    - **Customer Support Automation** 📗 [03-03-05.md](0-PROJECTS/03-03-05.md)  
      (*RAG, Zendesk/HelpScout, Sentence-Transformers, Redis, Claude-3*)  
      Automates ticket responses by pulling from FAQs and past resolved tickets with citation links.
    - **Personalized Learning Tutor** 📗 [03-03-06.md](0-PROJECTS/03-03-06.md)  
      (*RAG, Textbook knowledge base, InstructorXL embeddings, Qdrant, Mixtral*)  
      Answers student questions with textbook excerpts and generates practice quizzes.
    - **Multilingual News Explorer** 📗 [03-03-07.md](0-PROJECTS/03-03-07.md)  
      (*RAG, News API, multilingual embeddings, Milvus, GPT-4 Turbo*)  
      Retrieves and summarizes news articles across languages with bias detection.
    - **Code Documentation Assistant** 📗 [03-03-08.md](0-PROJECTS/03-03-08.md)  
      (*RAG, GitHub repos, CodeBERT, LanceDB, StarCoder2*)  
      Answers programming questions by searching codebases and generating examples.
    - **E-commerce Product Recommender** 📗 [03-03-09.md](0-PROJECTS/03-03-09.md)  
      (*RAG, Product catalog, CLIP embeddings, Elasticsearch, Command R+*)  
      Conversational product search that understands "I need a gift for my hiking-loving mom".
    - **Historical Events Explorer** 📗 [03-03-10.md](0-PROJECTS/03-03-10.md)  
      (*RAG, Wikipedia/archive.org, Time-aware embeddings, PostgreSQL, Gemini Pro*)  
      Answers complex historical queries like "What caused the fall of the Berlin Wall?" with sourced timelines.

## 0. PROJECTS <small>(*Generated by [Claude Sonnet 4](https://www.anthropic.com/news/claude-4)*)</small>

- **AI-LLM**
  
  - <small>Project selection by [Claude](https://claude.ai/) (04-01)</small>
    1. **Intelligent Code Review Assistant** (code analysis, static analysis, automated feedback, pull request optimization)
    2. **Multi-Language Document Translator** (neural machine translation, context preservation, domain adaptation, batch processing)
    3. **Smart Content Summarization Engine** (extractive summarization, abstractive summarization, key point extraction, multi-document synthesis)
    4. **Conversational SQL Query Builder** (natural language to SQL, database schema understanding, query optimization, error handling)
    5. **Automated Technical Documentation Generator** (code-to-docs, API documentation, inline comments analysis, markdown generation)
    6. **Intelligent Email Response System** (email classification, sentiment analysis, auto-reply generation, context understanding)
    7. **Advanced Chatbot with RAG** (retrieval-augmented generation, vector databases, knowledge base integration, contextual responses)
    8. **Creative Writing Assistant** (story generation, character development, plot suggestions, writing style analysis)
    9. **Legal Document Analyzer** (contract analysis, clause extraction, compliance checking, risk assessment)
    10. **Medical Report Interpretation Tool** (clinical text processing, medical terminology extraction, diagnosis suggestions, report summarization)
    11. **Personalized Learning Content Generator** (adaptive content creation, learning path optimization, skill assessment, educational material synthesis)
    12. **Social Media Content Optimizer** (hashtag generation, engagement prediction, content scheduling, audience analysis)
    13. **Financial News Sentiment Analyzer** (market sentiment analysis, news impact prediction, financial entity recognition, trend analysis)
    14. **Multilingual Customer Support Bot** (cross-language support, cultural adaptation, escalation handling, knowledge base integration)
    15. **Academic Research Paper Summarizer** (scientific text processing, citation analysis, methodology extraction, research gap identification)
    16. **Product Description Generator** (e-commerce optimization, SEO-friendly content, feature highlighting, competitor analysis)
    17. **Meeting Minutes Transcription & Analysis** (speech-to-text integration, action items extraction, participant analysis, follow-up generation)
    18. **Automated Bug Report Triage System** (issue classification, priority assignment, developer matching, similar bug detection)
    19. **Interactive Story Generator** (branching narratives, user choice integration, character consistency, plot coherence)
    20. **Recipe Recommendation Engine** (dietary restrictions, ingredient substitution, nutritional analysis, cooking skill adaptation)
    21. **Job Description Optimizer** (bias detection, skill requirement analysis, market comparison, candidate matching)
    22. **Press Release Generator** (company news analysis, tone adjustment, target audience adaptation, distribution optimization)
    23. **Travel Itinerary Planner** (destination research, budget optimization, activity recommendations, local insights)
    24. **Code Comment Generator** (function documentation, parameter explanation, usage examples, best practices)
    25. **Podcast Episode Summarizer** (audio transcription, key topics extraction, timestamp generation, speaker identification)
    26. **Grant Proposal Writing Assistant** (funding opportunity matching, proposal structure optimization, success factor analysis, compliance checking)
    27. **Brand Voice Consistency Checker** (tone analysis, style guide enforcement, content alignment, brand personality matching)
    28. **Scientific Literature Review Generator** (research synthesis, citation management, methodology comparison, gap analysis)
    29. **Language Learning Conversation Partner** (adaptive difficulty, grammar correction, cultural context, pronunciation feedback)
    30. **Real Estate Listing Optimizer** (property description enhancement, market analysis, pricing suggestions, buyer persona targeting)

  - <small>Project selection by [GPT](https://chatgpt.com/) (04-02)</small>
    1. **Legal Document Analyzer** *(RAG, LangChain, OpenAI, PDF parsing, summarization)*
    2. **Medical Assistant Chatbot** *(LLM, multi-turn dialogue, symptom checker, MedGPT, HIPAA-safe)*
    3. **Code-to-Comment Generator** *(code generation, transformer models, docstring generation, Python, GitHub Copilot-style)*
    4. **Multilingual Translator with Grammar Tips** *(LLM, translation, grammar correction, token alignment, NLLB)*
    5. **AI Study Buddy for Exams** *(vectorstore, flashcard generation, RAG, Pinecone, OpenAI Embeddings)*
    6. **Voice-Enabled AI Interview Coach** *(speech-to-text, Whisper, LLM, feedback generation, HR GPT)*
    7. **LLM-Powered News Summarizer with Bias Detection** *(news scraping, sentiment analysis, summarization, political bias detection)*
    8. **LLM-Powered E-commerce Product Recommender** *(chatbot, recommendation, retrieval, user history, embeddings)*
    9. **Interactive Story Generator for Kids** *(creative writing, LLM, prompt engineering, visual storytelling, audio synthesis)*
    10. **AI Contract Reviewer** *(legal clauses extraction, semantic search, contract type detection, LLM)*
    11. **Financial Report Analyzer** *(XBRL, summarization, key figure extraction, LLM+charts)*
    12. **Real-time Meeting Minutes Generator** *(audio, Whisper, summarization, highlights, tagging)*
    13. **Smart Resume Builder with LLM** *(job matching, skills extraction, text generation, HR GPT)*
    14. **AI-Powered Tutor for Math** *(step-by-step reasoning, equation solving, Wolfram Alpha, MathGPT)*
    15. **Personal Life Coach Bot** *(LLM, memory, journaling, daily planning, goal tracking)*
    16. **LLM-Powered Agent for Research Paper Insights** *(PDF ingestion, vectorstore, question answering, citation extraction)*
    17. **Movie Script Analyzer and Generator** *(screenplay format, genre detection, story arc analysis, text generation)*
    18. **Customer Support Agent with Memory** *(LangChain, vector DB, multi-session memory, persona)*
    19. **AI Lawyer Assistant for Legal Advice Simulation** *(RAG, GPT-4, legal ontology, jurisdiction rules)*
    20. **AI Dungeon Master for Tabletop RPGs** *(world-building, rule-following, real-time story generation, agent-based)*
    21. **Academic Plagiarism Detector with Explanation** *(similarity check, AI detector, LLM explanation, cross-document search)*
    22. **Nutrition Advisor with Meal Generator** *(diet analysis, health LLM, food DB, OpenAI + vector search)*
    23. **AI-Powered Translator with Cultural Context** *(LLM, translation memory, tone detection, adaptation)*
    24. **Voice-controlled Home Automation Chatbot** *(LLM + IoT, natural commands, action-to-intent mapping)*
    25. **AI Fashion Stylist Bot** *(style detection, outfit generation, multimodal LLM, seasonal trends)*
    26. **AI-Powered Therapy Simulator** *(empathetic agents, LLM, memory, emotional feedback)*
    27. **AI Assistant for Scientific Paper Writing** *(text generation, latex support, citation generation)*
    28. **LLM-Powered Game Guide Chatbot** *(game wiki ingestion, walkthrough generation, strategy hints)*
    29. **AI-Powered CV Skill Gap Analyzer** *(resume parsing, skill prediction, upskilling suggestions)*
    30. **Interactive Data Explorer using LLM** *(data analysis, pandas agent, CSV QA, chart generation)*

  - <small>Project selection by [DeepSeek](https://chat.deepseek.com/) (04-03)</small>
    1. **AI Code Reviewer** (LLM, static analysis, GitHub integration)  
    2. **Medical Chatbot** (HIPAA compliance, symptom checker, RAG)  
    3. **Legal Document Summarizer** (NER, contract analysis, LangChain)  
    4. **AI Dungeon Master** (text adventure, GPT-4, interactive storytelling)  
    5. **Automated Technical Writer** (Markdown generation, docstrings, Sphinx)  
    6. **Multilingual Podcast Transcriber** (ASR, translation, Whisper)  
    7. **AI Interview Coach** (speech analysis, emotion detection, feedback)  
    8. **Financial Report Analyzer** (LLM agents, pandas, earnings calls)  
    9. **Personalized Learning Tutor** (adaptive learning, knowledge graphs)  
    10. **AI Meme Generator** (CLIP, diffusion models, cultural context)  
    11. **Automated Bug Triage** (JIRA integration, error classification)  
    12. **Scientific Paper Explorer** (arXiv API, knowledge extraction)  
    13. **AI Fashion Advisor** (computer vision, style transfer, e-commerce)  
    14. **Codebase Documentation Bot** (repo crawling, UML generation)  
    15. **AI Debate Simulator** (argument mining, pro/con generation)  
    16. **Automated Meeting Minutes** (diarization, agenda extraction)  
    17. **AI Music Composer** (MIDI generation, music theory)  
    18. **Ethical AI Auditor** (bias detection, model cards)  
    19. **Recipe Generator** (ingredient substitution, dietary constraints)  
    20. **AI Sports Analyst** (statistics prediction, commentary generation)  
    21. **Automated Poetry Workshop** (meter analysis, style imitation)  
    22. **AI Plant Doctor** (image classification, gardening advice)  
    23. **Privacy-Preserving Chat** (federated learning, on-device LLM)  
    24. **AI Career Counselor** (skills gap analysis, roadmap generation)  
    25. **Historical Figure Chatbot** (biographical RAG, period-accurate speech)  
    26. **Automated API Builder** (OpenAPI spec generation, FastAPI)  
    27. **AI Standup Comedy Writer** (joke structure, audience adaptation)  
    28. **3D Model Generator from Text** (NeRF, Blender integration)  
    29. **AI Town (Virtual Society)** (multi-agent systems, emergent behavior)  
    30. **Cognitive Behavioral Therapy Bot** (mood tracking, CBT techniques)  

- **AI-AGENTS**
  
  - <small>Project selection by [Claude](https://claude.ai/) (05-01)</small>
    1. **Autonomous Trading Agent** (market analysis, risk management, portfolio optimization, execution strategies)
    2. **Smart Home Automation Agent** (IoT integration, energy optimization, security monitoring, predictive maintenance)
    3. **Personal Finance Management Agent** (expense tracking, budget optimization, investment advice, financial goal planning)
    4. **Recruitment Screening Agent** (resume analysis, candidate matching, interview scheduling, skill assessment)
    5. **Supply Chain Optimization Agent** (inventory management, demand forecasting, supplier evaluation, logistics coordination)
    6. **Social Media Management Agent** (content scheduling, engagement monitoring, trend analysis, influencer identification)
    7. **Customer Service Escalation Agent** (issue classification, priority routing, resolution tracking, satisfaction monitoring)
    8. **E-commerce Price Monitoring Agent** (competitor analysis, dynamic pricing, market trends, profit optimization)
    9. **Content Moderation Agent** (spam detection, harmful content filtering, community guidelines enforcement, user behavior analysis)
    10. **Research Data Collection Agent** (web scraping, data validation, source verification, automated reporting)
    11. **IT Infrastructure Monitoring Agent** (system health monitoring, anomaly detection, automated troubleshooting, performance optimization)
    12. **Learning Path Recommendation Agent** (skill gap analysis, course suggestions, progress tracking, adaptive learning)
    13. **Investment Portfolio Rebalancing Agent** (risk assessment, asset allocation, market timing, tax optimization)
    14. **Medical Appointment Scheduling Agent** (calendar integration, provider matching, insurance verification, reminder systems)
    15. **Content Creation Workflow Agent** (topic research, outline generation, fact-checking, publication scheduling)
    16. **Email Marketing Campaign Agent** (audience segmentation, A/B testing, send time optimization, performance analysis)
    17. **Fraud Detection Agent** (transaction monitoring, pattern recognition, risk scoring, alert generation)
    18. **Project Management Assistant Agent** (task prioritization, resource allocation, deadline tracking, team coordination)
    19. **SEO Optimization Agent** (keyword research, content analysis, backlink monitoring, ranking optimization)
    20. **Inventory Replenishment Agent** (stock level monitoring, supplier coordination, demand prediction, cost optimization)
    21. **Quality Assurance Testing Agent** (test case generation, bug detection, regression testing, performance monitoring)
    22. **Lead Generation Agent** (prospect identification, contact enrichment, qualification scoring, outreach automation)
    23. **Event Planning Coordination Agent** (venue booking, vendor management, guest coordination, timeline optimization)
    24. **Data Pipeline Monitoring Agent** (ETL process monitoring, data quality checks, error handling, performance optimization)
    25. **Competitive Intelligence Agent** (market research, competitor monitoring, pricing analysis, feature comparison)
    26. **Knowledge Base Maintenance Agent** (content updates, accuracy verification, user feedback integration, search optimization)
    27. **Backup and Recovery Agent** (automated backups, disaster recovery, data integrity checks, recovery testing)
    28. **Compliance Monitoring Agent** (regulatory tracking, policy enforcement, audit preparation, risk assessment)
    29. **Network Security Agent** (threat detection, vulnerability scanning, incident response, security policy enforcement)
    30. **Customer Onboarding Agent** (welcome sequences, document collection, account setup, training delivery)

  - <small>Project selection by [GPT](https://chatgpt.com/) (05-02)</small>
    1. **Personal Productivity Agent** (calendar integration, task management, RAG)
    2. **Travel Planning Agent** (web scraping, itinerary generation, real-time API)
    3. **Medical Diagnosis Assistant** (symptom checker, LLM reasoning, medical RAG)
    4. **Legal Research Agent** (legal document parsing, citation, retrieval augmentation)
    5. **Code Debugging Agent** (multi-agent, static analysis, test generation)
    6. **AI Tutor Agent** (curriculum planning, question answering, memory)
    7. **Game Strategy Agent** (multi-agent reinforcement learning, game state analysis)
    8. **Financial Advisor Agent** (LLM + time series, budget optimization, OpenBB)
    9. **Social Media Manager Agent** (content generation, scheduling, sentiment tracking)
    10. **Language Learning Agent** (translation, spaced repetition, grammar correction)
    11. **Virtual Psychologist Agent** (empathetic response, active memory, GPT + sentiment)
    12. **Customer Service Chatbot Agent** (retrieval, multi-turn, tool use)
    13. **News Aggregator & Summarizer Agent** (web browsing, summarization, RAG)
    14. **Technical Documentation Generator Agent** (code parsing, explanation, summarization)
    15. **Voice-Controlled Home Automation Agent** (speech-to-text, API control, IoT)
    16. **Fitness & Nutrition Coach Agent** (goal tracking, meal planning, wearable integration)
    17. **Resume & Job Matching Agent** (resume parser, job scraping, semantic match)
    18. **Scientific Research Assistant Agent** (semantic search, paper summarization, RAG)
    19. **Custom Legal Contract Generator Agent** (LLM + template filling, prompt chaining)
    20. **DevOps Deployment Agent** (CI/CD interaction, shell command execution, monitoring)
    21. **Academic Writing Assistant Agent** (citation insertion, coherence checking, grammar correction)
    22. **Image Captioning & Analysis Agent** (multimodal, vision + text, OpenAI GPT-4o)
    23. **Real-Time Translation Agent** (speech-to-text, translation, speech synthesis)
    24. **HR Interview Simulator Agent** (role-playing, dynamic question generation, evaluation)
    25. **E-commerce Product Recommender Agent** (collaborative filtering, LLM explanation)
    26. **Multiplayer Game Assistant Agent** (live communication, teammate feedback, LLM strategy)
    27. **Intelligent File Explorer Agent** (semantic search, tagging, embeddings)
    28. **AI Lawyer for Small Claims Agent** (form filling, evidence preparation, reasoning)
    29. **Content Repurposing Agent** (long video to short, summarization, multimodal processing)
    30. **AI Research Grant Writer Agent** (grant matching, drafting, RAG + financial planning)

  - <small>Project selection by [DeepSeek](https://chat.deepseek.com/) (05-03)</small>
    1. **Autonomous Research Assistant** (LLMs, RAG, PubMed/ArXiv API)  
    2. **Personal Health Coach Agent** (Wearable APIs, NLP, Reinforcement Learning)  
    3. **AI-Powered Legal Contract Analyzer** (Document AI, Clause Detection, NER)  
    4. **Multi-Agent Stock Trading System** (Time Series Forecasting, RL, Alpaca API)  
    5. **Smart Home Orchestrator** (IoT Control, Activity Prediction, Federated Learning)  
    6. **Automated Customer Support Agent** (Voice Synthesis, Intent Recognition, Live Chat API)  
    7. **AI Game Master for RPGs** (Procedural Generation, Narrative AI, Unity API)  
    8. **Cybersecurity Threat Hunter** (Anomaly Detection, SIEM Integration, MITRE ATT&CK)  
    9. **Personalized Education Tutor** (Knowledge Graphs, Adaptive Learning, EdTech APIs)  
    10. **Autonomous Drone Inspector** (Computer Vision, Path Planning, DJI SDK)  
    11. **AI Fashion Designer** (GANs, Style Transfer, E-commerce API)  
    12. **Automated Scientific Paper Reviewer** (Peer Review Analysis, Plagiarism Detection, LaTeX Parsing)  
    13. **AI Film Script Consultant** (Story Arc Analysis, Sentiment Tracking, Final Draft API)  
    14. **Smart Agriculture Agent** (Satellite Imagery, Soil Sensors, Predictive Analytics)  
    15. **AI Music Composer** (Music21, Transformers, MIDI Generation)  
    16. **Automated Video Editor Agent** (Scene Detection, Auto-Captioning, FFmpeg)  
    17. **AI Real Estate Negotiator** (Market Analysis, Persuasion Modeling, Zillow API)  
    18. **Cognitive Behavioral Therapy Coach** (Sentiment Analysis, Therapy Protocols, HIPAA Compliance)  
    19. **Autonomous Interview Practice Agent** (Speech Recognition, STAR Method, Feedback Generation)  
    20. **AI Urban Planning Assistant** (GIS Data, Traffic Simulation, CityEngine API)  
    21. **Automated Meme Generator** (Trend Analysis, Dank Learning, Reddit API)  
    22. **AI Dungeon Master for Board Games** (Rule Interpretation, Strategy Prediction, BoardGameGeek API)  
    23. **Personal Finance Optimizer** (Spending Pattern Recognition, Tax Law NLP, Plaid API)  
    24. **AI Sports Analyst** (Player Tracking, Performance Prediction, Stats API)  
    25. **Automated 3D Model Generator** (Point Cloud Processing, NeRF, Blender API)  
    26. **AI Newsroom Assistant** (Fact-Checking, Bias Detection, AP Style Guide)  
    27. **Autonomous Podcast Producer** (Voice Cloning, Topic Chaining, RSS Feed Generation)  
    28. **AI Car Mechanic Assistant** (OBD-II Data, Repair Manual NLP, Diagnostic Trees)  
    29. **Automated Accessibility Checker** (WCAG Compliance, Screen Reader Simulation, Axe Core)  
    30. **Multi-Language Cultural Bridge Agent** (Real-Time Translation, Cultural Norms DB, WebSockets) 