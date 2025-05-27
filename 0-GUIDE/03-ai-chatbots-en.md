# AI Chatbot Development Course Guide

Build custom AI chatbots with memory, personality, and contextual response capabilities using Python and modern AI frameworks.

## Learning Outcomes

After completing this course, you will be able to:
- Create custom AI assistants
- Integrate APIs and databases
- Work with data and RAG (Retrieval-Augmented Generation)
- Simulate personalities and decision-making logic

## Target Audience

### Traditional Developers
Integrate AI tools into applications using LangChain and GPT APIs, connect assistants to databases and external services.

### AI Specialists
Advanced tools for creating and fine-tuning AI assistants, vector databases, RAG, and knowledge graphs.

### Entrepreneurs
Improve customer experience with chatbots, deploy GPT assistants for support and communication.

---

## 01. Introduction to AI Assistants and Creating Your First GPT Assistant

Learn the fundamentals of AI assistants and build a basic conversational agent.

```python
import openai
from openai import OpenAI

class BasicGPTAssistant:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.conversation_history = []
    
    def chat(self, user_message):
        self.conversation_history.append({"role": "user", "content": user_message})
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=self.conversation_history,
            max_tokens=150
        )
        
        assistant_reply = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": assistant_reply})
        
        return assistant_reply

# Usage
assistant = BasicGPTAssistant("your-api-key")
reply = assistant.chat("Hello, how can you help me?")
print(reply)
```

## 02. Capabilities and Limitations of GPT Assistants

Understand strengths, weaknesses, and appropriate use cases for GPT assistants. Learn prompt engineering strategies.

```python
class PromptEngineer:
    def __init__(self):
        self.system_prompts = {
            "helpful": "You are a helpful assistant that provides clear, concise answers.",
            "creative": "You are a creative writing assistant that helps with storytelling.",
            "technical": "You are a technical expert who explains complex concepts simply."
        }
    
    def create_prompt(self, role, context, user_query):
        return f"""
        {self.system_prompts.get(role, self.system_prompts['helpful'])}
        
        Context: {context}
        
        User Query: {user_query}
        
        Please provide a response that is appropriate for the given context.
        """

# Usage
prompt_eng = PromptEngineer()
prompt = prompt_eng.create_prompt("technical", "Python programming", "How do loops work?")
```

## 03. Vector Databases and Their Applications

Implement vector storage and similarity search for document-based question answering using RAG.

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SimpleVectorDB:
    def __init__(self):
        self.documents = []
        self.vectors = None
        self.vectorizer = TfidfVectorizer()
    
    def add_documents(self, docs):
        self.documents.extend(docs)
        self.vectors = self.vectorizer.fit_transform(self.documents)
    
    def search(self, query, top_k=3):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = [(self.documents[i], similarities[i]) for i in top_indices]
        
        return results

# Usage
db = SimpleVectorDB()
db.add_documents([
    "Python is a programming language",
    "Machine learning uses algorithms",
    "Vectors represent data numerically"
])

results = db.search("What is Python?")
for doc, score in results:
    print(f"Score: {score:.3f} - {doc}")
```

## 04. Multi-Agent Orchestration with LangGraph

Create complex workflows with multiple AI agents using state management and decision logic.

```python
from enum import Enum
from typing import Dict, Any

class AgentState(Enum):
    ANALYZE = "analyze"
    RESEARCH = "research"
    RESPOND = "respond"

class MultiAgentOrchestrator:
    def __init__(self):
        self.agents = {
            "analyzer": self.analyze_query,
            "researcher": self.research_info,
            "responder": self.generate_response
        }
        self.state = {}
    
    def analyze_query(self, query):
        # Determine query type and complexity
        if "?" in query:
            return {"type": "question", "complexity": "medium", "next": "researcher"}
        return {"type": "statement", "complexity": "low", "next": "responder"}
    
    def research_info(self, context):
        # Simulate information gathering
        return {"research_data": f"Retrieved info about: {context['query']}", "next": "responder"}
    
    def generate_response(self, context):
        return f"Based on analysis: {context.get('research_data', 'Direct response')}"
    
    def process(self, query):
        self.state = {"query": query}
        
        # Start with analyzer
        analysis = self.agents["analyzer"](query)
        self.state.update(analysis)
        
        # Continue based on decision
        if analysis["next"] == "researcher":
            research = self.agents["researcher"](self.state)
            self.state.update(research)
        
        # Generate final response
        return self.agents["responder"](self.state)

# Usage
orchestrator = MultiAgentOrchestrator()
response = orchestrator.process("What is machine learning?")
print(response)
```

## 05. Advanced API Integration for Dynamic Responses

Connect your assistant to external APIs for real-time data and enhanced functionality.

```python
import requests
import os
from datetime import datetime

class APIIntegratedAssistant:
    def __init__(self):
        self.api_keys = {
            "weather": os.getenv("WEATHER_API_KEY"),
            "news": os.getenv("NEWS_API_KEY")
        }
    
    def get_weather(self, city):
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather"
            params = {
                "q": city,
                "appid": self.api_keys["weather"],
                "units": "metric"
            }
            response = requests.get(url, params=params)
            data = response.json()
            
            return f"Weather in {city}: {data['weather'][0]['description']}, {data['main']['temp']}°C"
        except Exception as e:
            return f"Weather data unavailable: {str(e)}"
    
    def get_news(self, topic):
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": topic,
                "apiKey": self.api_keys["news"],
                "pageSize": 3,
                "sortBy": "publishedAt"
            }
            response = requests.get(url, params=params)
            articles = response.json()["articles"]
            
            news_summary = f"Latest news about {topic}:\n"
            for article in articles:
                news_summary += f"- {article['title']}\n"
            
            return news_summary
        except Exception as e:
            return f"News data unavailable: {str(e)}"
    
    def process_request(self, user_input):
        if "weather" in user_input.lower():
            # Extract city name (simplified)
            city = user_input.split("weather")[-1].strip()
            return self.get_weather(city or "London")
        elif "news" in user_input.lower():
            topic = user_input.split("news about")[-1].strip()
            return self.get_news(topic or "technology")
        else:
            return "I can help with weather and news. Try asking about weather or news!"

# Usage
assistant = APIIntegratedAssistant()
response = assistant.process_request("What's the weather like?")
print(response)
```

## 06. Monitoring and Performance Optimization

Track chatbot performance and optimize responses using analytics and feedback loops.

```python
import time
import json
from collections import defaultdict
from datetime import datetime

class ChatbotMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.conversations = []
    
    def log_interaction(self, user_input, bot_response, response_time, user_satisfaction=None):
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "bot_response": bot_response,
            "response_time": response_time,
            "user_satisfaction": user_satisfaction,
            "input_length": len(user_input),
            "response_length": len(bot_response)
        }
        
        self.conversations.append(interaction)
        self.metrics["response_times"].append(response_time)
        self.metrics["input_lengths"].append(len(user_input))
        
        if user_satisfaction:
            self.metrics["satisfaction_scores"].append(user_satisfaction)
    
    def get_performance_report(self):
        if not self.conversations:
            return "No data available"
        
        avg_response_time = sum(self.metrics["response_times"]) / len(self.metrics["response_times"])
        avg_satisfaction = sum(self.metrics["satisfaction_scores"]) / len(self.metrics["satisfaction_scores"]) if self.metrics["satisfaction_scores"] else 0
        
        return {
            "total_interactions": len(self.conversations),
            "average_response_time": round(avg_response_time, 3),
            "average_satisfaction": round(avg_satisfaction, 2),
            "common_input_patterns": self.analyze_patterns()
        }
    
    def analyze_patterns(self):
        # Simple pattern analysis
        common_words = defaultdict(int)
        for conv in self.conversations:
            words = conv["user_input"].lower().split()
            for word in words:
                if len(word) > 3:  # Filter short words
                    common_words[word] += 1
        
        return dict(sorted(common_words.items(), key=lambda x: x[1], reverse=True)[:5])

# Usage with timing decorator
def monitor_response(monitor, user_input, bot_response_func):
    start_time = time.time()
    response = bot_response_func(user_input)
    end_time = time.time()
    
    monitor.log_interaction(user_input, response, end_time - start_time)
    return response

monitor = ChatbotMonitor()
```

## 07. Code Integration in GPT Assistant Responses

Enable your assistant to execute Python code for calculations, data processing, and dynamic content generation.

```python
import ast
import operator
import re
from io import StringIO
import sys

class CodeExecutorAssistant:
    def __init__(self):
        self.safe_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg
        }
    
    def safe_eval(self, expression):
        """Safely evaluate mathematical expressions"""
        try:
            node = ast.parse(expression, mode='eval')
            return self._eval_node(node.body)
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _eval_node(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self.safe_operators[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            return self.safe_operators[type(node.op)](operand)
        else:
            raise ValueError(f"Unsupported operation: {type(node)}")
    
    def execute_code_snippet(self, code):
        """Execute safe Python code snippets"""
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            # Only allow specific safe operations
            if any(dangerous in code for dangerous in ['import', 'open', 'exec', 'eval']):
                return "Error: Unsafe code detected"
            
            exec(code)
            result = captured_output.getvalue()
            return result if result else "Code executed successfully"
        except Exception as e:
            return f"Execution error: {str(e)}"
        finally:
            sys.stdout = old_stdout
    
    def process_query_with_code(self, user_query):
        # Detect calculation requests
        if re.search(r'calculate|compute|what is \d+', user_query, re.IGNORECASE):
            # Extract mathematical expression
            math_pattern = r'[\d+\-*/().\s]+'
            matches = re.findall(math_pattern, user_query)
            if matches:
                expression = matches[0].strip()
                result = self.safe_eval(expression)
                return f"Calculation result: {expression} = {result}"
        
        # Detect code execution requests
        code_match = re.search(r'```python\n(.*?)```', user_query, re.DOTALL)
        if code_match:
            code = code_match.group(1)
            result = self.execute_code_snippet(code)
            return f"Code execution result:\n{result}"
        
        return "I can help with calculations and safe code execution. Try asking me to calculate something!"

# Usage
assistant = CodeExecutorAssistant()
response = assistant.process_query_with_code("Calculate 15 * 8 + 32")
print(response)

# Code execution example
code_query = """
Can you run this code?
```python
for i in range(5):
    print(f"Number: {i}")
```
"""
print(assistant.process_query_with_code(code_query))
```

## 08. Customer Assistant Design and Configuration

Design and configure specialized assistants tailored to specific business needs and customer requirements.

```python
from dataclasses import dataclass
from typing import List, Dict, Any
import json

@dataclass
class AssistantPersonality:
    tone: str  # formal, casual, friendly, professional
    expertise_level: str  # beginner, intermediate, expert
    response_style: str  # concise, detailed, conversational
    brand_voice: str  # corporate, startup, personal

class CustomerAssistantBuilder:
    def __init__(self):
        self.configurations = {}
        self.knowledge_base = {}
        self.conversation_flows = {}
    
    def create_assistant_config(self, company_name: str, industry: str, 
                              personality: AssistantPersonality, 
                              capabilities: List[str]):
        config = {
            "company_name": company_name,
            "industry": industry,
            "personality": personality.__dict__,
            "capabilities": capabilities,
            "system_prompt": self._generate_system_prompt(company_name, industry, personality),
            "fallback_responses": self._generate_fallbacks(personality),
            "conversation_starters": self._generate_starters(industry)
        }
        
        self.configurations[company_name] = config
        return config
    
    def _generate_system_prompt(self, company: str, industry: str, personality: AssistantPersonality):
        return f"""
        You are a {personality.tone} assistant for {company}, a company in the {industry} industry.
        
        Your communication style is {personality.response_style} and your expertise level is {personality.expertise_level}.
        Brand voice: {personality.brand_voice}.
        
        Guidelines:
        - Always maintain professionalism while being {personality.tone}
        - Provide {personality.response_style} responses
        - If unsure, offer to connect the user with human support
        - Stay focused on {industry}-related topics
        - Represent {company}'s values and mission
        """
    
    def _generate_fallbacks(self, personality: AssistantPersonality):
        fallbacks = {
            "unknown_query": f"I don't have specific information about that. Let me connect you with our support team who can help better.",
            "technical_issue": f"For technical issues, I recommend contacting our technical support team directly.",
            "complaint": f"I understand your concern. Let me make sure this gets the attention it deserves from our customer service team."
        }
        
        if personality.tone == "casual":
            fallbacks["unknown_query"] = "Hmm, I'm not sure about that one! Let me get someone who knows more to help you out."
        
        return fallbacks
    
    def _generate_starters(self, industry: str):
        starters = {
            "technology": [
                "How can I help you with our tech solutions today?",
                "Looking for technical support or product information?",
                "What technology challenge can I help you solve?"
            ],
            "healthcare": [
                "How can I assist you with your healthcare needs?",
                "Do you have questions about our services or appointments?",
                "I'm here to help with your healthcare inquiries."
            ],
            "finance": [
                "How can I help you with your financial needs today?",
                "Do you have questions about our financial services?",
                "I'm here to assist with your banking or investment questions."
            ]
        }
        
        return starters.get(industry, ["How can I help you today?"])
    
    def add_knowledge_base(self, company: str, knowledge_items: Dict[str, str]):
        if company not in self.knowledge_base:
            self.knowledge_base[company] = {}
        self.knowledge_base[company].update(knowledge_items)
    
    def get_response(self, company: str, user_query: str):
        if company not in self.configurations:
            return "Assistant not configured for this company."
        
        config = self.configurations[company]
        knowledge = self.knowledge_base.get(company, {})
        
        # Simple keyword matching for demo
        query_lower = user_query.lower()
        for topic, info in knowledge.items():
            if topic.lower() in query_lower:
                return f"Based on our information: {info}"
        
        # Use fallback
        return config["fallback_responses"]["unknown_query"]

# Usage Example
personality = AssistantPersonality(
    tone="friendly",
    expertise_level="intermediate", 
    response_style="detailed",
    brand_voice="professional"
)

builder = CustomerAssistantBuilder()
config = builder.create_assistant_config(
    company_name="TechCorp Solutions",
    industry="technology",
    personality=personality,
    capabilities=["product_support", "technical_help", "billing_inquiries"]
)

# Add knowledge base
builder.add_knowledge_base("TechCorp Solutions", {
    "pricing": "Our plans start at $29/month for basic features.",
    "support": "Technical support is available 24/7 via chat, email, or phone.",
    "features": "We offer cloud storage, data analytics, and API integration."
})

response = builder.get_response("TechCorp Solutions", "What are your pricing options?")
print(response)
```

## 09. Testing and Optimization of Customer Assistants

Implement comprehensive testing frameworks and optimization strategies for production chatbots.

```python
import random
import statistics
from datetime import datetime, timedelta
from collections import defaultdict
import matplotlib.pyplot as plt

class ChatbotTester:
    def __init__(self):
        self.test_scenarios = []
        self.test_results = []
        self.performance_metrics = defaultdict(list)
    
    def add_test_scenario(self, scenario_name: str, test_inputs: List[str], 
                         expected_keywords: List[str], success_criteria: str):
        scenario = {
            "name": scenario_name,
            "inputs": test_inputs,
            "expected_keywords": expected_keywords,
            "success_criteria": success_criteria,
            "timestamp": datetime.now()
        }
        self.test_scenarios.append(scenario)
    
    def run_test_suite(self, chatbot_function):
        """Run all test scenarios against the chatbot"""
        results = []
        
        for scenario in self.test_scenarios:
            scenario_results = {
                "scenario_name": scenario["name"],
                "test_results": [],
                "success_rate": 0,
                "avg_response_time": 0
            }
            
            response_times = []
            successful_tests = 0
            
            for test_input in scenario["inputs"]:
                start_time = datetime.now()
                
                try:
                    response = chatbot_function(test_input)
                    end_time = datetime.now()
                    response_time = (end_time - start_time).total_seconds()
                    response_times.append(response_time)
                    
                    # Check if response contains expected keywords
                    keyword_matches = sum(1 for keyword in scenario["expected_keywords"] 
                                        if keyword.lower() in response.lower())
                    
                    success = keyword_matches >= len(scenario["expected_keywords"]) * 0.5
                    if success:
                        successful_tests += 1
                    
                    test_result = {
                        "input": test_input,
                        "output": response,
                        "response_time": response_time,
                        "keyword_matches": keyword_matches,
                        "success": success
                    }
                    
                    scenario_results["test_results"].append(test_result)
                    
                except Exception as e:
                    scenario_results["test_results"].append({
                        "input": test_input,
                        "output": f"Error: {str(e)}",
                        "response_time": 0,
                        "keyword_matches": 0,
                        "success": False
                    })
            
            scenario_results["success_rate"] = successful_tests / len(scenario["inputs"])
            scenario_results["avg_response_time"] = statistics.mean(response_times) if response_times else 0
            
            results.append(scenario_results)
        
        self.test_results = results
        return results
    
    def generate_optimization_report(self):
        """Generate actionable optimization recommendations"""
        if not self.test_results:
            return "No test results available. Run tests first."
        
        report = {
            "overall_performance": {},
            "problem_areas": [],
            "recommendations": []
        }
        
        # Calculate overall metrics
        all_success_rates = [result["success_rate"] for result in self.test_results]
        all_response_times = [result["avg_response_time"] for result in self.test_results]
        
        report["overall_performance"] = {
            "average_success_rate": statistics.mean(all_success_rates),
            "average_response_time": statistics.mean(all_response_times),
            "scenarios_tested": len(self.test_results)
        }
        
        # Identify problem areas
        for result in self.test_results:
            if result["success_rate"] < 0.7:
                report["problem_areas"].append({
                    "scenario": result["scenario_name"],
                    "success_rate": result["success_rate"],
                    "main_issues": self._analyze_failures(result["test_results"])
                })
        
        # Generate recommendations
        if report["overall_performance"]["average_success_rate"] < 0.8:
            report["recommendations"].append("Improve knowledge base coverage - success rate below 80%")
        
        if report["overall_performance"]["average_response_time"] > 2.0:
            report["recommendations"].append("Optimize response time - currently above 2 seconds")
        
        if len(report["problem_areas"]) > 0:
            report["recommendations"].append("Focus on problem scenarios with low success rates")
        
        return report
    
    def _analyze_failures(self, test_results):
        """Analyze common failure patterns"""
        failed_tests = [t for t in test_results if not t["success"]]
        
        if len(failed_tests) == 0:
            return []
        
        # Simple analysis of failure patterns
        common_issues = []
        error_count = sum(1 for t in failed_tests if "Error:" in t["output"])
        if error_count > len(failed_tests) * 0.3:
            common_issues.append("High error rate - check exception handling")
        
        short_responses = sum(1 for t in failed_tests if len(t["output"]) < 50)
        if short_responses > len(failed_tests) * 0.5:
            common_issues.append("Responses too brief - may lack sufficient information")
        
        return common_issues

class UserFeedbackCollector:
    def __init__(self):
        self.feedback_data = []
    
    def collect_feedback(self, conversation_id: str, user_rating: int, 
                        feedback_text: str, conversation_context: Dict):
        feedback = {
            "conversation_id": conversation_id,
            "timestamp": datetime.now(),
            "rating": user_rating,  # 1-5 scale
            "feedback_text": feedback_text,
            "context": conversation_context
        }
        self.feedback_data.append(feedback)
    
    def analyze_feedback_trends(self):
        if not self.feedback_data:
            return "No feedback data available"
        
        ratings = [f["rating"] for f in self.feedback_data]
        recent_feedback = [f for f in self.feedback_data 
                          if f["timestamp"] > datetime.now() - timedelta(days=7)]
        
        return {
            "average_rating": statistics.mean(ratings),
            "total_feedback_count": len(self.feedback_data),
            "recent_feedback_count": len(recent_feedback),
            "rating_distribution": {i: ratings.count(i) for i in range(1, 6)},
            "common_complaints": self._extract_common_themes([f["feedback_text"] for f in self.feedback_data if f["rating"] < 3])
        }
    
    def _extract_common_themes(self, negative_feedback):
        # Simple keyword frequency analysis
        all_text = " ".join(negative_feedback).lower()
        words = all_text.split()
        word_freq = defaultdict(int)
        
        for word in words:
            if len(word) > 4:  # Filter short words
                word_freq[word] += 1
        
        return dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5])

# Usage Example
tester = ChatbotTester()

# Add test scenarios
tester.add_test_scenario(
    scenario_name="Product Information",
    test_inputs=[
        "What products do you offer?",
        "Tell me about your services",
        "What can I buy from you?"
    ],
    expected_keywords=["product", "service", "offer", "available"],
    success_criteria="Response should mention products or services"
)

tester.add_test_scenario(
    scenario_name="Support Requests",
    test_inputs=[
        "I need help with my account",
        "How do I reset my password?",
        "I'm having technical issues"
    ],
    expected_keywords=["help", "support", "assist", "contact"],
    success_criteria="Response should offer help or support contact"
)

# Mock chatbot function for testing
def mock_chatbot(user_input):
    if "product" in user_input.lower():
        return "We offer various software products and cloud services to help your business."
    elif "help" in user_input.lower() or "support" in user_input.lower():
        return "I can help you with that! Please contact our support team for technical assistance."
    else:
        return "I'm here to help! What can I do for you today?"

# Run tests
results = tester.run_test_suite(mock_chatbot)
optimization_report = tester.generate_optimization_report()

print("Optimization Report:")
print(f"Overall Success Rate: {optimization_report['overall_performance']['average_success_rate']:.2%}")
print(f"Average Response Time: {optimization_report['overall_performance']['average_response_time']:.2f}s")
print(f"Recommendations: {optimization_report['recommendations']}")
```

## 10. Building Emotional Intelligence and Digital Twins

Create emotionally aware assistants that can detect, track, and respond to user emotional states.

```python
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np

@dataclass
class EmotionalState:
    valence: float  # -1 (negative) to 1 (positive)
    arousal: float  # 0 (calm) to 1 (excited)
    dominance: float  # 0 (submissive) to 1 (dominant)
    confidence: float  # 0 to 1 (how confident we are in this assessment)

class EmotionDetector:
    def __init__(self):
        self.emotion_keywords = {
            'joy': {'happy', 'excited', 'thrilled', 'delighted', 'pleased', 'great', 'awesome', 'wonderful'},
            'sadness': {'sad', 'depressed', 'unhappy', 'disappointed', 'down', 'upset', 'terrible', 'awful'},
            'anger': {'angry', 'mad', 'furious', 'irritated', 'annoyed', 'frustrated', 'rage', 'hate'},
            'fear': {'scared', 'afraid', 'worried', 'anxious', 'nervous', 'concerned', 'frightened', 'panic'},
            'surprise': {'surprised', 'shocked', 'amazed', 'astonished', 'unexpected', 'wow'},
            'disgust': {'disgusting', 'revolting', 'gross', 'sick', 'horrible', 'nasty'}
        }
        
        self.intensity_modifiers = {
            'very': 1.5, 'extremely': 2.0, 'really': 1.3, 'quite': 1.2,
            'somewhat': 0.7, 'a bit': 0.6, 'slightly': 0.5, 'totally': 1.8
        }
    
    def detect_emotion(self, text: str) -> Dict[str, float]:
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        emotion_scores = {emotion: 0.0 for emotion in self.emotion_keywords.keys()}
        
        for i, word in enumerate(words):
            for emotion, keywords in self.emotion_keywords.items():
                if word in keywords:
                    base_score = 1.0
                    
                    # Check for intensity modifiers before the emotion word
                    if i > 0 and words[i-1] in self.intensity_modifiers:
                        base_score *= self.intensity_modifiers[words[i-1]]
                    
                    # Check for negation
                    if i > 0 and words[i-1] in ['not', 'never', 'no', "don't", "won't"]:
                        base_score *= -0.5
                    elif i > 1 and words[i-2] in ['not', 'never', 'no']:
                        base_score *= -0.5
                    
                    emotion_scores[emotion] += base_score
        
        # Normalize scores
        max_score = max(emotion_scores.values()) if max(emotion_scores.values()) > 0 else 1
        return {emotion: score/max_score for emotion, score in emotion_scores.items()}
    
    def emotions_to_state(self, emotions: Dict[str, float]) -> EmotionalState:
        # Map emotions to dimensional model (valence, arousal, dominance)
        valence = (emotions['joy'] - emotions['sadness'] - emotions['anger'] - emotions['disgust']) / 2
        arousal = (emotions['anger'] + emotions['fear'] + emotions['surprise'] - emotions['sadness']) / 2
        dominance = (emotions['anger'] + emotions['joy'] - emotions['fear'] - emotions['sadness']) / 2
        
        # Calculate confidence based on strongest emotion
        confidence = max(emotions.values())
        
        return EmotionalState(
            valence=max(-1, min(1, valence)),
            arousal=max(0, min(1, arousal + 0.5)),
            dominance=max(0, min(1, dominance + 0.5)),
            confidence=confidence
        )

class DigitalTwin:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.emotional_history = []
        self.personality_traits = {
            'openness': 0.5,
            'conscientiousness': 0.5,
            'extraversion': 0.5,
            'agreeableness': 0.5,
            'neuroticism': 0.5
        }
        self.preferences = {}
        self.interaction_patterns = {
            'preferred_response_length': 'medium',
            'communication_style': 'neutral',
            'topic_interests': []
        }
    
    def update_emotional_state(self, new_state: EmotionalState, context: str):
        entry = {
            'timestamp': datetime.now(),
            'state': new_state,
            'context': context
        }
        self.emotional_history.append(entry)
        
        # Keep only recent history (last 50 interactions)
        if len(self.emotional_history) > 50:
            self.emotional_history = self.emotional_history[-50:]
        
        # Update personality traits based on patterns
        self._update_personality_inference()
    
    def _update_personality_inference(self):
        if len(self.emotional_history) < 5:
            return
        
        recent_states = [entry['state'] for entry in self.emotional_history[-10:]]
        
        # Simple personality inference
        avg_valence = np.mean([state.valence for state in recent_states])
        avg_arousal = np.mean([state.arousal for state in recent_states])
        valence_variance = np.var([state.valence for state in recent_states])
        
        # Update traits based on patterns
        if avg_valence > 0.2:
            self.personality_traits['extraversion'] = min(1.0, self.personality_traits['extraversion'] + 0.1)
        
        if valence_variance > 0.3:
            self.personality_traits['neuroticism'] = min(1.0, self.personality_traits['neuroticism'] + 0.1)
        
        if avg_arousal > 0.6:
            self.personality_traits['openness'] = min(1.0, self.personality_traits['openness'] + 0.05)
    
    def get_current_emotional_trend(self) -> Dict[str, float]:
        if len(self.emotional_history) < 3:
            return {'trend': 'neutral', 'stability': 1.0}
        
        recent_valences = [entry['state'].valence for entry in self.emotional_history[-5:]]
        
        if len(recent_valences) >= 2:
            trend = 'improving' if recent_valences[-1] > recent_valences[0] else 'declining'
            stability = 1.0 - np.var(recent_valences)
        else:
            trend = 'neutral'
            stability = 1.0
        
        return {'trend': trend, 'stability': max(0, stability)}
    
    def recommend_response_strategy(self) -> Dict[str, str]:
        if not self.emotional_history:
            return {'tone': 'neutral', 'length': 'medium', 'approach': 'informative'}
        
        current_state = self.emotional_history[-1]['state']
        trend = self.get_current_emotional_trend()
        
        # Determine response strategy based on emotional state
        if current_state.valence < -0.3:  # User seems upset
            return {
                'tone': 'empathetic',
                'length': 'concise',
                'approach': 'supportive',
                'suggestions': ['acknowledge_feelings', 'offer_help', 'be_patient']
            }
        elif current_state.valence > 0.3:  # User seems happy
            return {
                'tone': 'enthusiastic',
                'length': 'medium',
                'approach': 'engaging',
                'suggestions': ['match_energy', 'be_positive', 'encourage_further']
            }
        elif current_state.arousal > 0.7:  # User seems excited/agitated
            return {
                'tone': 'calm',
                'length': 'short',
                'approach': 'grounding',
                'suggestions': ['be_clear', 'structure_response', 'avoid_overwhelming']
            }
        else:  # Neutral state
            return {
                'tone': 'friendly',
                'length': 'medium',
                'approach': 'informative',
                'suggestions': ['be_helpful', 'ask_clarifying_questions']
            }

class EmpatheticAssistant:
    def __init__(self):
        self.emotion_detector = EmotionDetector()
        self.user_twins = {}  # user_id -> DigitalTwin
    
    def get_or_create_twin(self, user_id: str) -> DigitalTwin:
        if user_id not in self.user_twins:
            self.user_twins[user_id] = DigitalTwin(user_id)
        return self.user_twins[user_id]
    
    def process_message(self, user_id: str, message: str) -> str:
        # Detect emotions in the message
        emotions = self.emotion_detector.detect_emotion(message)
        emotional_state = self.emotion_detector.emotions_to_state(emotions)
        
        # Update user's digital twin
        twin = self.get_or_create_twin(user_id)
        twin.update_emotional_state(emotional_state, message)
        
        # Get response strategy
        strategy = twin.recommend_response_strategy()
        
        # Generate empathetic response
        return self._generate_empathetic_response(message, emotional_state, strategy)
    
    def _generate_empathetic_response(self, message: str, state: EmotionalState, strategy: Dict) -> str:
        base_response = "I understand you're reaching out about this."
        
        # Adjust response based on emotional state and strategy
        if strategy['approach'] == 'supportive':
            if state.valence < -0.5:
                base_response = "I can sense this is really bothering you, and I want to help."
            else:
                base_response = "I understand this is concerning you."
        
        elif strategy['approach'] == 'engaging':
            base_response = "That's great to hear! I'm excited to help you with this."
        
        elif strategy['approach'] == 'grounding':
            base_response = "Let me help you with this step by step."
        
        # Add appropriate follow-up based on message content
        if "help" in message.lower():
            base_response += " What specific assistance do you need?"
        elif "?" in message:
            base_response += " Let me address your question."
        else:
            base_response += " How can I best support you with this?"
        
        return base_response

# Usage Example
assistant = EmpatheticAssistant()

# Simulate conversation with emotional tracking
messages = [
    "I'm really frustrated with this software, it keeps crashing!",
    "Thanks for the help, that actually worked perfectly!",
    "I'm worried about the deadline, there's so much to do",
    "Great! That solution saved me so much time!"
]

user_id = "user_123"
for msg in messages:
    response = assistant.process_message(user_id, msg)
    print(f"User: {msg}")
    print(f"Assistant: {response}")
    print(f"Emotional trend: {assistant.user_twins[user_id].get_current_emotional_trend()}")
    print("---")
```

## 11. Future Development Planning and Advanced Applications

Plan your AI development roadmap and explore cutting-edge applications in conversational AI.

```python
from enum import Enum
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

class DevelopmentPhase(Enum):
    PROTOTYPE = "prototype"
    MVP = "mvp"  
    BETA = "beta"
    PRODUCTION = "production"
    SCALE = "scale"

class AICapability(Enum):
    BASIC_CHAT = "basic_chat"
    CONTEXT_MEMORY = "context_memory"
    API_INTEGRATION = "api_integration"
    MULTIMODAL = "multimodal"
    EMOTIONAL_AI = "emotional_ai"
    AGENT_ORCHESTRATION = "agent_orchestration"
    CUSTOM_MODELS = "custom_models"
    REAL_TIME_LEARNING = "real_time_learning"

@dataclass
class DevelopmentGoal:
    name: str
    description: str
    capabilities_required: List[AICapability]
    estimated_weeks: int
    prerequisites: List[str]
    success_metrics: List[str]

class AIProjectRoadmap:
    def __init__(self):
        self.goals = []
        self.current_capabilities = set()
        self.completed_goals = []
        self.technology_stack = {
            'frameworks': [],
            'apis': [],
            'databases': [],
            'deployment': []
        }
    
    def add_goal(self, goal: DevelopmentGoal):
        self.goals.append(goal)
    
    def set_current_capabilities(self, capabilities: List[AICapability]):
        self.current_capabilities = set(capabilities)
    
    def generate_learning_path(self) -> List[Dict]:
        """Generate a personalized learning path based on current skills and goals"""
        learning_path = []
        available_goals = self.goals.copy()
        
        while available_goals:
            # Find goals that can be started with current capabilities
            ready_goals = []
            for goal in available_goals:
                missing_capabilities = set(goal.capabilities_required) - self.current_capabilities
                if len(missing_capabilities) <= 2:  # Can start if missing ≤2 capabilities
                    ready_goals.append((goal, len(missing_capabilities)))
            
            if not ready_goals:
                # No goals are immediately accessible, suggest foundational learning
                learning_path.append({
                    'type': 'foundation',
                    'recommendation': 'Focus on building core capabilities first',
                    'suggested_capabilities': list(set.union(*[set(g.capabilities_required) for g in available_goals]) - self.current_capabilities)[:3]
                })
                break
            
            # Sort by fewest missing capabilities, then by estimated time
            ready_goals.sort(key=lambda x: (x[1], x[0].estimated_weeks))
            next_goal = ready_goals[0][0]
            
            learning_step = {
                'type': 'goal',
                'goal': next_goal,
                'missing_capabilities': list(set(next_goal.capabilities_required) - self.current_capabilities),
                'learning_resources': self._get_learning_resources(next_goal),
                'estimated_completion': datetime.now() + timedelta(weeks=next_goal.estimated_weeks)
            }
            
            learning_path.append(learning_step)
            available_goals.remove(next_goal)
            self.current_capabilities.update(next_goal.capabilities_required)
        
        return learning_path
    
    def _get_learning_resources(self, goal: DevelopmentGoal) -> Dict[str, List[str]]:
        """Recommend learning resources for a specific goal"""
        resources = {
            'documentation': [],
            'tutorials': [],
            'projects': [],
            'communities': []
        }
        
        capability_resources = {
            AICapability.BASIC_CHAT: {
                'documentation': ['OpenAI API docs', 'LangChain documentation'],
                'tutorials': ['Build your first chatbot', 'Prompt engineering basics'],
                'projects': ['Simple Q&A bot', 'FAQ assistant'],
                'communities': ['OpenAI Community', 'LangChain Discord']
            },
            AICapability.CONTEXT_MEMORY: {
                'documentation': ['Vector database guides', 'Embedding models'],
                'tutorials': ['RAG implementation', 'Memory management in chatbots'],
                'projects': ['Document Q&A system', 'Conversation memory bot'],
                'communities': ['Pinecone community', 'Weaviate Discord']
            },
            AICapability.API_INTEGRATION: {
                'documentation': ['REST API best practices', 'Authentication methods'],
                'tutorials': ['API integration patterns', 'Error handling strategies'],
                'projects': ['Weather bot', 'News aggregator bot'],
                'communities': ['API development forums', 'Stack Overflow']
            },
            AICapability.EMOTIONAL_AI: {
                'documentation': ['Sentiment analysis libraries', 'Emotion detection APIs'],
                'tutorials': ['Building empathetic AI', 'User state management'],
                'projects': ['Mood tracking bot', 'Therapeutic conversation AI'],
                'communities': ['AI Ethics groups', 'Psychology + AI forums']
            }
        }
        
        for capability in goal.capabilities_required:
            if capability in capability_resources:
                for resource_type, items in capability_resources[capability].items():
                    resources[resource_type].extend(items)
        
        return resources
    
    def track_progress(self, goal_name: str, completion_percentage: float, notes: str = ""):
        """Track progress on a specific goal"""
        for goal in self.goals:
            if goal.name == goal_name:
                progress_entry = {
                    'timestamp': datetime.now(),
                    'completion_percentage': completion_percentage,
                    'notes': notes
                }
                
                if not hasattr(goal, 'progress_history'):
                    goal.progress_history = []
                goal.progress_history.append(progress_entry)
                
                if completion_percentage >= 100:
                    self.completed_goals.append(goal)
                    self.current_capabilities.update(goal.capabilities_required)
                
                break
    
    def generate_project_ideas(self, industry: str = None, complexity: str = "medium") -> List[Dict]:
        """Generate AI project ideas based on current capabilities"""
        project_templates = {
            'beginner': [
                {
                    'name': 'Personal FAQ Bot',
                    'description': 'Create a bot that answers frequently asked questions about yourself or your business',
                    'required_capabilities': [AICapability.BASIC_CHAT],
                    'industries': ['general', 'personal', 'small_business']
                },
                {
                    'name': 'Simple Task Reminder',
                    'description': 'Build an AI assistant that helps track and remind about tasks',
                    'required_capabilities': [AICapability.BASIC_CHAT, AICapability.CONTEXT_MEMORY],
                    'industries': ['productivity', 'personal']
                }
            ],
            'medium': [
                {
                    'name': 'Customer Support Assistant',
                    'description': 'Intelligent customer service bot with knowledge base integration',
                    'required_capabilities': [AICapability.BASIC_CHAT, AICapability.CONTEXT_MEMORY, AICapability.API_INTEGRATION],
                    'industries': ['e-commerce', 'saas', 'service']
                },
                {
                    'name': 'Content Creation Assistant',
                    'description': 'AI that helps generate and optimize content for different platforms',
                    'required_capabilities': [AICapability.BASIC_CHAT, AICapability.MULTIMODAL],
                    'industries': ['marketing', 'media', 'education']
                },
                {
                    'name': 'Mental Health Companion',
                    'description': 'Empathetic AI that provides emotional support and wellness tracking',
                    'required_capabilities': [AICapability.EMOTIONAL_AI, AICapability.CONTEXT_MEMORY],
                    'industries': ['healthcare', 'wellness', 'therapy']
                }
            ],
            'advanced': [
                {
                    'name': 'Multi-Agent Research Assistant',
                    'description': 'Coordinated AI agents that can research, analyze, and present complex information',
                    'required_capabilities': [AICapability.AGENT_ORCHESTRATION, AICapability.API_INTEGRATION, AICapability.CONTEXT_MEMORY],
                    'industries': ['research', 'consulting', 'finance']
                },
                {
                    'name': 'Adaptive Learning Tutor',
                    'description': 'AI tutor that adapts teaching style based on student progress and preferences',
                    'required_capabilities': [AICapability.EMOTIONAL_AI, AICapability.REAL_TIME_LEARNING, AICapability.MULTIMODAL],
                    'industries': ['education', 'training', 'corporate']
                }
            ]
        }
        
        suitable_projects = []
        for project in project_templates.get(complexity, []):
            required_caps = set(project['required_capabilities'])
            if required_caps.issubset(self.current_capabilities):
                if not industry or industry in project['industries']:
                    suitable_projects.append(project)
        
        return suitable_projects

class AITechnologyStack:
    def __init__(self):
        self.recommended_stack = {
            'development_frameworks': {
                'langchain': {
                    'description': 'Framework for developing LLM applications',
                    'best_for': ['rapid_prototyping', 'chain_composition', 'agent_orchestration'],
                    'learning_curve': 'medium'
                },
                'langgraph': {
                    'description': 'Stateful multi-actor applications with LLMs',
                    'best_for': ['complex_workflows', 'multi_agent_systems'],
                    'learning_curve': 'high'
                },
                'openai_api': {
                    'description': 'Direct API access to GPT models',
                    'best_for': ['simple_integrations', 'cost_optimization'],
                    'learning_curve': 'low'
                }
            },
            'vector_databases': {
                'pinecone': {
                    'description': 'Managed vector database service',
                    'best_for': ['production_apps', 'scalability'],
                    'pricing': 'usage_based'
                },
                'chroma': {
                    'description': 'Open-source embedding database',
                    'best_for': ['local_development', 'cost_sensitive_projects'],
                    'pricing': 'free'
                },
                'weaviate': {
                    'description': 'Open-source vector search engine',
                    'best_for': ['hybrid_search', 'complex_schemas'],
                    'pricing': 'freemium'
                }
            },
            'monitoring_tools': {
                'langsmith': {
                    'description': 'LangChain debugging and monitoring platform',
                    'best_for': ['langchain_apps', 'prompt_optimization'],
                    'integration': 'native_langchain'
                },
                'weights_biases': {
                    'description': 'ML experiment tracking and monitoring',
                    'best_for': ['model_training', 'experiment_management'],
                    'integration': 'broad_ml_ecosystem'
                }
            },
            'deployment_platforms': {
                'streamlit': {
                    'description': 'Rapid web app development for ML/AI',
                    'best_for': ['prototypes', 'internal_tools', 'demos'],
                    'complexity': 'low'
                },
                'fastapi': {
                    'description': 'High-performance API framework',
                    'best_for': ['production_apis', 'integration_services'],
                    'complexity': 'medium'
                },
                'cloud_functions': {
                    'description': 'Serverless deployment (AWS Lambda, Google Cloud Functions)',
                    'best_for': ['scalable_apis', 'cost_optimization'],
                    'complexity': 'medium'
                }
            }
        }
    
    def recommend_stack(self, project_type: str, team_size: str, budget: str) -> Dict:
        recommendations = {
            'primary_framework': '',
            'vector_db': '',
            'monitoring': '',
            'deployment': '',
            'reasoning': []
        }
        
        # Framework recommendation logic
        if project_type in ['prototype', 'mvp'] and team_size == 'small':
            recommendations['primary_framework'] = 'openai_api'
            recommendations['reasoning'].append('OpenAI API for rapid development with small team')
        elif 'multi_agent' in project_type or 'complex' in project_type:
            recommendations['primary_framework'] = 'langgraph'
            recommendations['reasoning'].append('LangGraph for complex multi-agent workflows')
        else:
            recommendations['primary_framework'] = 'langchain'
            recommendations['reasoning'].append('LangChain for balanced functionality and ease of use')
        
        # Vector DB recommendation
        if budget == 'low' or project_type == 'prototype':
            recommendations['vector_db'] = 'chroma'
            recommendations['reasoning'].append('Chroma for cost-effective local development')
        elif 'production' in project_type:
            recommendations['vector_db'] = 'pinecone'
            recommendations['reasoning'].append('Pinecone for production scalability')
        else:
            recommendations['vector_db'] = 'weaviate'
            recommendations['reasoning'].append('Weaviate for flexible hybrid search capabilities')
        
        # Monitoring recommendation
        if recommendations['primary_framework'] == 'langchain':
            recommendations['monitoring'] = 'langsmith'
            recommendations['reasoning'].append('LangSmith for native LangChain integration')
        else:
            recommendations['monitoring'] = 'weights_biases'
            recommendations['reasoning'].append('Weights & Biases for comprehensive ML monitoring')
        
        # Deployment recommendation
        if project_type == 'prototype':
            recommendations['deployment'] = 'streamlit'
            recommendations['reasoning'].append('Streamlit for rapid prototyping and demos')
        elif 'api' in project_type or 'production' in project_type:
            recommendations['deployment'] = 'fastapi'
            recommendations['reasoning'].append('FastAPI for production-ready APIs')
        else:
            recommendations['deployment'] = 'cloud_functions'
            recommendations['reasoning'].append('Serverless for scalable, cost-effective deployment')
        
        return recommendations

# Usage Example - Create Development Roadmap
def create_ai_development_roadmap():
    roadmap = AIProjectRoadmap()
    
    # Define development goals
    goals = [
        DevelopmentGoal(
            name="Basic Chatbot MVP",
            description="Create a functional chatbot with basic conversation capabilities",
            capabilities_required=[AICapability.BASIC_CHAT],
            estimated_weeks=2,
            prerequisites=["Python basics", "API understanding"],
            success_metrics=["Handles 80% of basic queries", "Response time < 2s", "User satisfaction > 3.5/5"]
        ),
        DevelopmentGoal(
            name="Knowledge-Enhanced Assistant",
            description="Add document knowledge base and contextual memory",
            capabilities_required=[AICapability.BASIC_CHAT, AICapability.CONTEXT_MEMORY],
            estimated_weeks=3,
            prerequisites=["Completed Basic Chatbot MVP"],
            success_metrics=["Accurate responses from knowledge base", "Maintains conversation context", "Handles follow-up questions"]
        ),
        DevelopmentGoal(
            name="Integrated Business Assistant",
            description="Connect to external APIs and business systems",
            capabilities_required=[AICapability.BASIC_CHAT, AICapability.CONTEXT_MEMORY, AICapability.API_INTEGRATION],
            estimated_weeks=4,
            prerequisites=["Knowledge-Enhanced Assistant", "API development knowledge"],
            success_metrics=["Successfully integrates 3+ APIs", "Handles real-time data queries", "Error handling for API failures"]
        ),
        DevelopmentGoal(
            name="Empathetic AI Companion",
            description="Build emotionally intelligent assistant with user state tracking",
            capabilities_required=[AICapability.EMOTIONAL_AI, AICapability.CONTEXT_MEMORY],
            estimated_weeks=5,
            prerequisites=["Understanding of sentiment analysis", "Psychology basics"],
            success_metrics=["Detects user emotions accurately", "Adapts responses to emotional state", "Improves user engagement"]
        )
    ]
    
    for goal in goals:
        roadmap.add_goal(goal)
    
    # Set current capabilities (user starting point)
    roadmap.set_current_capabilities([AICapability.BASIC_CHAT])
    
    # Generate learning path
    learning_path = roadmap.generate_learning_path()
    
    # Get project recommendations
    projects = roadmap.generate_project_ideas(complexity="medium")
    
    # Get technology stack recommendations
    tech_stack = AITechnologyStack()
    stack_rec = tech_stack.recommend_stack("mvp", "small", "medium")
    
    return {
        'learning_path': learning_path,
        'project_ideas': projects,
        'tech_stack': stack_rec,
        'roadmap': roadmap
    }

# Generate comprehensive development plan
development_plan = create_ai_development_roadmap()

print("🚀 AI Development Roadmap Generated!")
print("\n📚 Learning Path:")
for i, step in enumerate(development_plan['learning_path'][:3]):  # Show first 3 steps
    if step['type'] == 'goal':
        print(f"{i+1}. {step['goal'].name} ({step['goal'].estimated_weeks} weeks)")
        print(f"   Missing skills: {', '.join(step['missing_capabilities'])}")
    else:
        print(f"{i+1}. Foundation Building")
        print(f"   Focus areas: {', '.join(step['suggested_capabilities'][:3])}")

print("\n💡 Recommended Project Ideas:")
for project in development_plan['project_ideas'][:2]:
    print(f"• {project['name']}: {project['description']}")

print("\n🛠️ Technology Stack Recommendations:")
stack = development_plan['tech_stack']
print(f"• Framework: {stack['primary_framework']}")
print(f"• Vector DB: {stack['vector_db']}")
print(f"• Deployment: {stack['deployment']}")

class CareerPathPlanner:
    def __init__(self):
        self.career_paths = {
            'ai_engineer': {
                'description': 'Build and deploy AI systems in production environments',
                'key_skills': ['Python', 'ML/DL', 'Cloud platforms', 'MLOps'],
                'typical_progression': ['Junior AI Engineer', 'AI Engineer', 'Senior AI Engineer', 'AI Architect'],
                'salary_range': '$80K - $200K+',
                'growth_outlook': 'Very High'
            },
            'conversational_ai_specialist': {
                'description': 'Specialize in chatbots, voice assistants, and conversational interfaces',
                'key_skills': ['NLP', 'Dialogue systems', 'UX design', 'Psychology'],
                'typical_progression': ['Chatbot Developer', 'Conversation Designer', 'Senior Conv AI Specialist', 'Head of AI Experience'],
                'salary_range': '$70K - $180K+',
                'growth_outlook': 'High'
            },
            'ai_product_manager': {
                'description': 'Guide AI product strategy and development',
                'key_skills': ['Product strategy', 'AI understanding', 'Data analysis', 'User research'],
                'typical_progression': ['AI PM', 'Senior AI PM', 'Principal PM', 'VP of AI Products'],
                'salary_range': '$90K - $250K+',
                'growth_outlook': 'Very High'
            },
            'ai_consultant': {
                'description': 'Help businesses implement AI solutions',
                'key_skills': ['Business strategy', 'AI technologies', 'Communication', 'Project management'],
                'typical_progression': ['AI Consultant', 'Senior Consultant', 'Principal', 'Partner/Founder'],
                'salary_range': '$60K - $300K+',
                'growth_outlook': 'High'
            }
        }
    
    def assess_fit(self, interests: List[str], current_skills: List[str]) -> Dict[str, float]:
        """Assess fit for different AI career paths"""
        scores = {}
        
        for path_name, path_info in self.career_paths.items():
            score = 0.0
            
            # Check skill alignment
            skill_matches = len(set(current_skills) & set(path_info['key_skills']))
            score += (skill_matches / len(path_info['key_skills'])) * 0.6
            
            # Check interest alignment (simplified)
            interest_keywords = {
                'ai_engineer': ['technical', 'coding', 'systems', 'backend'],
                'conversational_ai_specialist': ['conversation', 'user_experience', 'language', 'psychology'],
                'ai_product_manager': ['strategy', 'business', 'leadership', 'product'],
                'ai_consultant': ['business', 'strategy', 'communication', 'variety']
            }
            
            if path_name in interest_keywords:
                interest_matches = len(set(interests) & set(interest_keywords[path_name]))
                score += (interest_matches / len(interest_keywords[path_name])) * 0.4
            
            scores[path_name] = min(1.0, score)
        
        return scores

# Example usage for career planning
career_planner = CareerPathPlanner()
user_interests = ['technical', 'conversation', 'user_experience']
user_skills = ['Python', 'API development', 'basic ML']

career_scores = career_planner.assess_fit(user_interests, user_skills)
print("\n🎯 Career Path Assessment:")
for path, score in sorted(career_scores.items(), key=lambda x: x[1], reverse=True):
    path_info = career_planner.career_paths[path]
    print(f"• {path.replace('_', ' ').title()}: {score:.1%} match")
    print(f"  {path_info['description']}")
    print(f"  Salary: {path_info['salary_range']}")
    print()
```