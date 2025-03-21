import streamlit as st
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import pathlib

# Check if .env file exists and load environment variables
env_path = pathlib.Path('.env')
env_exists = env_path.exists()
if env_exists:
    load_dotenv()

# Set up Streamlit page
st.set_page_config(page_title="Travel Planning Crew", page_icon="✈️")
st.title("Travel Planning Assistant")

# Sidebar for API key
with st.sidebar:
    st.header("Configuration")
    
    # Get API key from environment if available
    default_api_key = os.getenv("OPENAI_API_KEY", "")
    
    # Show appropriate message based on whether .env file exists and has API key
    if env_exists and default_api_key:
        api_key_message = "API key loaded from .env file"
        st.success(api_key_message)
        api_key_placeholder = "••••••••" # Masked placeholder for security
    else:
        if env_exists:
            api_key_message = "No API key found in .env file. Please enter manually:"
        else:
            api_key_message = "No .env file found. Please enter your API key:"
        st.warning(api_key_message)
        api_key_placeholder = ""
    
    # API key input field
    api_key = st.text_input("OpenAI API Key", 
                           type="password",
                           value=default_api_key,
                           placeholder=api_key_placeholder)
    
    # Update environment variable with entered key
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    st.markdown("---")
    st.markdown("Built with CrewAI and Streamlit")

# Main form
destination = st.text_input("Where would you like to travel?", "Paris, France")
days = st.number_input("How many days is your trip?", min_value=1, max_value=30, value=3)
budget = st.number_input("What's your budget (USD)?", min_value=100, max_value=10000, value=1000)
interests = st.text_area("What are your interests?", "Art, history, local cuisine")

# Run the crew when button is clicked
if st.button("Plan My Trip"):
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    else:
        try:
            with st.spinner("The crew is planning your perfect trip..."):
                # Create LLM
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
                
                # Create agents
                destination_expert = Agent(
                    role="Destination Expert",
                    goal="Provide detailed information about travel destinations",
                    backstory="You have traveled to over 100 countries and know all the hidden gems.",
                    verbose=True,
                    llm=llm
                )
                
                itinerary_planner = Agent(
                    role="Itinerary Planner",
                    goal="Create optimal day-by-day travel plans",
                    backstory="You are excellent at organizing time and activities for the perfect trip experience.",
                    verbose=True,
                    llm=llm
                )
                
                budget_advisor = Agent(
                    role="Budget Advisor",
                    goal="Provide cost estimates and budget-friendly recommendations",
                    backstory="You know how to make the most of any travel budget and find great deals.",
                    verbose=True,
                    llm=llm
                )
                
                # Create tasks with expected_output field
                destination_research = Task(
                    description=f"Research {destination} and provide key information about the location, best time to visit, and main attractions based on interests: {interests}",
                    expected_output=f"A comprehensive guide about {destination} including key attractions, best time to visit, and recommendations based on the interests: {interests}",
                    agent=destination_expert
                )
                
                create_itinerary = Task(
                    description=f"Create a detailed {days}-day itinerary for {destination} considering these interests: {interests}. Include specific attractions, activities, and time management.",
                    expected_output=f"A detailed day-by-day itinerary for {days} days in {destination}, with specific attractions, activities, and timing for each day.",
                    agent=itinerary_planner,
                    context=[destination_research]
                )
                
                budget_analysis = Task(
                    description=f"Analyze the proposed itinerary and provide a budget breakdown for a ${budget} budget. Suggest cost-saving alternatives if needed.",
                    expected_output=f"A detailed budget breakdown for the itinerary, including estimated costs for accommodations, food, attractions, and transportation. Include cost-saving tips if the total exceeds ${budget}.",
                    agent=budget_advisor,
                    context=[create_itinerary]
                )
                
                # Create and run the crew
                crew = Crew(
                    agents=[destination_expert, itinerary_planner, budget_advisor],
                    tasks=[destination_research, create_itinerary, budget_analysis],
                    verbose=True
                )
                
                result = crew.kickoff()
                
                # Display results
                st.success("Trip planning complete!")
                st.markdown("## Your Travel Plan")
                st.markdown(result)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")