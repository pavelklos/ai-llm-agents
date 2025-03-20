import os
import streamlit as st
import autogen
from autogen import (
    AssistantAgent,
    UserProxyAgent,
    AfterWork,
    OnCondition,
    AfterWorkOption,
    initiate_swarm_chat,
    register_hand_off,
    SwarmResult
)
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# App title and configuration
st.set_page_config(page_title="Recipe Creator Swarm", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better UI
st.markdown("""
<style>
    .agent-response {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .final-result {
        background-color: #e6f3ff;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        border-left: 5px solid #4361ee;
    }
    .stProgress .st-bo {
        background-color: #4361ee;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.image("https://docs.ag2.ai/img/logo.svg", width=100)
    st.title("Recipe Creator Swarm")
    
    # API Key input
    api_key_option = st.radio("OpenAI API Key Source:", ["Use .env file", "Enter manually"])
    
    if api_key_option == "Enter manually":
        api_key = st.text_input("Enter OpenAI API Key:", type="password")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("No API key found in .env file!")
        else:
            st.success("API key loaded from .env file")
    
    # Model selection
    model = st.selectbox(
        "Select OpenAI Model:",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0
    )
    
    # Temperature slider
    temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    # Divider
    st.divider()
    
    # About section with expander
    with st.expander("About this app"):
        st.write("""
        This app demonstrates an AutoGen Swarm with three specialized agents:
        
        1. **Recipe Planner**: Plans the recipe structure and ingredients
        2. **Chef**: Creates detailed cooking instructions
        3. **Nutritionist**: Adds nutritional information and tips
        
        The agents work together in a coordinated workflow to create a complete recipe.
        """)
    
    # GitHub link
    st.markdown("[View source on GitHub](https://github.com/yourusername/recipe-swarm)")

# Main content area
st.title("🍳 Recipe Creator Swarm")
st.markdown("Enter a recipe idea and let our AI swarm create a complete recipe with instructions and nutritional information.")

# User input
recipe_idea = st.text_area("Recipe Idea:", placeholder="E.g., A healthy vegetarian pasta dish with seasonal vegetables", value="Healthy vegetarian pasta dish with seasonal vegetables")
cuisine_type = st.selectbox("Cuisine Type:", ["Italian", "Mexican", "Indian", "Chinese", "Mediterranean", "American", "Thai", "Japanese", "French", "Other"])
dietary_restrictions = st.multiselect("Dietary Restrictions:", ["Vegetarian", "Vegan", "Gluten-Free", "Dairy-Free", "Nut-Free", "Low-Carb", "Keto", "Paleo"])
cooking_time = st.slider("Maximum Cooking Time (minutes):", 10, 120, 30)

# Context variables
context = {
    "recipe_idea": recipe_idea,
    "cuisine_type": cuisine_type,
    "dietary_restrictions": dietary_restrictions,
    "cooking_time": cooking_time,
    "planner_response": "",
    "chef_response": "",
    "nutritionist_response": "",
    "final_recipe": ""
}

# Function to record planner's response
def record_planner_response(response: str, context_variables: dict) -> SwarmResult:
    """Record the recipe planner's response"""
    context_variables["planner_response"] = response
    return SwarmResult(context_variables=context_variables)

# Function to record chef's response
def record_chef_response(response: str, context_variables: dict) -> SwarmResult:
    """Record the chef's response"""
    context_variables["chef_response"] = response
    return SwarmResult(context_variables=context_variables)

# Function to record nutritionist's response and finalize recipe
def record_nutritionist_response(response: str, context_variables: dict) -> SwarmResult:
    """Record the nutritionist's response and create final recipe"""
    context_variables["nutritionist_response"] = response
    
    # Combine all responses into a final recipe
    final_recipe = f"""
    # {context_variables['recipe_idea']}
    
    ## Ingredients and Plan
    {context_variables['planner_response']}
    
    ## Cooking Instructions
    {context_variables['chef_response']}
    
    ## Nutritional Information
    {context_variables['nutritionist_response']}
    """
    
    context_variables["final_recipe"] = final_recipe
    return SwarmResult(context_variables=context_variables)

# Run the swarm when the user clicks the button
if st.button("Create Recipe", type="primary", disabled=not api_key or not recipe_idea):
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Configure LLM
    llm_config = {
        "api_type": "openai",
        "model": model,
        "temperature": temperature,
        "api_key": api_key
    }
    
    # Create the agents
    status_text.text("Creating agents...")
    progress_bar.progress(10)
    
    recipe_planner = AssistantAgent(
        name="RecipePlanner",
        system_message=f"""You are a Recipe Planner who specializes in planning recipes.
        Based on the user's idea, create a structured recipe plan with:
        1. A catchy title for the recipe
        2. A list of all required ingredients with quantities
        3. Any special equipment needed
        
        Consider these requirements:
        - Cuisine type: {cuisine_type}
        - Dietary restrictions: {', '.join(dietary_restrictions) if dietary_restrictions else 'None'}
        - Maximum cooking time: {cooking_time} minutes
        
        Be thorough but concise. Focus only on planning, not cooking instructions.
        """,
        llm_config=llm_config,
    )
    
    chef = AssistantAgent(
        name="Chef",
        system_message=f"""You are a professional Chef who creates detailed cooking instructions.
        Based on the Recipe Planner's ingredients and plan, create step-by-step cooking instructions that:
        1. Are clear and easy to follow
        2. Include cooking times and temperatures
        3. Mention techniques and tips for best results
        4. Can be completed within {cooking_time} minutes total
        
        Be thorough but practical. Focus only on the cooking process.
        """,
        llm_config=llm_config,
    )
    
    nutritionist = AssistantAgent(
        name="Nutritionist",
        system_message=f"""You are a Nutritionist who provides health information about recipes.
        Based on the recipe plan and cooking instructions, provide:
        1. Estimated nutritional information (calories, protein, carbs, fats)
        2. Health benefits of key ingredients
        3. Suggestions for healthy substitutions if applicable
        4. Serving suggestions and portion advice
        
        Consider these dietary restrictions: {', '.join(dietary_restrictions) if dietary_restrictions else 'None'}
        Be informative but concise. Focus only on nutritional aspects.
        """,
        llm_config=llm_config,
    )
    
    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        code_execution_config=False
    )
    
    # Register functions with agents
    status_text.text("Setting up agent functions...")
    progress_bar.progress(20)
    
    recipe_planner.register_function(
        function_map={"record_planner_response": record_planner_response}
    )
    
    chef.register_function(
        function_map={"record_chef_response": record_chef_response}
    )
    
    nutritionist.register_function(
        function_map={"record_nutritionist_response": record_nutritionist_response}
    )
    
    # Register handoffs
    status_text.text("Configuring agent workflow...")
    progress_bar.progress(30)

    register_hand_off(
        recipe_planner,
        [
            AfterWork(
                lambda context_vars, *args: recipe_planner.run_function(
                    function_name="record_planner_response",
                    arguments={"response": recipe_planner.last_message()["content"], "context_variables": context_vars}
                )
            ),
            OnCondition(chef, "Create cooking instructions based on this plan.")
        ]
    )

    register_hand_off(
        chef,
        [
            AfterWork(
                lambda context_vars, *args: chef.run_function(
                    function_name="record_chef_response",
                    arguments={"response": chef.last_message()["content"], "context_variables": context_vars}
                )
            ),
            OnCondition(nutritionist, "Provide nutritional information for this recipe.")
        ]
    )

    register_hand_off(
        nutritionist,
        [
            AfterWork(
                lambda context_vars, *args: nutritionist.run_function(
                    function_name="record_nutritionist_response",
                    arguments={"response": nutritionist.last_message()["content"], "context_variables": context_vars}
                )
            ),
            AfterWork(AfterWorkOption.TERMINATE)
        ]
    )
    
    # Start the swarm
    status_text.text("Starting the recipe creation swarm...")
    progress_bar.progress(40)
    
    prompt = f"""Create a recipe for: {recipe_idea}
    Cuisine type: {cuisine_type}
    Dietary restrictions: {', '.join(dietary_restrictions) if dietary_restrictions else 'None'}
    Maximum cooking time: {cooking_time} minutes
    """
    
    try:
        chat_result, updated_context, last_agent = initiate_swarm_chat(
            initial_agent=recipe_planner,
            agents=[recipe_planner, chef, nutritionist],
            messages=prompt,
            context_variables=context,
        )
        
        progress_bar.progress(100)
        status_text.text("Recipe created successfully!")
        
        # Display agent responses in tabs
        st.subheader("Agent Contributions")
        tabs = st.tabs(["Recipe Planner", "Chef", "Nutritionist", "Final Recipe"])
        
        with tabs[0]:
            st.markdown(f"<div class='agent-response'>{updated_context['planner_response']}</div>", unsafe_allow_html=True)
            
        with tabs[1]:
            st.markdown(f"<div class='agent-response'>{updated_context['chef_response']}</div>", unsafe_allow_html=True)
            
        with tabs[2]:
            st.markdown(f"<div class='agent-response'>{updated_context['nutritionist_response']}</div>", unsafe_allow_html=True)
            
        with tabs[3]:
            st.markdown(f"<div class='final-result'>{updated_context['final_recipe']}</div>", unsafe_allow_html=True)
            
            # Download button for the recipe
            st.download_button(
                label="Download Recipe",
                data=updated_context['final_recipe'],
                file_name=f"recipe_{cuisine_type.lower().replace(' ', '_')}.md",
                mime="text/markdown"
            )
            
            # Rating system
            st.subheader("Rate this recipe:")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                if st.button("⭐"):
                    st.session_state.rating = 1
            with col2:
                if st.button("⭐⭐"):
                    st.session_state.rating = 2
            with col3:
                if st.button("⭐⭐⭐"):
                    st.session_state.rating = 3
            with col4:
                if st.button("⭐⭐⭐⭐"):
                    st.session_state.rating = 4
            with col5:
                if st.button("⭐⭐⭐⭐⭐"):
                    st.session_state.rating = 5
            
            if "rating" in st.session_state:
                st.success(f"Thanks for rating this recipe {st.session_state.rating} stars!")
        
        # Chat history in expander
        with st.expander("View Full Agent Conversation"):
            for msg in chat_result.chat_history:
                if msg["role"] == "assistant":
                    st.markdown(f"**{msg.get('name', 'Assistant')}**: {msg['content']}")
                else:
                    st.markdown(f"**{msg['role'].title()}**: {msg['content']}")
                st.divider()
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)
else:
    # Display placeholder content when not running
    st.info("Enter your recipe idea and click 'Create Recipe' to start the AI swarm.")
    
    # Example recipe with tabs
    st.subheader("Example Recipe")
    example_tabs = st.tabs(["Recipe Planner", "Chef", "Nutritionist", "Final Recipe"])
    
    with example_tabs[0]:
        st.markdown("""
        <div class='agent-response'>
        # Mediterranean Vegetable Pasta
        
        ## Ingredients:
        - 8 oz whole wheat pasta
        - 1 zucchini, diced
        - 1 red bell pepper, sliced
        - 1 yellow bell pepper, sliced
        - 1 cup cherry tomatoes, halved
        - 1/4 cup kalamata olives, pitted and sliced
        - 3 cloves garlic, minced
        - 2 tbsp olive oil
        - 1 tsp dried oregano
        - 1/2 tsp red pepper flakes
        - 1/4 cup fresh basil, chopped
        - 2 tbsp fresh parsley, chopped
        - 2 oz feta cheese, crumbled (optional)
        - Salt and pepper to taste
        
        ## Equipment:
        - Large pot for pasta
        - Large skillet or wok
        - Cutting board and knife
        - Colander
        </div>
        """, unsafe_allow_html=True)
    
    with example_tabs[1]:
        st.markdown("""
        <div class='agent-response'>
        ## Cooking Instructions:
        
        1. Bring a large pot of salted water to a boil. Add pasta and cook according to package directions until al dente (approximately 8-10 minutes).
        
        2. While pasta is cooking, heat olive oil in a large skillet over medium-high heat.
        
        3. Add minced garlic and sauté for 30 seconds until fragrant.
        
        4. Add zucchini and bell peppers to the skillet. Cook for 5-6 minutes, stirring occasionally, until vegetables begin to soften.
        
        5. Add cherry tomatoes, olives, dried oregano, and red pepper flakes. Cook for another 2-3 minutes until tomatoes begin to burst.
        
        6. Drain pasta, reserving 1/4 cup of pasta water.
        
        7. Add pasta to the skillet with vegetables. Toss to combine. If mixture seems dry, add a splash of reserved pasta water.
        
        8. Remove from heat and stir in fresh basil and parsley.
        
        9. Season with salt and pepper to taste.
        
        10. Serve immediately, topped with crumbled feta cheese if desired.
        
        Total cooking time: Approximately 25 minutes
        </div>
        """, unsafe_allow_html=True)
    
    with example_tabs[2]:
        st.markdown("""
        <div class='agent-response'>
        ## Nutritional Information:
        
        **Per Serving (Makes 4 servings):**
        - Calories: ~320 per serving
        - Protein: 10g
        - Carbohydrates: 45g
        - Fiber: 7g
        - Fat: 12g (mostly healthy unsaturated fats)
        - Sodium: ~300mg (varies with added salt)
        
        **Health Benefits:**
        - Whole wheat pasta provides complex carbohydrates and additional fiber
        - Bell peppers are high in vitamins A and C
        - Olive oil contains heart-healthy monounsaturated fats
        - Tomatoes provide lycopene, an antioxidant
        - Herbs add flavor without sodium or calories
        
        **Substitution Options:**
        - For vegan diets: Omit feta cheese or replace with nutritional yeast
        - For gluten-free diets: Use gluten-free pasta made from rice, corn, or legumes
        - For lower-carb diets: Reduce pasta portion and increase vegetables
        
        **Serving Suggestions:**
        - Serve with a simple side salad for additional vegetables
        - Portion size is approximately 1.5 cups per serving
        - Leftovers can be enjoyed cold as a pasta salad
        </div>
        """, unsafe_allow_html=True)
    
    with example_tabs[3]:
        st.markdown("""
        <div class='final-result'>
        # Mediterranean Vegetable Pasta
        
        ## Ingredients and Plan
        # Mediterranean Vegetable Pasta
        
        ## Ingredients:
        - 8 oz whole wheat pasta
        - 1 zucchini, diced
        - 1 red bell pepper, sliced
        - 1 yellow bell pepper, sliced
        - 1 cup cherry tomatoes, halved
        - 1/4 cup kalamata olives, pitted and sliced
        - 3 cloves garlic, minced
        - 2 tbsp olive oil
        - 1 tsp dried oregano
        - 1/2 tsp red pepper flakes
        - 1/4 cup fresh basil, chopped
        - 2 tbsp fresh parsley, chopped
        - 2 oz feta cheese, crumbled (optional)
        - Salt and pepper to taste
        
        ## Equipment:
        - Large pot for pasta
        - Large skillet or wok
        - Cutting board and knife
        - Colander
        
        ## Cooking Instructions
        ## Cooking Instructions:
        
        1. Bring a large pot of salted water to a boil. Add pasta and cook according to package directions until al dente (approximately 8-10 minutes).
        
        2. While pasta is cooking, heat olive oil in a large skillet over medium-high heat.
        
        3. Add minced garlic and sauté for 30 seconds until fragrant.
        
        4. Add zucchini and bell peppers to the skillet. Cook for 5-6 minutes, stirring occasionally, until vegetables begin to soften.
        
        5. Add cherry tomatoes, olives, dried oregano, and red pepper flakes. Cook for another 2-3 minutes until tomatoes begin to burst.
        
        6. Drain pasta, reserving 1/4 cup of pasta water.
        
        7. Add pasta to the skillet with vegetables. Toss to combine. If mixture seems dry, add a splash of reserved pasta water.
        
        8. Remove from heat and stir in fresh basil and parsley.
        
        9. Season with salt and pepper to taste.
        
        10. Serve immediately, topped with crumbled feta cheese if desired.
        
        Total cooking time: Approximately 25 minutes
        
        ## Nutritional Information
        ## Nutritional Information:
        
        **Per Serving (Makes 4 servings):**
        - Calories: ~320 per serving
        - Protein: 10g
        - Carbohydrates: 45g
        - Fiber: 7g
        - Fat: 12g (mostly healthy unsaturated fats)
        - Sodium: ~300mg (varies with added salt)
        
        **Health Benefits:**
        - Whole wheat pasta provides complex carbohydrates and additional fiber
        - Bell peppers are high in vitamins A and C
        - Olive oil contains heart-healthy monounsaturated fats
        - Tomatoes provide lycopene, an antioxidant
        - Herbs add flavor without sodium or calories
        
        **Substitution Options:**
        - For vegan diets: Omit feta cheese or replace with nutritional yeast
        - For gluten-free diets: Use gluten-free pasta made from rice, corn, or legumes
        - For lower-carb diets: Reduce pasta portion and increase vegetables
        
        **Serving Suggestions:**
        - Serve with a simple side salad for additional vegetables
        - Portion size is approximately 1.5 cups per serving
        - Leftovers can be enjoyed cold as a pasta salad
        </div>
        """, unsafe_allow_html=True)
        
        # Example download button
        st.download_button(
            label="Download Example Recipe",
            data="# Mediterranean Vegetable Pasta\n\n[Example recipe content]",
            file_name="example_recipe.md",
            mime="text/markdown",
            disabled=True
        )

# Footer
st.markdown("---")
st.markdown("Created with ❤️ using AutoGen and Streamlit")
