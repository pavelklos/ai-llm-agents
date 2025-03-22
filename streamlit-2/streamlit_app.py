import streamlit as st

# Day 2 ------------------------------------------------------------------------
# [Building your first Streamlit app]

st.write('Hello world!')

# Day 3 ------------------------------------------------------------------------
# [st.button]

st.header('st.button')

if st.button('Say hello'):
     st.write('Why hello there')
else:
     st.write('Goodbye')

# Day 4 ------------------------------------------------------------------------
# [Building Streamlit apps with Ken Jee]
# - https://www.youtube.com/watch?v=Yk-unX4KnV4

# Day 5 ------------------------------------------------------------------------
# [st.write]

import numpy as np
import altair as alt
import pandas as pd
import streamlit as st

st.header('st.write')

st.write('Hello, *World!* :sunglasses:')

st.write(1234)

df = pd.DataFrame({
     'first column': [1, 2, 3, 4],
     'second column': [10, 20, 30, 40]
     })
st.write(df)

st.write('Below is a DataFrame:', df, 'Above is a dataframe.')

df2 = pd.DataFrame(
     np.random.randn(200, 3),
     columns=['a', 'b', 'c'])
c = alt.Chart(df2).mark_circle().encode(
     x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])
st.write(c)

# Day 6 ------------------------------------------------------------------------
# [Uploading your Streamlit app to GitHub]

# Day 7 ------------------------------------------------------------------------
# [Deploying your Streamlit app with Streamlit Community Cloud]

# Day 8 ------------------------------------------------------------------------
# [st.slider]

import streamlit as st
from datetime import time, datetime

st.header('st.slider')

st.subheader('Slider')

age = st.slider('How old are you?', 0, 130, 25)
st.write("I'm ", age, 'years old')

st.subheader('Range slider')

values = st.slider(
     'Select a range of values',
     0.0, 100.0, (25.0, 75.0))
st.write('Values:', values)

st.subheader('Range time slider')

appointment = st.slider(
     "Schedule your appointment:",
     value=(time(11, 30), time(12, 45)))
st.write("You're scheduled for:", appointment)

st.subheader('Datetime slider')

start_time = st.slider(
     "When do you start?",
     value=datetime(2020, 1, 1, 9, 30),
     format="MM/DD/YY - hh:mm")
st.write("Start time:", start_time)

# Day 9 ------------------------------------------------------------------------
# [st.line_chart]

import streamlit as st
import pandas as pd
import numpy as np

st.header('Line chart')

chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)

# Day 10 -----------------------------------------------------------------------
# [st.selectbox

import streamlit as st

st.header('st.selectbox')

option = st.selectbox(
     'What is your favorite color?',
     ('Blue', 'Red', 'Green'))

st.write('Your favorite color is ', option)

# Day 11 -----------------------------------------------------------------------
# [st.multiselect]

import streamlit as st

st.header('st.multiselect')

options = st.multiselect(
     'What are your favorite colors',
     ['Green', 'Yellow', 'Red', 'Blue'],
     ['Yellow', 'Red'])

st.write('You selected:', options)

# Day 12 -----------------------------------------------------------------------
# [st.checkbox]

import streamlit as st

st.header('st.checkbox')

st.write ('What would you like to order?')

icecream = st.checkbox('Ice cream')
coffee = st.checkbox('Coffee')
cola = st.checkbox('Cola')

if icecream:
     st.write("Great! Here's some more 🍦")

if coffee: 
     st.write("Okay, here's some coffee ☕")

if cola:
     st.write("Here you go 🥤")

# Day 13 -----------------------------------------------------------------------
# [Spin up a cloud development environment] with Gitpod

# Day 14 -----------------------------------------------------------------------
# [Streamlit Components]
# TODO: Check the code below

# import streamlit as st
# import pandas as pd
# import pandas_profiling
# from streamlit_pandas_profiling import st_profile_report

# st.header('`streamlit_pandas_profiling`')

# df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')

# pr = df.profile_report()
# st_profile_report(pr)

# Day 15 -----------------------------------------------------------------------
# [st.latex]

import streamlit as st

st.header('st.latex')

st.latex(r'''
     a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
     \sum_{k=0}^{n-1} ar^k =
     a \left(\frac{1-r^{n}}{1-r}\right)
     ''')

# Day 16 -----------------------------------------------------------------------
# [Customizing the theme of Streamlit apps]

import streamlit as st

st.title('Customizing theme of Streamlit')

st.write('Contents of the `.streamlit/config.toml` file of this app')

st.code("""
[theme]
primaryColor="#F39C12"
backgroundColor="#2E86C1"
secondaryBackgroundColor="#AED6F1"
textColor="#FFFFFF"
font="monospace"
""")

number = st.sidebar.slider('Select a number:', 0, 10, 5)
st.write('Selected number from slider widget is:', number)

url_1 = "https://docs.streamlit.io/library/advanced-features/theming"
url_2 = "https://htmlcolorcodes.com/"
st.write("Theming [link](%s)" % url_1)
st.markdown("HTML Color Codes [link](%s)" % url_2)

st.markdown("""
- [Theming](https://docs.streamlit.io/library/advanced-features/theming)
- [HTML Color Codes](https://htmlcolorcodes.com/)
""")

# Day 17 -----------------------------------------------------------------------
# [st.secrets]

# import streamlit as st

# st.title('st.secrets')

# st.write(st.secrets['message'])

# st.write(st.secrets['whitelist'])
# "sally" in st.secrets.whitelist

# st.write(st.secrets["database"]["user"])
# st.write(st.secrets.database.password)
# st.secrets["database"]["user"] == "your username"
# st.secrets.database.password == "your password"

# Day 18 -----------------------------------------------------------------------
# [st.file_uploader]

import streamlit as st
import pandas as pd

st.title('st.file_uploader')

st.subheader('Input CSV')
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  st.subheader('DataFrame')
  st.write(df)
  st.subheader('Descriptive Statistics')
  st.write(df.describe())
else:
  st.info('☝️ Upload a CSV file')

# Day 19 -----------------------------------------------------------------------
# [How to layout your Streamlit app]
# file [streamlit_app_layout.py]

# Day 20 -----------------------------------------------------------------------
# [Tech Twitter Space on What is Streamlit?]

# Day 21 -----------------------------------------------------------------------
# [st.progress]

import streamlit as st
import time

st.title('st.progress')

with st.expander('About this app'):
     st.write('You can now display the progress of your calculations in a Streamlit app with the `st.progress` command.')

# my_bar = st.progress(0)
progress_text = "Operation in progress. Please wait."
my_bar = st.progress(0, text=progress_text)



for percent_complete in range(100):
     # time.sleep(0.05)
     time.sleep(0.01)
     # my_bar.progress(percent_complete + 1)
     my_bar.progress(percent_complete + 1, text=progress_text)
time.sleep(1)
my_bar.empty()
st.balloons()

st.button("Rerun")

# Day 22 -----------------------------------------------------------------------
# [st.form]

import streamlit as st

st.title('st.form')

# Full example of using the with notation
st.header('1. Example of using `with` notation')
st.subheader('Coffee machine')

with st.form('my_form'):
    st.subheader('**Order your coffee**')

    # Input widgets
    coffee_bean_val = st.selectbox('Coffee bean', ['Arabica', 'Robusta'])
    coffee_roast_val = st.selectbox('Coffee roast', ['Light', 'Medium', 'Dark'])
    brewing_val = st.selectbox('Brewing method', ['Aeropress', 'Drip', 'French press', 'Moka pot', 'Siphon'])
    serving_type_val = st.selectbox('Serving format', ['Hot', 'Iced', 'Frappe'])
    milk_val = st.select_slider('Milk intensity', ['None', 'Low', 'Medium', 'High'])
    owncup_val = st.checkbox('Bring own cup')

    # Every form must have a submit button
    submitted = st.form_submit_button('Submit')

if submitted:
    st.markdown(f'''
        ☕ You have ordered:
        - Coffee bean: `{coffee_bean_val}`
        - Coffee roast: `{coffee_roast_val}`
        - Brewing: `{brewing_val}`
        - Serving type: `{serving_type_val}`
        - Milk: `{milk_val}`
        - Bring own cup: `{owncup_val}`
        ''')
else:
    st.write('☝️ Place your order!')


# Short example of using an object notation
st.header('2. Example of object notation')

form = st.form('my_form_2')
selected_val = form.slider('Select a value')
form.form_submit_button('Submit')

st.write('Selected value: ', selected_val)

# Day 23 -----------------------------------------------------------------------
# [st.experimental_get_query_params]

# import streamlit as st

# st.title('st.experimental_get_query_params')

# with st.expander('About this app'):
#   st.write("`st.experimental_get_query_params` allows the retrieval of query parameters directly from the URL of the user's browser.")

# # 1. Instructions
# st.header('1. Instructions')
# st.markdown('''
# In the above URL bar of your internet browser, append the following:
# `?firstname=Jack&surname=Beanstalk`
# after the base URL `http://share.streamlit.io/dataprofessor/st.experimental_get_query_params/`
# such that it becomes 
# `http://share.streamlit.io/dataprofessor/st.experimental_get_query_params/?firstname=Jack&surname=Beanstalk`
# ''')


# # 2. Contents of st.experimental_get_query_params
# st.header('2. Contents of st.experimental_get_query_params')
# st.write(st.experimental_get_query_params())


# # 3. Retrieving and displaying information from the URL
# st.header('3. Retrieving and displaying information from the URL')

# firstname = st.experimental_get_query_params()['firstname'][0]
# surname = st.experimental_get_query_params()['surname'][0]

# st.write(f'Hello **{firstname} {surname}**, how are you?')

# Day 24 -----------------------------------------------------------------------
# [st.cache]

# import streamlit as st
# import numpy as np
# import pandas as pd
# from time import time

# st.title('st.cache')

# # Using cache
# a0 = time()
# st.subheader('Using st.cache')

# @st.cache(suppress_st_warning=True)
# def load_data_a():
#   df = pd.DataFrame(
#     np.random.rand(2000000, 5),
#     columns=['a', 'b', 'c', 'd', 'e']
#   )
#   return df

# st.write(load_data_a())
# a1 = time()
# st.info(a1-a0)


# # Not using cache
# b0 = time()
# st.subheader('Not using st.cache')

# def load_data_b():
#   df = pd.DataFrame(
#     np.random.rand(2000000, 5),
#     columns=['a', 'b', 'c', 'd', 'e']
#   )
#   return df

# st.write(load_data_b())
# b1 = time()
# st.info(b1-b0)

# Day 25 -----------------------------------------------------------------------
# [st.session_state]

import streamlit as st

st.title('st.session_state')

def lbs_to_kg():
  st.session_state.kg = st.session_state.lbs/2.2046
def kg_to_lbs():
  st.session_state.lbs = st.session_state.kg*2.2046

st.header('Input')
col1, spacer, col2 = st.columns([2,1,2])
with col1:
  pounds = st.number_input("Pounds:", key = "lbs", on_change = lbs_to_kg)
with col2:
  kilogram = st.number_input("Kilograms:", key = "kg", on_change = kg_to_lbs)

st.header('Output')
st.write("st.session_state object:", st.session_state)

# Day 26 -----------------------------------------------------------------------
# [How to use API by building the Bored API app]

import streamlit as st
import requests

st.title('🏀 Bored API app')

st.markdown("""
- [The Bored API](https://bored.api.lewagon.com/)
- [Bored API Documentation](https://bored.api.lewagon.com/documentation)
""")

st.sidebar.header('Input')
selected_type = st.sidebar.selectbox('Select an activity type', ["education", "recreational", "social", "diy", "charity", "cooking", "relaxation", "music", "busywork"])

# suggested_activity_url = f'http://www.boredapi.com/api/activity?type={selected_type}'
suggested_activity_url = f'https://bored.api.lewagon.com/api/activity?type={selected_type}'
json_data = requests.get(suggested_activity_url)
suggested_activity = json_data.json()

c1, c2 = st.columns(2)
with c1:
  with st.expander('About this app'):
    st.write('Are you bored? The **Bored API app** provides suggestions on activities that you can do when you are bored. This app is powered by the Bored API.')
with c2:
  with st.expander('JSON data'):
    st.write(suggested_activity)

st.header('Suggested activity')
st.info(suggested_activity['activity'])

col1, col2, col3 = st.columns(3)
with col1:
  st.metric(label='Number of Participants', value=suggested_activity['participants'], delta='')
with col2:
  st.metric(label='Type of Activity', value=suggested_activity['type'].capitalize(), delta='')
with col3:
  st.metric(label='Price', value=suggested_activity['price'], delta='')

# Day 27 -----------------------------------------------------------------------
# [Build a draggable and resizable dashboard with Streamlit Elements]
# file [streamlit_app_dashboard.py]

# Day 28 -----------------------------------------------------------------------
# [streamlit-shap]
# file [streamlit_app_shap.py]

# Day 29 -----------------------------------------------------------------------
# [How to make a zero-shot learning text classifier using Hugging Face and Streamlit]
# file [streamlit_app_zero_shot_classifier.py]

# Day 30 -----------------------------------------------------------------------
# [The Art of Creating Streamlit Apps]
# file [streamlit_app_art.py]
