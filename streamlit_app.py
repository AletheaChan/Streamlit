import streamlit as st
import pandas as pd
import numpy as np
import PIL.Image
import snowflake.connector
import pydeck as pdk
import pickle
import requests
from urllib.error import URLError


tab1,tab2 = st.tabs(["tab1","tab2"])

with tab1:
# Define the app title and favicon
  st.title('How much can you make from the TastyBytes locations?')
  st.markdown("This tab allows you to make predictions on the price of a listing based on the neighbourhood and room type. The model used is a Random Forest Regressor trained on the Airbnb Singapore listings dataset.")
  st.write('Choose a Truck Brand Name, City, Truck Location and Time Frame to get the predicted sales.')

  bn_mapping = { "Cheeky Greek": 0,
                  "Guac n' Roll": 1,
                  "Smoky BBQ": 2,
                  "Peking Truck": 3,
                  "Tasty Tibs": 4,
                  "Better Off Bread": 5,
                  "The Mega Melt": 6,
                  "Le Coin des Crêpes": 7,
                  "The Mac Shack": 8,
                  "Nani's Kitchen": 9,
                  "Plant Palace": 10,
                  "Kitakata Ramen Bar": 11,
                  "Amped Up Franks": 12,
                  "Freezing Point": 13,
                  "Revenge of the Curds": 14 }

  ct_mapping = { 'San Mateo': 0, 'Seattle': 1, 'New York City': 2, 'Boston': 3, 'Denver':4 }

  def get_brandName():
      brandName = st.selectbox('Select a truck brand name', bn_mapping)
      return brandName
    
  def get_city():
      CITY = st.selectbox('Select a city', ct_mapping)
      return CITY

  # Define the user input fields
  bn_input = get_brandName()
  ct_input = get_city()
  
  # Map user inputs to integer encoding
  bn_int = bn_mapping[bn_input]
  ct_int = ct_mapping[ct_input]
  

with tab2:
  st.title('My Parents New Healthy Diner')

  st.header('Breakfast Favourites')
  st.text('🥣 Omega 3 & Blueberry Oatmeal')
  st.text('🥗 Kale, Spinach & Rocket Smoothie')
  st.text('🐔 Hard-Boiled Free-Range Egg')
  st.text('🥑🍞 Avocado Toast')
  
  # streamlit.header('🍌🥭 Build Your Own Fruit Smoothie 🥝🍇')
  # my_fruit_list = pandas.read_csv("https://uni-lab-files.s3.us-west-2.amazonaws.com/dabw/fruit_macros.txt")
  # my_fruit_list = my_fruit_list.set_index('Fruit')
  
  # # Interactive widget (Multi-select), A pick list to pick the fruit they want to include 
  # fruits_selected = streamlit.multiselect("Pick some fruits:", list(my_fruit_list.index),['Avocado', 'Strawberries'])
  # # Filtering table data
  # fruits_to_show = my_fruit_list.loc[fruits_selected]
  # # Table display
  # streamlit.dataframe(fruits_to_show)
  
  # # Repeatable code block
  # def get_fruityvice_data(this_fruit_choice):
  #   fruityvice_response = requests.get("https://fruityvice.com/api/fruit/"+this_fruit_choice)
  #   fruityvice_normalized = pandas.json_normalize(fruityvice_response.json())
  #   return fruityvice_normalized
  
  # # New Fruityvice API Response
  # streamlit.header("Fruityvice Fruit Advice!")
  # try:
  #   fruit_choice = streamlit.text_input('What fruit would you like information about?')
  #   if not fruit_choice:
  #     streamlit.error("Please select a fruit to get information.")
  #   else:
  #     back_from_function = get_fruityvice_data(fruit_choice)
  #     streamlit.dataframe(back_from_function)
   
  # except URLError as e:
  #   streamlit.error()
  
  # # Fruit Load List
  # streamlit.header("View Our Fruit List – Add Your Favourites!")
  # def get_fruit_load_list():
  #   with my_cnx.cursor() as my_cur:
  #     my_cur.execute("select * from fruit_load_list")
  #     return my_cur.fetchall()
  
  # # Button to load the fruit list
  # if streamlit.button('Get Fruit Load List'):
  #   my_cnx = snowflake.connector.connect(**streamlit.secrets["snowflake"])
  #   my_data_rows = get_fruit_load_list()
  #   my_cnx.close()
  #   streamlit.dataframe(my_data_rows)
  
  # # Allow end user to add fruit to the list
  # def insert_row_snowflake(new_fruit):
  #   with my_cnx.cursor() as my_cur:
  #     my_cur.execute("Insert into fruit_load_list values ('" + new_fruit + "')")
  #     return 'Thanks for adding '+new_fruit
    
  # add_my_fruit = streamlit.text_input('What fruit would you like to add?')
  # if streamlit.button('Add a Fruit to the List'):
  #   my_cnx = snowflake.connector.connect(**streamlit.secrets["snowflake"])
  #   back_from_funcction = insert_row_snowflake(add_my_fruit)
  #   streamlit.text(back_from_function)


