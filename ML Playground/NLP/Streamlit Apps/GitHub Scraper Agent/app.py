#################################### 
##         Import Libraries       ##
####################################

import streamlit as st
from main import get_data

#################################### 
##             MAIN               ##
####################################

# Streamlit app title
st.title("Github Scraper")
# Username input
user_name = st.text_input('Enter Github Username')

# Display user information
if user_name != '':
  try:
    # Getting user data
    user_info, repos_info = get_data(user_name)
    # For each record of user information create a subheader and print the user information
    for key , value in user_info.items():
      # The image of the user does not need a subheader
      if key != 'image_url':
        st.subheader(
          '''
           {} : {}
          '''.format(key, value)
          )
      else:
        st.image(value)
    st.subheader(" Repositories: ")
    st.table(repos_info)
  except:
    st.subheader("User doesn't exist")