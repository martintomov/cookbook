import streamlit as st
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import pandas as pd

st.title("Data Analyst")
st.write("Upload a CSV file and ask questions about the data.")

uploaded_file = st.file_uploader("Upload a CSV file.", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Here are the first five rows of your file:")
    st.write(df.head())

    llm = ChatOpenAI(model="gpt-3.5-turbo")
    agent_prompt_prefix = 'Your name is LangAgent and you are working with a pandas dataframe called "df".'

    agent = create_pandas_dataframe_agent(
        llm, df, prefix=agent_prompt_prefix, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS
    )

    user_query = st.text_input("Ask a question about your data:")
    if user_query:
        response = agent.invoke(user_query)
        st.write("Response:", response)

else:
    st.write("Please upload a CSV file to continue.")
