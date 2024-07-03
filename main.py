import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import os
from sqlalchemy import create_engine
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.agents import AgentType
from openai import OpenAI


# Get the service account credentials from secrets
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)

# Get OpenAI API key
openai_api_key = st.secrets['OPENAI_API_KEY']
openai = OpenAI(api_key=openai_api_key)

if not openai_api_key:
    st.error("Please set your OpenAI API key as an environment variable.")
    st.stop()

# Define the project and dataset id
project_id = 'you-dwh'
dataset_id = 'datawarehouse'

# Define your tables with full references
table_name1 = 'fact_secondary_sales'
table_name2 = 'dim_secondary_target_branch'
table_name3 = 'dim_secondary_forecast_branch'
table_names = (table_name1, table_name2, table_name3)

# Create an Engine and SQLDatabaseToolkit 
try:
    engine = create_engine(
    f'bigquery://{project_id}/{dataset_id}',
    credentials_info=st.secrets["gcp_service_account"]
)
    db = SQLDatabase(engine)
except Exception as e:
    st.error(f"Failed to create SQLAlchemy engine: {e}")
    st.stop()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    openai_api_key=openai_api_key,
    verbose= True
)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
toolkit.get_tools()

system_prompt = f"""
You are a BigQuery SQL expert. Given an input question, generate a syntactically correct BigQuery SQL query, execute it, and return the answer. 

Only use the following tables:
  ('datawarehouse.fact_secondary_sales', 'master_data.dim_secondary_target_branch', 'master_data.dim_secondary_forecast_branch')

Follow these rules:
1. Reject if user asking for raw data
2. Query only the necessary columns to answer the question.
3. Always use the full table references provided above, including backticks.
4. Ensure the columns and tables used in the queries exist in the database.
5. Use table aliases and refer to columns using the format alias.column_name.
6. Make sure there is no (```sql```), make the code clean, ONLY PROVIDE THE CODE
7. Always look at the data dictionary first
8. Do not include additional backticks or any unnecessary characters around the generated SQL query.
9. When aggregating data from fact_secondary_sales, create a CTE to aggregate on a Monthly level using DATE_TRUNC on the report_date column.
10. Only use distributor_account_id if a question is specifically about a store.
11. For daily sales queries, select from the table with necessary filters.
12. When querying dim_secondary_forecast_branch, rename sales_amount to forecast_amount, quantity to forecast_quantity, and quantity_carton to forecast_quantity_carton in the output.
13. When querying dim_secondary_target_branch, rename external_calculated_target_sales_amount to target_amount, external_calculated_target_qty to target_quantity, and external_calculated_target_qty_carton to target_quantity_carton in the output.

Use the following format:
  Question: "Question here"
  SQLQuery: "SQL Query to run"
  SQLResult: "Result of the SQLQuery"
  Answer: "Final answer here in tabular data, makes sure it was have descending or ascending order based on questions"

Example Query:
SELECT
    fss.account_id,
    fss.account_name,
    fss.city,
    fss.country,
    fss.net_sale_amount,
    fss.ingestion_time
FROM
    `{'datawarehouse.fact_secondary_sales'}` fss
JOIN
    `{'master_data.dim_secondary_target_branch'}` dstb ON fss.account_hq = dstb.retailer_hq
WHERE
    fss.country = 'Indonesia'
    AND fss.ingestion_time BETWEEN '2023-01-01' AND '2023-06-30'
LIMIT 10;

Question: {{input}}
"""

# Create SQL AgentExecutor 
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    top_k=1000,
    handle_parsing_errors=True,
)

# Creating Prompt Template for engine
GOOGLESQL_PROMPT = PromptTemplate(
        input_variables=["input", "table_info","top_k","project_id","dataset_id"],
        template=system_prompt,
    )
    
# Streamlit UI setup
st.title("💬 Ask You Database")
st.write("You Database is Youvit's chatbot that answers all your questions about our business. Simply ask, and it provides easy-to-understand insights from our company data. It's like having a smart coworker who knows everything about our business, always ready to help!")

# Managing session state for chat
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    
# Explanation sidebar
with st.sidebar:
    st.write('*Lorem Ipsum*')
    st.caption('''
               *Lorem Ipsum*
    ''')

    st.divider()

    st.caption("<p style ='text-align:center'> created by BI Team</p>", unsafe_allow_html=True)

# Managing session state for chat
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat input
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Generate SQL query using OpenAI
    final_prompt = GOOGLESQL_PROMPT.format(input=prompt, project_id=project_id, table_info =table_names, top_k = 10000)
    response = agent_executor.run(final_prompt)
    response_content = response
    
    # Generate insight from data
    messages = [
        {"role": "system", "content": '''
        Role: You are a data storyteller who transforms complex data into clear, relatable insights.
        Task: Analyze datasets or data descriptions and present key findings concisely.
        Instructions:
        
        Create a section for Key Insight, make sure it was:
        - Provide the important insight based on the data in list or point format.
        - Explain the key insight in a brief, comprehensive manner.
        - Use simple language and relatable examples to make the insight accessible to all.

        Keep responses concise and focused on the single most impactful insight from the data.
        '''},
        {"role": "user", "content": response_content}
    ]
    
    descriptive_agent = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.3,
        max_tokens=1500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    descriptive_result = descriptive_agent.choices[0].message.content.strip()
    
    answer_tabular = response_content.find("Answer:")
    if answer_tabular != -1:
        tabluar_data = response_content[answer_tabular + 7:].strip()


    # Append the response and update the UI
    st.session_state.messages.append({"role": "assistant", "content": tabluar_data})
    st.chat_message("assistant").write(tabluar_data)
    st.session_state.messages.append({"role": "assistant", "content": descriptive_result})
    st.chat_message("assistant").write(descriptive_result)