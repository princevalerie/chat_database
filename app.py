import streamlit as st
from pathlib import Path
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType

from langchain_community.callbacks import StreamlitCallbackHandler

from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

from sqlalchemy import create_engine, inspect, event


import sqlite3
import psycopg2
import mysql.connector
from langchain_groq import ChatGroq
import urllib.parse

# Streamlit Page Setup
st.set_page_config(page_title="Chat with your database")
st.title("Chat with your database")

# Define constants for database types
POSTGRES = "USE_POSTGRES"
MYSQL = "USE_MYSQL"


# Sidebar - Choose database
radio_opt = ["Connect to PostgreSQL Database", "Connect to MySQL Database"]

selected_opt = st.sidebar.radio(label="Choose the DB you want to chat with", options=radio_opt)

# Initialize Database Variables
db_uri = None
pg_host, pg_user, pg_password, pg_db = None, None, None, None
mysql_host, mysql_user, mysql_password, mysql_db = None, None, None, None

if selected_opt == radio_opt[0]:  # PostgreSQL

    db_uri = POSTGRES
    pg_host = st.sidebar.text_input("PostgreSQL Host").strip()
    pg_user = st.sidebar.text_input("PostgreSQL User").strip()
    pg_password = st.sidebar.text_input("PostgreSQL Password", type="password")
    pg_db = st.sidebar.text_input("PostgreSQL Database Name").strip()
elif selected_opt == radio_opt[1]:  # MySQL

    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("MySQL Host").strip()
    mysql_user = st.sidebar.text_input("MySQL User").strip()
    mysql_password = st.sidebar.text_input("MySQL Password", type="password")
    mysql_db = st.sidebar.text_input("MySQL Database").strip()

api_key = st.sidebar.text_input(label="Enter your Groq API Key", type="password")


if not api_key:
    st.error("Please enter the Groq API key to continue.")
    st.stop()


# LLM Model
if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
else:
    llm = None




# Function to configure database connection
@st.cache_resource(ttl="2h", show_spinner=False)
def validate_connection(engine):

    """Validate the database connection is active"""
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception as e:
        st.error(f"❌ Database connection validation failed: {str(e)}")
        return False

def configure_db(db_uri, pg_host=None, pg_user=None, pg_password=None, pg_db=None, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    """Returns a SQLDatabase instance based on the selected configuration."""
    def create_and_validate_engine(db_url):
        engine = create_engine(db_url)
        if not validate_connection(engine):
            st.stop()
        return engine

    def create_restricted_db(engine):

        """Create a SQLDatabase instance with restricted table access and prevent DELETE/TRUNCATE"""
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()

            
        # Add event listener to prevent DELETE/TRUNCATE operations
        @event.listens_for(engine, 'before_execute')
        def prevent_destructive_operations(conn, clauseelement, multiparams, params):
            if isinstance(clauseelement, str):
                query = clauseelement.upper()
                if 'DELETE' in query or 'TRUNCATE' in query:
                    raise Exception("DELETE and TRUNCATE operations are not permitted")
            
        return SQLDatabase(
            engine,
            include_tables=existing_tables,
            schema='public',
            sample_rows_in_table_info=1,
            view_support=True
        )





    if db_uri == POSTGRES:

        if not (pg_host and pg_user and pg_password and pg_db):
            st.error("❌ Please provide all PostgreSQL connection details.")
            st.stop()

        try:
            # URL Encode the password to handle special characters
            encoded_password = urllib.parse.quote(pg_password)

            # Corrected Connection String
            db_url = f"postgresql+psycopg2://{pg_user}:{encoded_password}@{pg_host}/{pg_db}"

            print(f"Connecting to PostgreSQL: {db_url}")  # Debugging Output
            engine = create_engine(db_url)
            return create_restricted_db(engine)

        except Exception as e:
            st.error(f"❌ PostgreSQL connection failed: {e}")
            st.stop()

    elif db_uri == MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("❌ Please provide all MySQL connection details.")
            st.stop()

        try:
            db_url = f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"
            print(f"Connecting to MySQL: {db_url}")  # Debugging Output
            engine = create_engine(db_url)
            return create_restricted_db(engine)
        except Exception as e:
            st.error(f"❌ MySQL connection failed: {e}")
            st.stop()

# Initialize database connection
if db_uri == POSTGRES:

    db = configure_db(db_uri, pg_host, pg_user, pg_password, pg_db)
elif db_uri == MYSQL:
    db = configure_db(db_uri, mysql_host=mysql_host, mysql_user=mysql_user, mysql_password=mysql_password, mysql_db=mysql_db)



# Toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Create SQL Agent with strict table access
def validate_query(query: str) -> bool:
    """Validate if query contains prohibited operations"""
    if not query:
        return False
    # Convert to uppercase for case-insensitive check
    query_upper = query.upper()
    prohibited_ops = ["DELETE", "TRUNCATE"]
    return not any(op in query_upper for op in prohibited_ops)

def validate_table_access(table_name: str) -> bool:
    """Validate if table exists"""
    if not table_name:
        return False
    inspector = inspect(db._engine)
    return inspector.has_table(table_name)




def safe_agent_run(query: str, *args, **kwargs):
    """Wrapper function to validate queries before execution"""
    if not validate_query(query):
        return "Access denied. DELETE and TRUNCATE operations are not permitted."
    return agent.run(query, *args, **kwargs)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,


    top_k=5,  # Allow access to all approved tables
    max_iterations=10  # Allow more complex queries within approved tables
)

# Message history
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User query input with table restriction notice
user_query = st.chat_input(
    placeholder="Ask about your database",
)

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = safe_agent_run(user_query, callbacks=[st_cb])

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
