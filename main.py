import json
import os
import logging
from typing import Tuple, Optional 

# Import necessary components from autogen and local modules
import autogen
import streamlit as st # Import Streamlit
from autogen import UserProxyAgent, GroupChatManager
from LLMConfiguration import LLMConfiguration, VERTEX_AI, AZURE, ANTHROPIC # Keep if LLMConfig logic stays here
from common_functions import (
    create_agent,
    create_groupchat,
    create_groupchat_manager,
    initiate_chat_task, # Keep this import if setup_chat calls initiate_chat_task
    run_agent_step,     # Will be used by app.py
    send_user_message   # Will be used by app.py
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
# Define agent names and system message file paths
BOSS_NAME = "Boss"
POLICY_GUARD_NAME = "PolicyGuard"
CHALLENGER_NAME = "FirstLineChallenger"

BOSS_SYS_MSG_FILE = "Boss.md"
POLICY_GUARD_SYS_MSG_FILE = "PolicyGuard.md"
CHALLENGER_SYS_MSG_FILE = "FirstLineChallenger.md"

# Define credentials path (No longer needed for Vertex AI if using st.secrets)
# CREDENTIALS_FILE_PATH = '../sa3.json' # Commented out or remove

# Helper function to read system message from file
def _read_system_message(file_path: str) -> str:
    try:
        with open(file_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        script_dir = os.path.dirname(__file__)
        alt_path = os.path.join(script_dir, file_path)
        with open(alt_path, "r") as f:
            return f.read()


def setup_chat(llm_provider: str = VERTEX_AI, model_name: str = "gemini-1.5-pro-002") -> Tuple[GroupChatManager, UserProxyAgent]:
    """
    Sets up the agents, group chat, and manager based on selected LLM provider.
    Reads Vertex AI credentials from st.secrets.

    Args:
        llm_provider (str): The LLM provider to use (e.g., VERTEX_AI, AZURE).
        model_name (str): The specific model name for the chosen provider.

    Returns:
        Tuple[GroupChatManager, UserProxyAgent]: The configured GroupChatManager and the Boss agent.

    Raises:
        FileNotFoundError: If agent system message files are not found.
        ValueError: If the LLM configuration is invalid, secrets are missing, or agent creation fails.
        Exception: For other potential setup errors.
    """
    logger.info(f"Setting up chat with provider: {llm_provider}, model: {model_name}")

    # --- LLM Configuration ---
    if llm_provider == VERTEX_AI:
        try:
            # ---- Read Credentials from Streamlit Secrets ----
            if "gcp_credentials" not in st.secrets:
                 raise ValueError("Missing 'gcp_credentials' section in Streamlit secrets.")

            required_keys = ["project_id", "private_key", "client_email", "type"]
            if not all(key in st.secrets["gcp_credentials"] for key in required_keys):
                 raise ValueError(f"Missing required keys ({required_keys}) within 'gcp_credentials' in Streamlit secrets.")

            # Reconstruct the credentials dictionary from secrets
            vertex_credentials = dict(st.secrets["gcp_credentials"])

            llm_config = LLMConfiguration(
                VERTEX_AI,
                model_name,
                project_id=vertex_credentials.get('project_id'),
                location="us-central1", # Or make this configurable
                vertex_credentials=vertex_credentials, # Pass the dict from secrets
            )
            logger.info("Vertex AI LLM configuration loaded from st.secrets.")

        except ValueError as e:
             logger.error(f"Credential error: {e}")
             raise # Re-raise value errors related to secrets
        except Exception as e:
            logger.error(f"Error loading Vertex AI credentials from st.secrets: {e}", exc_info=True)
            raise

    # Add elif blocks here for AZURE, ANTHROPIC etc. if needed
    # elif llm_provider == AZURE:
    #     # Read Azure credentials from st.secrets["azure_credentials"] for example
    #     if "azure_credentials" not in st.secrets or not all(k in st.secrets["azure_credentials"] for k in ["api_key", "base_url", "api_version"]):
    #         raise ValueError("Missing Azure credentials in Streamlit secrets.")
    #     azure_creds = st.secrets["azure_credentials"]
    #     llm_config = LLMConfiguration(AZURE, model_name, api_key=azure_creds["api_key"], ...)
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    if not llm_config.get_config():
        # This might happen if get_config itself has issues or returns None/empty
        raise ValueError("Failed to create a valid LLM configuration object or dictionary.")

    # --- Agent Creation ---
    try:
        # Ensure system message files exist - use absolute paths or ensure relative paths work from execution context
        for file_path in [BOSS_SYS_MSG_FILE, POLICY_GUARD_SYS_MSG_FILE, CHALLENGER_SYS_MSG_FILE]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Agent system message file not found: {file_path}")
        # Load system messages from Streamlit session state
        boss_sys_msg = _read_system_message(BOSS_SYS_MSG_FILE)  # Assuming this helper exists or create it
        policy_guard_sys_msg = st.session_state.get("policy_guard_sys_msg", _read_system_message(POLICY_GUARD_SYS_MSG_FILE))
        first_line_challenger_sys_msg = st.session_state.get("first_line_challenger_sys_msg", _read_system_message(CHALLENGER_SYS_MSG_FILE))
        boss = create_agent(
            BOSS_NAME,
            system_message=boss_sys_msg,  # Pass the loaded system message
            llm_config=llm_config,
            agent_type="user_proxy",
        )
        policy_guard = create_agent(
            POLICY_GUARD_NAME,
            system_message=policy_guard_sys_msg,
            llm_config=llm_config
        )
        first_line_challenger = create_agent(
            CHALLENGER_NAME,
            CHALLENGER_SYS_MSG_FILE,
            llm_config
        )
        logger.info("Agents created successfully.")
    except FileNotFoundError as e:
        logger.error(e)
        raise

    except ValueError as e:
        logger.error(f"Agent creation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during agent creation: {e}", exc_info=True)
        raise

    # --- Group Chat Setup ---
    policy_team = [boss, policy_guard, first_line_challenger]
    try:
        groupchat = create_groupchat(policy_team, max_round=50)
        logger.info("GroupChat created successfully.")
    except ValueError as e:
        logger.error(f"GroupChat creation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during GroupChat creation: {e}", exc_info=True)
        raise

    manager_llm_config = llm_config # Reusing the same config
    try:
        manager = create_groupchat_manager(groupchat, manager_llm_config)
        logger.info("GroupChatManager created successfully.")
    except ValueError as e:
        logger.error(f"GroupChatManager creation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during GroupChatManager creation: {e}", exc_info=True)
        raise

    logger.info("Chat setup completed.")
    return manager, boss


# The main execution block IS NOW PRIMARILY FOR LOCAL TESTING (if possible).
# NOTE: This block WILL FAIL when run directly if the default llm_provider is VERTEX_AI,
# because it relies on st.secrets which is only available via 'streamlit run app.py'.
# You would need to manually change llm_provider or add fallback logic here for local testing.
if __name__ == "__main__":
    print("Running main.py script (for local testing - may not work with st.secrets)...")
    try:
        print("Attempting to set up chat via setup_chat() locally...")
        # WARNING: This call will likely fail if setup_chat() tries to access st.secrets
        test_manager, test_boss_agent = setup_chat() # Example: Call setup_chat
        print("-" * 20)
        print("Local setup successful (if st.secrets wasn't required).")
        print(f"  Manager Name: {test_manager.name}")
        print(f"  Boss Agent Name: {test_boss_agent.name}")
        print(f"  Team Agents: {[agent.name for agent in test_manager.groupchat.agents]}")
        print("-" * 20)
        # You could add a simple initiation test here too, IF setup succeeded:
        # print("Attempting to initiate chat task (Testing purposes)...")
        # test_initial_prompt = "This is a local test task description.\nPolicy: Test policy content."
        # initial_messages, next_agent = initiate_chat_task(test_boss_agent, test_manager, test_initial_prompt)
        # print("Chat initiated for testing.")
        # print(f"Initial messages count: {len(initial_messages)}")
        # print(f"First message content: {initial_messages[0]['content']}")
        # print(f"Next agent to speak: {next_agent.name if next_agent else 'None'}")
        # print("-" * 20)

    except FileNotFoundError as e:
        print(f"\n*** LOCAL SETUP FAILED: File Not Found. ***")
        print(f"    Error details: {e}")
        # If it was credentials file:
        # print(f"    Please ensure '{CREDENTIALS_FILE_PATH}' exists relative to the script's execution directory if testing locally without secrets.")
    except (ValueError, KeyError) as e:
        # This will likely catch the 'st.secrets not found' error when run locally
        print(f"\n*** LOCAL SETUP FAILED: Configuration, Value, or Key Error. ***")
        print(f"    Error details: {e}")
        print(f"    (This might be expected if running locally and setup requires st.secrets)")
    except Exception as e:
        # Catch any other unexpected errors during setup test
        print(f"\n*** UNEXPECTED LOCAL SETUP FAILED ***")
        print(f"    Error type: {type(e).__name__}")
        print(f"    Error details: {e}")
        # import traceback
        # traceback.print_exc()

    print("\nmain.py local execution attempt finished.")
    # The script will now exit. Run 'streamlit run app.py' for the full application.