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

# Marker for policy injection
POLICY_INJECTION_MARKER = "## Policies"

# Helper function to read system message from file
def _read_system_message(file_path: str) -> str:
    """Reads system message from a file, trying relative path first, then script directory."""
    try:
        # Try relative path first
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        try:
            # Try path relative to the script directory as a fallback
            script_dir = os.path.dirname(__file__)
            alt_path = os.path.join(script_dir, file_path)
            with open(alt_path, "r", encoding="utf-8") as f:
                logger.warning(f"Could not find {file_path} directly, reading from {alt_path}")
                return f.read()
        except FileNotFoundError:
             logger.error(f"System message file not found at {file_path} or {alt_path}")
             raise FileNotFoundError(f"Agent system message file not found: {file_path}")
        except Exception as e:
             logger.error(f"Error reading system message file {file_path} (or alt): {e}", exc_info=True)
             raise # Re-raise other errors

# --- UPDATED setup_chat function ---
def setup_chat(
    llm_provider: str = VERTEX_AI,
    model_name: str = "gemini-1.5-pro-002",
    policy_text: Optional[str] = None  # Add policy_text parameter
    ) -> Tuple[GroupChatManager, UserProxyAgent]:
    """
    Sets up the agents, group chat, and manager based on selected LLM provider.
    Injects provided policy_text into PolicyGuard's system message.
    Reads Vertex AI credentials from st.secrets.

    Args:
        llm_provider (str): The LLM provider to use (e.g., VERTEX_AI, AZURE).
        model_name (str): The specific model name for the chosen provider.
        policy_text (Optional[str]): The policy text provided by the user.

    Returns:
        Tuple[GroupChatManager, UserProxyAgent]: The configured GroupChatManager and the Boss agent.

    Raises:
        FileNotFoundError: If agent system message files are not found.
        ValueError: If the LLM configuration is invalid, secrets are missing, or agent creation fails.
        Exception: For other potential setup errors.
    """
    logger.info(f"Setting up chat with provider: {llm_provider}, model: {model_name}")
    if policy_text:
        logger.info("Policy text provided, will inject into PolicyGuard.")
    else:
        logger.info("No policy text provided, PolicyGuard will use default from file.")

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
    #     # Example: Read Azure credentials from st.secrets["azure_credentials"]
    #     # ...
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    if not llm_config.get_config():
        # This might happen if get_config itself has issues or returns None/empty
        raise ValueError("Failed to create a valid LLM configuration object or dictionary.")

    # --- Agent Creation ---
    try:
        # Ensure non-PolicyGuard system message files exist
        for file_path in [BOSS_SYS_MSG_FILE, CHALLENGER_SYS_MSG_FILE, POLICY_GUARD_SYS_MSG_FILE]: # Read policy guard file for base
            if not os.path.exists(file_path):
                # Try alternative path check using helper's logic (though helper will raise if needed)
                script_dir = os.path.dirname(__file__)
                alt_path = os.path.join(script_dir, file_path)
                if not os.path.exists(alt_path):
                    raise FileNotFoundError(f"Agent system message file not found: {file_path}")

        # --- Create Boss Agent (using file) ---
        boss = create_agent(
            name=BOSS_NAME,
            llm_config=llm_config,
            system_message_file=BOSS_SYS_MSG_FILE, # Reads from file
            agent_type="user_proxy",
        )

        # --- Prepare PolicyGuard System Message ---
        base_policy_guard_sys_msg = _read_system_message(POLICY_GUARD_SYS_MSG_FILE)
        policy_guard_final_sys_msg = base_policy_guard_sys_msg

        if policy_text and policy_text.strip():
            # Inject the policy text
            if POLICY_INJECTION_MARKER in base_policy_guard_sys_msg:
                policy_guard_final_sys_msg = base_policy_guard_sys_msg.replace(
                    POLICY_INJECTION_MARKER,
                    f"""{POLICY_INJECTION_MARKER}

{policy_text.strip()}""",
                    1 # Replace only the first occurrence
                )
                logger.info(f"Injected policy text into PolicyGuard system message under '{POLICY_INJECTION_MARKER}'.")
            else:
                logger.warning(f"Policy injection marker '{POLICY_INJECTION_MARKER}' not found in {POLICY_GUARD_SYS_MSG_FILE}. Appending policy text instead.")
                policy_guard_final_sys_msg += f"""

## Policies

{policy_text.strip()}"""
        else:
             logger.info(f"Using default system message for PolicyGuard from {POLICY_GUARD_SYS_MSG_FILE}.")

        # --- Create PolicyGuard Agent (using content) ---
        policy_guard = create_agent(
            name=POLICY_GUARD_NAME,
            llm_config=llm_config,
            system_message_content=policy_guard_final_sys_msg, # Pass constructed content
            system_message_file=None # Explicitly set file to None
        )

        # --- Create Challenger Agent (using file) ---
        # Load system message for Challenger (example using session state if needed for future features)
        # first_line_challenger_sys_msg = st.session_state.get("first_line_challenger_sys_msg", _read_system_message(CHALLENGER_SYS_MSG_FILE))
        first_line_challenger = create_agent(
            name=CHALLENGER_NAME,
            llm_config=llm_config,
            system_message_file=CHALLENGER_SYS_MSG_FILE, # Reads from file
            # system_message_content=first_line_challenger_sys_msg # Or use content if implementing editable prompts feature
        )

        logger.info("Agents created successfully.")
    except FileNotFoundError as e:
        logger.error(e)
        raise
    except ValueError as e: # Catch errors from create_agent or LLM config
        logger.error(f"Agent creation failed: {e}", exc_info=True)
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
    print("Running main.py script (for local testing - may not work with st.secrets).")
    # Add a dummy policy for local testing if desired
    test_policy = "This is a sample policy for local testing."
    print(f"Using test policy for local run: '{test_policy}'")
    try:
        print("Attempting to set up chat via setup_chat() locally...")
        # WARNING: This call will likely fail if setup_chat() tries to access st.secrets
        # Pass the test policy text here for local execution
        test_manager, test_boss_agent = setup_chat(policy_text=test_policy) # Example: Call setup_chat with test policy
        print("-" * 20)
        print("Local setup successful (if st.secrets wasn't required).")
        print(f"  Manager Name: {test_manager.name}")
        print(f"  Boss Agent Name: {test_boss_agent.name}")
        print(f"  Team Agents: {[agent.name for agent in test_manager.groupchat.agents]}")
        # You could inspect the PolicyGuard's system message if setup succeeded
        # policy_guard_agent = next((a for a in test_manager.groupchat.agents if a.name == POLICY_GUARD_NAME), None)
        # if policy_guard_agent:
        #     print("\nPolicyGuard System Message (First 200 chars):")
        #     print(policy_guard_agent.system_message[:200] + "...")
        print("-" * 20)
        # You could add a simple initiation test here too, IF setup succeeded:
        # print("Attempting to initiate chat task (Testing purposes)...")
        # test_initial_prompt = "This is a local test task description." # Don't include policy here
        # initial_messages, next_agent = initiate_chat_task(test_boss_agent, test_manager, test_initial_prompt)
        # print("Chat initiated for testing.")
        # print(f"Initial messages count: {len(initial_messages)}")
        # print(f"First message content: {initial_messages[0]['content']}")
        # print(f"Next agent to speak: {next_agent.name if next_agent else 'None'}")
        # print("-" * 20)

    except FileNotFoundError as e:
        print(f"\n*** LOCAL SETUP FAILED: File Not Found. ***")
        print(f"    Error details: {e}")
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
