
# Unit Test Plan

This document outlines the necessary unit tests to ensure the stability and correctness of the Risk Management Challenge application's core logic, primarily within `LLMConfiguration.py` and `common_functions.py`.

## LLMConfiguration.py

Tests for the LLM configuration class.

*   **`test_get_vertex_ai_config_success`:** Verify correct config dictionary structure for Vertex AI with valid inputs.
*   **`test_get_vertex_ai_config_missing_params`:** Ensure `ValueError` is raised if required Vertex AI parameters (`project_id`, `location`, `vertex_credentials`) are missing.
*   **`test_get_azure_config_success`:** Verify correct config dictionary structure for Azure with valid inputs.
*   **`test_get_azure_config_missing_params`:** Ensure `ValueError` is raised if required Azure parameters are missing.
*   **`test_get_anthropic_config_success`:** Verify correct config dictionary structure for Anthropic with valid inputs.
*   **`test_get_anthropic_config_missing_params`:** Ensure `ValueError` is raised if required Anthropic parameters are missing.
*   **`test_unsupported_provider`:** Ensure `ValueError` is raised when an unknown provider string is used.

## common_functions.py

Tests for helper functions managing agents, chat flow, and utilities.

### File Reading

*   **`test_read_system_message_success`:** Verify correct content return when reading an existing file (mock `open`).
*   **`test_read_system_message_not_found`:** Verify fallback message is returned when file doesn't exist (mock `open` raising `FileNotFoundError`).
*   **`test_read_system_message_read_error`:** Verify fallback message is returned on other file read errors (mock `open` raising `Exception`).

### Agent Creation (`create_agent`)

*   **`test_create_agent_user_proxy`:** Ensure a `UserProxyAgent` is created with `human_input_mode="NEVER"`.
*   **`test_create_agent_assistant`:** Ensure an `AssistantAgent` is created for the assistant type.
*   **`test_create_agent_priority_content_over_file`:** Ensure `system_message_content` is used when both content and file are provided.
*   **`test_create_agent_uses_file`:** Ensure content from `system_message_file` is used when only the file is provided (mock `read_system_message`).
*   **`test_create_agent_no_message_error`:** Ensure `ValueError` is raised if neither content nor file yields a system message.
*   **`test_create_agent_invalid_llm_config`:** Ensure `ValueError` is raised if the provided `llm_config` is invalid (mock `llm_config.get_config()` returning `None`).

### Custom Speaker Selection (`custom_speaker_selection`)

*   **`test_custom_speaker_selection_initial`:** Verify `ProductLead` is selected when message history is empty.
*   **`test_custom_speaker_selection_mention_policyguard_last`:** Verify `PolicyGuard` is selected when mentioned last in the message.
*   **`test_custom_speaker_selection_mention_challenger_last`:** Verify `Challenger` is selected when mentioned last. # Updated name
*   **`test_custom_speaker_selection_mention_productlead_last`:** Verify `ProductLead` is selected when mentioned last.
*   **`test_custom_speaker_selection_multiple_mentions`:** Verify the *last* mentioned agent is selected if multiple agents are named.
*   **`test_custom_speaker_selection_no_mentions`:** Verify `ProductLead` is selected when no agent names are found.
*   **`test_custom_speaker_selection_mention_self`:** Verify `ProductLead` is selected if the last speaker mentions only themselves.
*   **`test_custom_speaker_selection_terminate`:** Verify `ProductLead` is selected when the message ends with "TERMINATE".
*   **`test_custom_speaker_selection_no_productlead_agent`:** Verify the first agent is returned if no `UserProxyAgent` (ProductLead) is in the chat (and log error).
*   **`test_custom_speaker_selection_error_accessing_message`:** Verify `ProductLead` is selected if there's an error reading the last message content.

### Group Chat Setup (`create_groupchat`, `create_groupchat_manager`)

*   **`test_create_groupchat_success`:** Ensure `GroupChat` object is created with the custom speaker selection function when valid agents (including ProductLead) are provided.
*   **`test_create_groupchat_no_userproxy`:** Ensure `ValueError` is raised if no `UserProxyAgent` is included in the agent list.
*   **`test_create_groupchat_manager_success`:** Ensure `GroupChatManager` is created with valid inputs.
*   **`test_create_groupchat_manager_invalid_llm`:** Ensure `ValueError` is raised if the LLM config is invalid.

### Chat Flow (`initiate_chat_task`, `run_agent_step`, `send_user_message`)

*   **`test_initiate_chat_task_success`:** Verify history reset, correct initial message addition (from ProductLead), and `PolicyGuard` as the first speaker.
*   **`test_initiate_chat_task_no_policyguard`:** Verify speaker selection fallback works if `PolicyGuard` isn't found initially.
*   **`test_run_agent_step_framework_adds_message`:** Test agent step where AutoGen automatically adds the message to history. Verify correct new messages and next speaker returned.
*   **`test_run_agent_step_manual_add_message`:** Test agent step where AutoGen *doesn't* add the message. Verify manual addition works and correct messages/speaker are returned.
*   **`test_run_agent_step_no_reply`:** Test agent step where agent returns `None`. Verify no message is added, empty list returned, and next speaker is selected.
*   **`test_run_agent_step_error`:** Test agent step where `generate_reply` raises an error. Verify error handling, empty list returned, and `ProductLead` selected next.
*   **`test_send_user_message_success`:** Verify user message is correctly added to history and correct message/speaker are returned.
*   **`test_send_user_message_empty`:** Verify empty user messages are ignored (not added to history), empty list returned, and next speaker is selected.
