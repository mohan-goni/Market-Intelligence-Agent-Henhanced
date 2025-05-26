from supabase import create_client, Client
import os
from dotenv import load_dotenv
from typing import Optional, Dict, Any

# Load environment variables from .env file
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client | None = None

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Warning: SUPABASE_URL and SUPABASE_KEY must be set in the environment variables for Supabase client to initialize.")
    # supabase remains None, functions using it should handle this
else:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Error initializing Supabase client: {e}")
        supabase = None # Ensure supabase is None if initialization fails

def is_supabase_connected() -> bool:
    if supabase is None:
        print("Supabase client not initialized.")
        return False
    try:
        # Attempt to get server version or a similar lightweight, non-data-specific call
        # As of supabase-py v1.x, there isn't a direct "ping" or "server_version"
        # A common workaround is to try a very simple, non-table-specific query
        # For this example, we'll try to list functions, which should exist even if empty
        # This is less ideal than a specific health check endpoint if available in future versions
        # Using a schema list or a generic non-table specific call if available
        # The original example 'users' table check is okay but assumes the table exists.
        # Let's try a more generic check if possible, or stick to the simple table query.
        # For now, using the provided example which is a simple select, assuming 'users' might not exist yet.
        # A more robust check might involve listing schemas or a meta-table if Supabase API supports.
        # Given the constraints, we'll use a placeholder for a real check.
        # A simple check on the 'auth.users' table (if using Supabase Auth) or a custom table.
        # Let's assume 'users' table for now as per problem description.
        # If the 'users' table might not exist, this check will fail.
        # A truly generic check is hard without knowing more about the specific Supabase setup or available meta-queries.
        
        # Placeholder: a real check would be more robust.
        # This will try to select from a potentially non-existent table 'users'.
        # A better check might be to list schemas or a specific metadata query.
        # For this exercise, we'll assume this simple query is acceptable.
        supabase.table('users').select('id', count='exact').limit(1).execute()
        print("Supabase connection check successful (simulated).")
        return True
    except Exception as e:
        print(f"Supabase connection check failed: {e}")
        return False

async def create_user(email: str, hashed_password: str, full_name: Optional[str] = None) -> Dict[str, Any] | None:
    if supabase is None:
        print("Error: Supabase client not initialized. Cannot create user.")
        return None
    try:
        user_data = {
            "email": email,
            "hashed_password": hashed_password,
        }
        if full_name:
            user_data["full_name"] = full_name
        
        # Supabase automatically handles 'created_at' if the column has a default value like now()
        response = await supabase.table("users").insert(user_data).execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]
        # Supabase insert might return an error in response.error if unique constraint fails or other issues
        if response.error:
            print(f"Error creating user in Supabase: {response.error.message}")
            # Check for unique constraint violation (example, actual error code/message might vary)
            if "duplicate key value violates unique constraint" in response.error.message.lower():
                 print(f"User with email {email} already exists.")
            return None
        return None # Should not happen if no data and no error, but as a fallback
    except Exception as e:
        print(f"Exception during user creation: {e}")
        return None

async def get_user_by_email(email: str) -> Dict[str, Any] | None:
    if supabase is None:
        print("Error: Supabase client not initialized. Cannot get user.")
        return None
    try:
        response = await supabase.table("users").select("*").eq("email", email).limit(1).execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]
        if response.error:
            print(f"Error fetching user from Supabase: {response.error.message}")
            return None
        return None # User not found
    except Exception as e:
        print(f"Exception during fetching user: {e}")
        return None

# --- API Key Management Functions ---

async def save_user_api_key(user_id: str, service_name: str, api_key: str) -> bool:
    if supabase is None:
        print("Error: Supabase client not initialized. Cannot save API key.")
        return False
    try:
        # Upsert operation: if (user_id, service_name) exists, update api_key; otherwise, insert new row.
        # Supabase `upsert` requires a conflict resolution target, e.g., a primary key or unique constraint.
        # Assuming `user_api_keys` table has a composite primary key or unique constraint on (user_id, service_name).
        response = await supabase.table("user_api_keys").upsert({
            "user_id": user_id,
            "service_name": service_name,
            "api_key": api_key  # Ensure this key is encrypted if stored in DB
        }).execute()
        
        if response.error:
            print(f"Error saving API key for user {user_id}, service {service_name}: {response.error.message}")
            return False
        return True
    except Exception as e:
        print(f"Exception during saving API key: {e}")
        return False

async def get_user_api_key(user_id: str, service_name: str) -> str | None:
    if supabase is None:
        print("Error: Supabase client not initialized. Cannot get API key.")
        return None
    try:
        response = await supabase.table("user_api_keys").select("api_key").eq("user_id", user_id).eq("service_name", service_name).limit(1).execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0].get("api_key")
        if response.error:
            print(f"Error fetching API key for user {user_id}, service {service_name}: {response.error.message}")
            return None
        return None # Key not found
    except Exception as e:
        print(f"Exception during fetching API key: {e}")
        return None

async def get_all_user_api_keys(user_id: str) -> Dict[str, str] | None:
    if supabase is None:
        print("Error: Supabase client not initialized. Cannot get all API keys.")
        return None
    try:
        response = await supabase.table("user_api_keys").select("service_name, api_key").eq("user_id", user_id).execute()
        
        if response.data:
            keys = {item["service_name"]: item["api_key"] for item in response.data}
            return keys
        if response.error:
            print(f"Error fetching all API keys for user {user_id}: {response.error.message}")
            return None
        return {} # No keys found for the user, return empty dict
    except Exception as e:
        print(f"Exception during fetching all API keys: {e}")
        return None

# --- Analysis Results Functions ---

async def save_analysis_result(user_id: str, query: str, market_domain: str, result_data: Dict[str, Any], status: str, error_message: Optional[str] = None) -> str | None:
    if supabase is None:
        print("Error: Supabase client not initialized. Cannot save analysis result.")
        return None
    try:
        record = {
            "user_id": user_id,
            "query": query,
            "market_domain": market_domain,
            "result_data": result_data, # Ensure this is JSON serializable
            "status": status,
            "error_message": error_message
            # 'created_at' and 'updated_at' should be handled by DB (e.g. default now())
        }
        response = await supabase.table("analysis_results").insert(record).execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0].get("id") # Assuming 'id' is the primary key
        if response.error:
            print(f"Error saving analysis result for user {user_id}: {response.error.message}")
            return None
        return None
    except Exception as e:
        print(f"Exception during saving analysis result: {e}")
        return None

async def get_analysis_results_for_user(user_id: str, limit: int = 20, offset: int = 0) -> list[Dict[str, Any]] | None:
    if supabase is None:
        print("Error: Supabase client not initialized. Cannot get analysis results.")
        return None
    try:
        response = await supabase.table("analysis_results").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(limit).offset(offset).execute()
        
        if response.data:
            return response.data
        if response.error:
            print(f"Error fetching analysis results for user {user_id}: {response.error.message}")
            return None
        return [] # No results found
    except Exception as e:
        print(f"Exception during fetching analysis results: {e}")
        return None

async def get_analysis_result_by_id(user_id: str, result_id: str) -> Dict[str, Any] | None:
    if supabase is None:
        print("Error: Supabase client not initialized. Cannot get analysis result by ID.")
        return None
    try:
        response = await supabase.table("analysis_results").select("*").eq("id", result_id).eq("user_id", user_id).limit(1).execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]
        if response.error:
            print(f"Error fetching analysis result by ID {result_id} for user {user_id}: {response.error.message}")
            return None
        return None # Result not found or not owned by user
    except Exception as e:
        print(f"Exception during fetching analysis result by ID: {e}")
        return None

async def update_analysis_status(user_id: str, result_id: str, status: str, error_message: Optional[str] = None) -> bool:
    if supabase is None:
        print("Error: Supabase client not initialized. Cannot update analysis status.")
        return False
    try:
        update_data = {"status": status, "updated_at": "now()"} # Let DB handle timestamp
        if error_message is not None: # Allow clearing error message by passing None
            update_data["error_message"] = error_message
        
        response = await supabase.table("analysis_results").update(update_data).eq("id", result_id).eq("user_id", user_id).execute()
        
        # Check if update was successful.
        # Supabase update response.data might be empty on success if returning="minimal" (default)
        # A more reliable check might be to see if response.error is None and if any rows were affected (if available)
        # For now, if no error, assume success.
        if response.error:
            print(f"Error updating analysis status for result ID {result_id}, user {user_id}: {response.error.message}")
            return False
        # To confirm a row was actually updated, you might need to check response.count or similar attribute if available
        # For supabase-py, the data field in the response often contains the updated records if `returning="representation"` was used.
        # Without it, we rely on absence of error.
        return True
    except Exception as e:
        print(f"Exception during updating analysis status: {e}")
        return False

# --- Chat History Functions ---

async def save_chat_message(user_id: str, session_id: str, message_type: str, message_content: str, metadata: Optional[Dict[str, Any]] = None) -> str | None:
    if supabase is None:
        print("Error: Supabase client not initialized. Cannot save chat message.")
        return None
    try:
        record = {
            "user_id": user_id,
            "session_id": session_id,
            "message_type": message_type, # e.g., 'user', 'ai', 'system'
            "message_content": message_content,
            "metadata": metadata if metadata else {} # Ensure metadata is at least an empty dict if None
            # 'created_at' should be handled by DB
        }
        response = await supabase.table("chat_history").insert(record).execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0].get("id") # Assuming 'id' is the primary key
        if response.error:
            print(f"Error saving chat message for user {user_id}, session {session_id}: {response.error.message}")
            return None
        return None
    except Exception as e:
        print(f"Exception during saving chat message: {e}")
        return None

async def get_chat_history_for_session(user_id: str, session_id: str, limit: int = 50, offset: int = 0) -> list[Dict[str, Any]] | None:
    if supabase is None:
        print("Error: Supabase client not initialized. Cannot get chat history.")
        return None
    try:
        response = await supabase.table("chat_history").select("*").eq("user_id", user_id).eq("session_id", session_id).order("created_at", desc=False).limit(limit).offset(offset).execute() # Ascending for chat
        
        if response.data:
            return response.data
        if response.error:
            print(f"Error fetching chat history for user {user_id}, session {session_id}: {response.error.message}")
            return None
        return [] # No messages found
    except Exception as e:
        print(f"Exception during fetching chat history: {e}")
        return None

# Example usage (optional, for testing purposes)
# async def main():
#     if is_supabase_connected():
#         print("Supabase is connected.")
#         # Test create_user
#         # Note: Hashing passwords should be done before calling create_user
#         # For example, using passlib:
#         # from passlib.context import CryptContext
#         # pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
#         # hashed_pw = pwd_context.hash("a_secure_password")
#         # new_user = await create_user("test2@example.com", hashed_pw, "Test User Two")
#         # if new_user:
#         #     print(f"Created user: {new_user}")
#         # else:
#         #     print("Failed to create user or user already exists.")

#         # Test get_user_by_email
#         # existing_user = await get_user_by_email("test@example.com")
#         # if existing_user:
#         #     print(f"Found user: {existing_user}")
#         # else:
#         #     print("User not found or error fetching.")
#     else:
#         print("Supabase is NOT connected. Check SUPABASE_URL and SUPABASE_KEY.")

# if __name__ == "__main__":
#     import asyncio
#     # To run the main function for testing (requires .env to be set up correctly)
#     # Ensure SUPABASE_URL and SUPABASE_KEY are valid and the Supabase instance is running with the 'users' table.
#     # The 'users' table schema should be: id (uuid, pk), email (text, unique), ...
#     # The 'user_api_keys' table schema: user_id (uuid, fk to users.id), service_name (text), api_key (text, encrypted), created_at, updated_at. PK: (user_id, service_name)
#     # The 'analysis_results' table schema: id (uuid, pk), user_id (uuid, fk to users.id), query (text), market_domain (text), result_data (jsonb), status (text), error_message (text, nullable), created_at, updated_at.
#     # The 'chat_history' table schema: id (uuid, pk), user_id (uuid, fk to users.id), session_id (text), message_type (text), message_content (text), metadata (jsonb, nullable), created_at.
#     # asyncio.run(main())
#
# # To use these functions, ensure your Supabase instance has tables like:
# #
# # CREATE TABLE user_api_keys (
# #   user_id UUID REFERENCES users(id) ON DELETE CASCADE,
# #   service_name TEXT NOT NULL,
# #   api_key TEXT NOT NULL, -- Consider encrypting this column in the database
# #   created_at TIMESTAMPTZ DEFAULT NOW(),
# #   updated_at TIMESTAMPTZ DEFAULT NOW(),
# #   PRIMARY KEY (user_id, service_name)
# # );
# #
# # CREATE TABLE analysis_results (
# #   id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
# #   user_id UUID REFERENCES users(id) ON DELETE CASCADE,
# #   query TEXT NOT NULL,
# #   market_domain TEXT NOT NULL,
# #   result_data JSONB,
# #   status TEXT NOT NULL, -- e.g., 'processing', 'completed', 'failed'
# #   error_message TEXT,
# #   created_at TIMESTAMPTZ DEFAULT NOW(),
# #   updated_at TIMESTAMPTZ DEFAULT NOW()
# # );
# #
# # CREATE TABLE chat_history (
# #   id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
# #   user_id UUID REFERENCES users(id) ON DELETE CASCADE,
# #   session_id TEXT NOT NULL, -- Could be a UUID or any string identifier for a chat session
# #   message_type TEXT NOT NULL, -- e.g., 'user', 'ai', 'system'
# #   message_content TEXT NOT NULL,
# #   metadata JSONB,
# #   created_at TIMESTAMPTZ DEFAULT NOW()
# # );
# #
# # -- Optional: Trigger to auto-update 'updated_at' columns
# # CREATE OR REPLACE FUNCTION trigger_set_timestamp()
# # RETURNS TRIGGER AS $$
# # BEGIN
# #   NEW.updated_at = NOW();
# #   RETURN NEW;
# # END;
# # $$ LANGUAGE plpgsql;
# #
# # CREATE TRIGGER set_user_api_keys_updated_at
# # BEFORE UPDATE ON user_api_keys
# # FOR EACH ROW
# # EXECUTE FUNCTION trigger_set_timestamp();
# #
# # CREATE TRIGGER set_analysis_results_updated_at
# # BEFORE UPDATE ON analysis_results
# # FOR EACH ROW
# # EXECUTE FUNCTION trigger_set_timestamp();
#
# # email (text, unique)
#     # hashed_password (text)
#     # full_name (text, nullable)
#     # created_at (timestamp with time zone, default now())
#     # asyncio.run(main())
