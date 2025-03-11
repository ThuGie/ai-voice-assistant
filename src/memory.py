"""
Memory Module

This module handles storing and retrieving conversation history
for the AI assistant to maintain context over time.
"""

import os
import json
import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

# Set up logging
logger = logging.getLogger(__name__)

class Memory:
    """Handles storing and retrieving conversation memory"""
    
    def __init__(self, db_path: str = "conversations.db"):
        """
        Initialize the memory module.
        
        Args:
            db_path: Path to the database file
        """
        self.db_path = db_path
        
        try:
            self._initialize_db()
            logger.info(f"Initialized memory module with database at {db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize memory module: {e}")
            raise
    
    def _initialize_db(self):
        """Initialize the database and create tables if they don't exist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create conversations table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                timestamp TEXT,
                metadata TEXT
            )
            ''')
            
            # Create messages table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                role TEXT,
                content TEXT,
                timestamp TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )
            ''')
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def create_conversation(self, title: Optional[str] = None, metadata: Optional[Dict] = None) -> int:
        """
        Create a new conversation.
        
        Args:
            title: Title of the conversation
            metadata: Additional metadata for the conversation
            
        Returns:
            ID of the created conversation
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Generate a title if none is provided
            if title is None:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                title = f"Conversation {timestamp}"
            
            # Convert metadata to JSON if provided
            metadata_json = json.dumps(metadata) if metadata else None
            
            # Get current timestamp
            timestamp = datetime.now().isoformat()
            
            # Insert new conversation
            cursor.execute(
                "INSERT INTO conversations (title, timestamp, metadata) VALUES (?, ?, ?)",
                (title, timestamp, metadata_json)
            )
            
            # Get ID of the created conversation
            conversation_id = cursor.lastrowid
            
            conn.commit()
            conn.close()
            
            logger.info(f"Created conversation with ID {conversation_id}")
            return conversation_id
        except Exception as e:
            logger.error(f"Failed to create conversation: {e}")
            raise
    
    def add_message(self, 
                   conversation_id: int, 
                   role: str, 
                   content: str) -> int:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: ID of the conversation
            role: Role of the message sender ("user", "assistant", "system")
            content: Content of the message
            
        Returns:
            ID of the created message
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Validate role
            valid_roles = ["user", "assistant", "system"]
            if role not in valid_roles:
                logger.warning(f"Invalid role '{role}'. Using 'user' instead.")
                role = "user"
            
            # Get current timestamp
            timestamp = datetime.now().isoformat()
            
            # Insert new message
            cursor.execute(
                "INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (conversation_id, role, content, timestamp)
            )
            
            # Get ID of the created message
            message_id = cursor.lastrowid
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Added message with ID {message_id} to conversation {conversation_id}")
            return message_id
        except Exception as e:
            logger.error(f"Failed to add message: {e}")
            raise
    
    def get_conversation(self, conversation_id: int) -> Dict[str, Any]:
        """
        Get a conversation by ID, including all messages.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Dictionary with conversation data and messages
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable row factory for dict-like rows
            cursor = conn.cursor()
            
            # Get conversation data
            cursor.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conversation_id,)
            )
            conversation_row = cursor.fetchone()
            
            if not conversation_row:
                logger.warning(f"Conversation with ID {conversation_id} not found")
                conn.close()
                return {}
            
            # Convert row to dict
            conversation = dict(conversation_row)
            
            # Parse metadata JSON if it exists
            if conversation.get("metadata"):
                conversation["metadata"] = json.loads(conversation["metadata"])
            
            # Get all messages for this conversation
            cursor.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp",
                (conversation_id,)
            )
            message_rows = cursor.fetchall()
            
            # Convert rows to dicts
            messages = [dict(row) for row in message_rows]
            
            # Add messages to conversation data
            conversation["messages"] = messages
            
            conn.close()
            return conversation
        except Exception as e:
            logger.error(f"Failed to get conversation: {e}")
            return {}
    
    def get_messages(self, 
                    conversation_id: int, 
                    limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get messages from a conversation.
        
        Args:
            conversation_id: ID of the conversation
            limit: Maximum number of messages to retrieve (most recent)
            
        Returns:
            List of message dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get messages with limit if specified
            if limit is not None:
                cursor.execute(
                    "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp DESC LIMIT ?",
                    (conversation_id, limit)
                )
            else:
                cursor.execute(
                    "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp",
                    (conversation_id,)
                )
            
            message_rows = cursor.fetchall()
            
            # Convert rows to dicts
            messages = [dict(row) for row in message_rows]
            
            # Sort by timestamp if limit was used (since we retrieved in DESC order)
            if limit is not None:
                messages.reverse()
            
            conn.close()
            return messages
        except Exception as e:
            logger.error(f"Failed to get messages: {e}")
            return []
    
    def list_conversations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get a list of recent conversations.
        
        Args:
            limit: Maximum number of conversations to retrieve
            
        Returns:
            List of conversation dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get recent conversations
            cursor.execute(
                "SELECT * FROM conversations ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
            
            conversation_rows = cursor.fetchall()
            
            # Convert rows to dicts
            conversations = []
            for row in conversation_rows:
                conversation = dict(row)
                
                # Parse metadata JSON if it exists
                if conversation.get("metadata"):
                    conversation["metadata"] = json.loads(conversation["metadata"])
                
                # Get message count
                cursor.execute(
                    "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
                    (conversation["id"],)
                )
                
                message_count = cursor.fetchone()[0]
                conversation["message_count"] = message_count
                
                conversations.append(conversation)
            
            conn.close()
            return conversations
        except Exception as e:
            logger.error(f"Failed to list conversations: {e}")
            return []
    
    def search_conversations(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for conversations by title or content.
        
        Args:
            query: Search query
            limit: Maximum number of results to retrieve
            
        Returns:
            List of conversation dictionaries matching the query
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Search in conversation titles and message content
            cursor.execute(
                """
                SELECT DISTINCT c.*
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                WHERE c.title LIKE ? OR m.content LIKE ?
                ORDER BY c.timestamp DESC
                LIMIT ?
                """,
                (f"%{query}%", f"%{query}%", limit)
            )
            
            conversation_rows = cursor.fetchall()
            
            # Convert rows to dicts
            conversations = []
            for row in conversation_rows:
                conversation = dict(row)
                
                # Parse metadata JSON if it exists
                if conversation.get("metadata"):
                    conversation["metadata"] = json.loads(conversation["metadata"])
                
                # Get message count
                cursor.execute(
                    "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
                    (conversation["id"],)
                )
                
                message_count = cursor.fetchone()[0]
                conversation["message_count"] = message_count
                
                conversations.append(conversation)
            
            conn.close()
            return conversations
        except Exception as e:
            logger.error(f"Failed to search conversations: {e}")
            return []
    
    def delete_conversation(self, conversation_id: int) -> bool:
        """
        Delete a conversation and all its messages.
        
        Args:
            conversation_id: ID of the conversation to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete all messages in the conversation
            cursor.execute(
                "DELETE FROM messages WHERE conversation_id = ?",
                (conversation_id,)
            )
            
            # Delete the conversation
            cursor.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,)
            )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Deleted conversation with ID {conversation_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete conversation: {e}")
            return False
    
    def update_conversation_title(self, conversation_id: int, title: str) -> bool:
        """
        Update the title of a conversation.
        
        Args:
            conversation_id: ID of the conversation
            title: New title
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                (title, conversation_id)
            )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Updated title of conversation {conversation_id} to '{title}'")
            return True
        except Exception as e:
            logger.error(f"Failed to update conversation title: {e}")
            return False
    
    def export_conversation(self, conversation_id: int, format: str = "json") -> str:
        """
        Export a conversation to a specified format.
        
        Args:
            conversation_id: ID of the conversation
            format: Export format ("json" or "txt")
            
        Returns:
            Exported conversation as a string
        """
        try:
            # Get conversation data
            conversation = self.get_conversation(conversation_id)
            
            if not conversation:
                logger.warning(f"Conversation with ID {conversation_id} not found")
                return ""
            
            if format.lower() == "json":
                # Return conversation as JSON
                return json.dumps(conversation, indent=2)
            
            elif format.lower() == "txt":
                # Generate text format
                text = f"Title: {conversation.get('title', 'Untitled')}\n"
                text += f"Date: {conversation.get('timestamp', '')}\n\n"
                
                for message in conversation.get("messages", []):
                    role = message.get("role", "").capitalize()
                    content = message.get("content", "")
                    text += f"{role}: {content}\n\n"
                
                return text
            
            else:
                logger.warning(f"Unsupported export format: {format}")
                return ""
        except Exception as e:
            logger.error(f"Failed to export conversation: {e}")
            return ""
    
    def import_conversation(self, data: str, format: str = "json") -> Optional[int]:
        """
        Import a conversation from a string.
        
        Args:
            data: Conversation data as a string
            format: Import format ("json" only for now)
            
        Returns:
            ID of the imported conversation if successful, None otherwise
        """
        try:
            if format.lower() == "json":
                # Parse JSON data
                conversation_data = json.loads(data)
                
                # Create new conversation
                title = conversation_data.get("title", f"Imported {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                metadata = conversation_data.get("metadata")
                
                conversation_id = self.create_conversation(title, metadata)
                
                # Add messages
                for message in conversation_data.get("messages", []):
                    role = message.get("role", "user")
                    content = message.get("content", "")
                    
                    self.add_message(conversation_id, role, content)
                
                logger.info(f"Imported conversation with ID {conversation_id}")
                return conversation_id
            
            else:
                logger.warning(f"Unsupported import format: {format}")
                return None
        except Exception as e:
            logger.error(f"Failed to import conversation: {e}")
            return None


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize memory
    memory = Memory("test_conversations.db")
    
    # Create a new conversation
    conversation_id = memory.create_conversation("Test Conversation")
    
    # Add messages
    memory.add_message(conversation_id, "system", "You are a helpful AI assistant.")
    memory.add_message(conversation_id, "user", "Hello! Can you help me with something?")
    memory.add_message(conversation_id, "assistant", "Of course! I'm here to help. What do you need assistance with?")
    
    # Get conversation data
    conversation = memory.get_conversation(conversation_id)
    print(f"Conversation: {conversation['title']}")
    print(f"Messages: {len(conversation['messages'])}")
    
    # Export conversation
    exported = memory.export_conversation(conversation_id, format="txt")
    print("\nExported conversation:")
    print(exported)
