"""
Memory Store for long-term user information.
Supports four memory types: preference, profile, constraint, fact.
Stores in SQLite (memory.sqlite) with retrieve/upsert capabilities.
"""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal


MemoryType = Literal["preference", "profile", "constraint", "fact"]


@dataclass
class MemoryEntry:
    memory_type: MemoryType
    key: str
    value: str
    

class MemoryStore:
    """SQLite-backed memory storage with structured extract and retrieve."""
    
    def __init__(self, db_path: str = "memory.sqlite"):
        self.db_path = Path(db_path)
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite schema."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_type TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(memory_type, key)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_type_key 
                ON memories(memory_type, key)
            """)
            conn.commit()
        finally:
            conn.close()
    
    def upsert(self, memory_type: MemoryType, key: str, value: str):
        """Insert or update a memory entry."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT INTO memories (memory_type, key, value, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(memory_type, key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
            """, (memory_type, key, value))
            conn.commit()
        finally:
            conn.close()
    
    def upsert_many(self, entries: List[MemoryEntry]):
        """Batch upsert multiple entries."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.executemany("""
                INSERT INTO memories (memory_type, key, value, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(memory_type, key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
            """, [(e.memory_type, e.key, e.value) for e in entries])
            conn.commit()
        finally:
            conn.close()
    
    def retrieve(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """
        Simple keyword-based retrieve (no vector search).
        Returns top_k memories where key or value contains any query token.
        """
        tokens = [t.lower() for t in query.split() if t]
        if not tokens:
            return []
        
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("""
                SELECT memory_type, key, value FROM memories
                ORDER BY updated_at DESC
            """)
            rows = cursor.fetchall()
        finally:
            conn.close()
        
        results = []
        for memory_type, key, value in rows:
            text = f"{key} {value}".lower()
            score = sum(text.count(t) for t in tokens)
            if score > 0:
                results.append((score, MemoryEntry(memory_type, key, value)))
        
        results.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in results[:top_k]]
    
    def format_for_prompt(self, entries: List[MemoryEntry]) -> str:
        """Format memory entries for injection into system context."""
        if not entries:
            return ""
        
        lines = ["[User Memory]"]
        for e in entries:
            lines.append(f"- {e.memory_type.capitalize()}: {e.key} = {e.value}")
        return "\n".join(lines)


def extract_memories_with_llm(llm, conversation_history: str) -> List[MemoryEntry]:
    """
    Use an extra LLM call to extract structured memories from conversation.
    Returns a list of MemoryEntry objects.
    """
    extraction_prompt = f"""Analyze the following conversation and extract user-related information.
Return a JSON array of memory objects. Each object should have:
- "memory_type": one of ["preference", "profile", "constraint", "fact"]
- "key": a short label (e.g., "language", "role", "tech_stack")
- "value": the specific information

Only extract information that is explicitly stated or strongly implied about the user.

Examples:
- preference: language=Chinese, style=concise
- profile: role=data scientist, project=ML research
- constraint: deadline=next week, budget=limited
- fact: uses Python 3.11, works on Windows

Conversation:
{conversation_history}

Return valid JSON array only, no markdown or extra text:"""

    try:
        response = llm.invoke(extraction_prompt)
        content = response.content.strip()
        
        # Remove markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
        
        data = json.loads(content)
        
        if not isinstance(data, list):
            return []
        
        entries = []
        valid_types = {"preference", "profile", "constraint", "fact"}
        for item in data:
            if not isinstance(item, dict):
                continue
            mem_type = item.get("memory_type", "")
            if mem_type not in valid_types:
                continue
            key = str(item.get("key", "")).strip()
            value = str(item.get("value", "")).strip()
            if key and value:
                entries.append(MemoryEntry(mem_type, key, value))
        
        return entries
    except Exception:
        # If extraction fails, return empty list
        return []
