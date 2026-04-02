import asyncio
import json
from dotenv import load_dotenv

load_dotenv()
from graph import execute_query

def run_single():
    query = "A song with slow doom metal mood but 170 BPM upbeat tempo"
    res = execute_query(query)
    
    with open("debate_proof.json", "w", encoding="utf-8") as f:
        json.dump({
            "Agents Called": res.get("agents_called"),
            "Tokens": res.get("token_usage"),
            "Final JSON": res.get("final_answer")
        }, f, indent=2)

if __name__ == "__main__":
    run_single()
