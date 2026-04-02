import asyncio
import json
from dotenv import load_dotenv

# Load API Key from .env
load_dotenv()

from main import evaluate

async def run_demo():
    print("Initiating Wubble System Evaluation...")
    print("Running 5 concurrent stress tests through the LangGraph Supervisor...\n")
    
    result = await evaluate()
    
    print("--- EVALUATION COMPLETE ---\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(run_demo())
