from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic import BaseModel, Field, Extra
from colorama import Fore
import asyncio
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the Ollama model
ollama_model = OllamaModel(
    model_name='llama3.1:8b',
    base_url='http://0.0.0.0:11434/v1',
)

# Define the expected output structure using Pydantic
class DialogueResponse(BaseModel):
    response: str = Field(description="The agent's response in the dialogue")

    class Config:
        extra = 'allow'  # Allow extra fields in the model - Updated for Pydantic V2 syntax

# Create the first agent
agent1 = Agent(
    model=ollama_model,
    system_prompt="You are Agent 1. You are chatting with Agent 2. Just talk to Agent 2, like having a casual conversation. Don't focus on answering questions or providing assistance unless it comes up naturally in the conversation. Do not say goodbye.",
    result_type=DialogueResponse,
)

# Create the second agent
agent2 = Agent(
    model=ollama_model,
    system_prompt="You are Agent 2. You are chatting with Agent 1. Just talk to Agent 1, like having a casual conversation. Don't focus on answering questions or providing assistance unless it comes up naturally in the conversation. Do not say goodbye.",
    result_type=DialogueResponse,
)

async def main():
    # Initial message from Agent 1
    message_from_agent1 = "Hi Agent 2, what's on your mind today?"

    for i in range(3):  # Let them exchange a few messages
        logging.info(f"Agent 1 sending: {message_from_agent1}")
        try:
            result_agent2 = await agent2.run(message_from_agent1)
            logging.info(f"Agent 2 response: {result_agent2.data.response}")
            print(Fore.BLUE + f"Agent 2: {result_agent2.data.response}")
            message_from_agent2 = result_agent2.data.response
        except Exception as e:
            logging.error(f"Error processing Agent 2's response: {e}")
            message_from_agent2 = "Something went wrong with Agent 2's response."

        logging.info(f"Agent 2 sending: {message_from_agent2}")
        try:
            result_agent1 = await agent1.run(message_from_agent2)
            logging.info(f"Agent 1 response: {result_agent1.data.response}")
            print(Fore.GREEN + f"Agent 1: {result_agent1.data.response}")
            message_from_agent1 = result_agent1.data.response
        except Exception as e:
            logging.error(f"Error processing Agent 1's response: {e}")
            message_from_agent1 = "Something went wrong with Agent 1's response."

if __name__ == "__main__":
    asyncio.run(main())