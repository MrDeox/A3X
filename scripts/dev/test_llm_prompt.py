import sys
import os
import asyncio
from pathlib import Path

# Add project root to sys.path to allow imports like a3x.core
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    # Use absolute import based on the adjusted sys.path
    from a3x.core.llm_interface import LLMInterface
except ImportError:
    print("Error: Could not import LLMInterface. Ensure the script is run from the project root or the path is correctly set.")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

# --- LLM Setup ---
# Instantiate the LLM Interface.
try:
    llm = LLMInterface()
    # Simple check if URL was configured
    if not hasattr(llm, 'llm_url') or not llm.llm_url:
        print(f"Error: LLMInterface did not configure an LLM URL. Check environment/config.")
        sys.exit(1)
except Exception as e:
    print(f"Error instantiating LLMInterface: {e}")
    sys.exit(1)

# --- Prompt Definition (Split into roles) ---
system_prompt = """SYSTEM: Você está atuando como um assistente simbólico. Quando receber um erro como:
'refletir sobre erro ao executar comando ... Fragmento não encontrado'

Você deve sugerir ações como:
- listar fragmentos disponíveis
- criar um novo fragmento com nome apropriado
- ou qualquer outra correção relevante

Exemplo de saída esperada:
listar fragmentos
criar fragmento 'novo_teste' tipo 'neural'

Agora processe a seguinte entrada:"""

user_prompt = "USER: refletir sobre erro ao executar comando '{'type': 'reflect_fragment', 'fragment_id': 'aprendiz', ...}': Fragmento não encontrado"""

# Format messages for call_llm
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]

# --- Async Main Function to Call LLM ---
async def main():
    print("--- Sending Formatted Messages to LLM via call_llm ---")
    print(f"System Prompt: {system_prompt}")
    print(f"User Prompt: {user_prompt}")
    print("-----------------------------------------------------")
    print("\n--- LLM Response ---")
    full_response = ""
    try:
        # Call the async generator and collect the response
        async for chunk in llm.call_llm(messages=messages, stream=False):
            full_response += chunk
        print(full_response)
    except Exception as e:
        print(f"\nError during LLM call_llm: {e}")
    print("--------------------")

# --- Run the async main function ---
if __name__ == "__main__":
    asyncio.run(main()) 