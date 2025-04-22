from a3x.core import mem
import asyncio
import logging

# Configure logging basic setup if not already configured elsewhere
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def main():
    command = "RESPONDER 'Olá, mundo simbólico!'"
    print(f"\n🚀 Executando comando A³L: {command}")
    try:
        result = await mem.execute(command)
        print("\n🧠 Resultado:")
        print(result)
    except Exception as e:
        logging.exception("Erro ao executar mem.execute")
        print(f"\n❌ Erro: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 