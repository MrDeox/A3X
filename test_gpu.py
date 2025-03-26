import llama_cpp
import time
import os

def test_gpu_inference():
    print("Iniciando teste de inferência com GPU AMD...")
    
    # Configuração do modelo com suporte a GPU
    llm = llama_cpp.Llama(
        model_path="models/dolphin-2.2.1-mistral-7b.Q4_K_M.gguf",
        n_ctx=2048,  # Tamanho do contexto
        n_gpu_layers=32,  # Número de camadas para GPU (ajustado para o número real de camadas do modelo)
        n_threads=8,  # Threads para CPU
        n_batch=512,  # Tamanho do batch para otimização
        offload_kqv=True,  # Habilita offload de KQV para GPU
        f16_kv=True,  # Usa FP16 para KV cache
        use_mmap=True,  # Usa mmap para carregar o modelo
        use_mlock=False,  # Não usa mlock
        embedding=False,  # Não precisa de embeddings
        logits_all=False,  # Não precisa de todos os logits
    )
    
    # Prompt de teste
    prompt = "Explique o que é inteligência artificial em 3 linhas:"
    
    print("\nExecutando inferência...")
    start_time = time.time()
    
    # Geração do texto
    output = llm(
        prompt,
        max_tokens=100,
        stop=["\n\n"],
        echo=False,
        temperature=0.7,
        top_p=0.95,
        repeat_penalty=1.1,
        top_k=40
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\nResultado:")
    print(output['choices'][0]['text'])
    print(f"\nTempo de inferência: {duration:.2f} segundos")
    
    # Verificar informações do modelo
    print("\nInformações do modelo:")
    print(f"Contexto: {llm.n_ctx}")
    print(f"Threads CPU: {llm.n_threads}")
    print(f"Tamanho do batch: {llm.n_batch}")

if __name__ == "__main__":
    # Configurar variáveis de ambiente para ROCm 6.3
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
    os.environ["HIP_VISIBLE_DEVICES"] = "0"
    os.environ["HIPBLAS_MAIN_MEM_PERCENT"] = "95"
    os.environ["HIP_LAUNCH_BLOCKING"] = "1"
    os.environ["HSA_ENABLE_SDMA"] = "0"
    os.environ["ROCBLAS_LAYER"] = "1"
    os.environ["AMD_OCL_WAIT_COMMAND"] = "1"
    
    test_gpu_inference() 