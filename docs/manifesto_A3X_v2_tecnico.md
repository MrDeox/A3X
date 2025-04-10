# Estratégias para Evolução Cognitiva Autônoma em Sistemas de IA Local

Este relatório técnico apresenta uma análise aprofundada e recomendações para o desenvolvimento do sistema A³X, com foco em implementar capacidades de aprendizado contínuo e autoevolução em ambientes com recursos computacionais limitados. A análise examina os desafios fundamentais apresentados no manifesto e propõe soluções práticas baseadas em tecnologias atuais e emergentes.

## Arquitetura para Aprendizado Contínuo Local

O aprendizado contínuo representa um avanço significativo em relação aos modelos de aprendizado tradicionais, permitindo que sistemas de IA se adaptem a novos dados sem retreinamento completo. Diferentemente de abordagens convencionais que dependem de conjuntos de dados estáticos, o aprendizado contínuo atualiza iterativamente os parâmetros do modelo para refletir novas distribuições nos dados[20].

### Coleta e Seleção de Dados

Para implementar um sistema eficaz de aprendizado contínuo no A³X, recomendamos:

1. **Captura seletiva de interações**: Estabeleça um sistema de logging que registre interações baseadas em classificações de relevância, focando em:
   - Comandos que resultaram em erros ou falhas
   - Interações com alta complexidade computacional
   - Padrões de uso recorrentes
   - Casos extremos ou atípicos que podem representar novos domínios

2. **Pipeline de processamento de dados**: Estruture um pipeline de machine learning completo que inclua:
   - Coleta e pré-processamento de dados brutos de fontes variadas
   - Análise exploratória para identificar padrões e anomalias
   - Pré-processamento para normalização e codificação
   - Seleção de características mais relevantes para o problema[6]

### Critérios para Treinamento Incremental

Para estabelecer quando e como realizar o aprendizado incremental:

1. **Gatilhos para treinamento**:
   - Acúmulo de volume mínimo de dados novos significativos (ex: 100-500 exemplos)
   - Detecção de queda de desempenho em tarefas específicas
   - Falhas sistemáticas em novos tipos de solicitações
   - Ciclos temporais predefinidos (ex: treinamento noturno quando o sistema está em baixo uso)

2. **Balanceamento de dados**:
   - Mantenha representação equilibrada entre diferentes tipos de tarefas
   - Implemente técnicas de amostragem para prevenir vieses em direção a tarefas mais frequentes
   - Utilize técnicas de augmentação de dados para classes sub-representadas

## Técnicas de Treinamento Viáveis para Hardware Limitado

A implementação de treinamento em hardware limitado como a GPU AMD RX 6400 (4GB VRAM) requer abordagens altamente otimizadas:

### Métodos de Quantização e Fine-tuning

1. **QLoRA para treinamento eficiente**:
   - Utilize quantização de 4-bit com técnicas como double quantization para reduzir requisitos de memória[4]
   - Implemente gradient checkpointing para minimizar o uso de VRAM durante backpropagation
   - Configure `BitsAndBytesConfig` com parâmetros otimizados como:
     ```python
     bnb_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_use_double_quant=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_compute_dtype=torch.bfloat16
     )
     ```

2. **Conversão eficiente para GGUF**:
   - Utilize a biblioteca Unsloth para conversão direta de modelos LoRA para GGUF:
     ```python
     from unsloth import FastLanguageModel
     model, tokenizer = FastLanguageModel.from_pretrained("lora_model")
     model.save_pretrained_gguf("gguf_model", tokenizer, quantization_method = "q4_k_m")
     ```
   - Para abordagens alternativas sem dependências adicionais:
     ```python
     from peft import PeftConfig, PeftModel
     from transformers import AutoModelForCausalLM, AutoTokenizer
     model = AutoModelForCausalLM.from_pretrained(base_model_name)
     model = PeftModel.from_pretrained(model, adapter_model_name)
     model = model.merge_and_unload()
     model.save_pretrained("merged_adapters")
     ```

3. **Otimização para RX 6400**:
   - A GPU RX 6400 com 4GB de VRAM é viável para inferência de LLM, especialmente quando combinada com modelos quantizados e pequenos como o Gemma 3B[9][10]
   - Devido ao baixo consumo energético e tamanho compacto, é adequada para sistemas que precisam operar continuamente[9]

### Inovações em Treinamento Local

1. **Treinamento incremental adaptativo**:
   - Implemente freeze seletivo de camadas, mantendo fixas todas as camadas exceto as últimas
   - Utilize adapters de tarefa específica que podem ser treinados independentemente
   - Explore técnicas de destilação contínua, onde um "modelo professor" temporário é criado a partir do modelo base + novas interações, e então destilado de volta para o modelo principal

2. **Técnicas de RLHF adaptadas localmente**:
   - Adapte o conceito de RLHF (Reinforcement Learning from Human Feedback) para um contexto autônomo[3]
   - Implemente um sistema de recompensa baseado em métricas objetivas como:
     - Tempo de conclusão de tarefas
     - Economia de recursos
     - Precisão das respostas em tarefas verificáveis
   - Utilize contrastes entre diferentes versões do modelo para medir melhoria relativa

## Modularidade do Conhecimento

A implementação de conhecimento modular através de LoRAs temáticas permite um sistema mais adaptável e eficiente:

### Arquitetura Modular

1. **LoRAs temáticas e especializadas**:
   - Desenvolva adaptadores LoRA específicos para domínios distintos (codificação, compreensão de texto, raciocínio matemático)
   - Mantenha um modelo base quantizado compartilhado com LoRAs como camadas adaptativas
   - Implemente um sistema dinâmico que carregue apenas os adaptadores LoRA relevantes para o contexto atual[16][18]

2. **Gestão de ativação contextual**:
   - Crie um classificador leve que determine quais módulos de conhecimento ativar baseado no input
   - Mantenha metadados sobre o domínio de especialidade e performance de cada LoRA
   - Implemente um mecanismo de votação quando múltiplos módulos são aplicáveis, priorizando aqueles com melhor desempenho histórico

### Banco de Dados Vetorial para Memória Semântica

A integração de FAISS com SQLite fornece uma solução eficiente para memória semântica:

1. **Implementação FAISS+SQLite**:
   - Mantenha vetores no FAISS e metadados no SQLite como uma solução eficiente para RAG (Retrieval-Augmented Generation)[2]
   - Estruture a busca para associar IDs de vetores no FAISS com metadados no SQLite:
     ```python
     # Exemplo de workflow
     # 1. Consulta FAISS para encontrar vizinhos mais próximos
     # 2. Use os IDs resultantes para consultar SQLite e recuperar metadados
     ```

2. **Otimização de consultas**:
   - Indexe eficientemente os metadados no SQLite para recuperação rápida
   - Implemente caching de resultados frequentes
   - Utilize janelas deslizantes temporais para priorizar informações mais recentes

## Integração Técnica e Conversão de Modelos

A criação de um pipeline eficiente para converter modelos treinados em formatos utilizáveis é essencial:

### Pipeline de Conversão GGUF

1. **Estratégia de conversão direta**:
   - Utilize o recente suporte a adaptadores LoRA no formato GGUF, permitindo carregar adaptadores diretamente sobre modelos quantizados[16]
   - Implemente um processo automatizado de conversão após cada ciclo de treinamento:
     ```bash
     # Conversão para GGUF
     python llama.cpp/convert.py caminho_para_modelo --outfile modelo.gguf --outtype q4_k_m
     ```

2. **Otimização de parâmetros de quantização**:
   - Experimente diferentes métodos de quantização como Q4_K_M (equilibrado) ou Q5_K_S (baixa perda de qualidade)[18]
   - Considere o uso de matrizes de calibração (imatrix) durante a conversão para GGUF para melhorar a precisão da quantização[16]

### Carregamento de LoRAs em Modelos Quantizados

1. **Técnicas de carregamento eficiente**:
   - Aproveite o suporte recente do llama.cpp para carregar adaptadores LoRA diretamente em modelos GGUF[16]
   - Implemente um sistema de gerenciamento de cache para alternar entre diferentes adaptadores LoRA minimizando a sobrecarga de memória

## Estratégias de Autoavaliação e Autoevolução

A implementação de mecanismos robustos de autoavaliação é essencial para a evolução autônoma:

### Métricas de Autoavaliação

1. **Sistema de benchmarking interno**:
   - Mantenha um conjunto de tarefas de referência com respostas esperadas
   - Avalie periodicamente o desempenho nesses benchmarks
   - Compare métricas como precisão, tempo de resposta e uso de recursos entre versões do modelo

2. **Análise de falhas**:
   - Implemente logging detalhado de erros e falhas
   - Categorize tipos de falhas e áreas problemáticas
   - Priorize treinamento em áreas com falhas recorrentes

### Ciclos de Autoevolução

1. **Processo iterativo de melhoria**:
   - Estabeleça ciclos regulares: Execução → Coleta de dados → Avaliação → Treinamento → Validação
   - Implemente critérios claros para aceitar ou rejeitar atualizações de modelo baseados em métricas objetivas
   - Mantenha múltiplas versões do modelo para comparação e fallback

2. **Gestão de recursos adaptativos**:
   - Ajuste dinamicamente os requisitos de treinamento com base na disponibilidade de recursos
   - Priorize tarefas de treinamento durante períodos de baixa utilização
   - Implemente mecanismos de economia de energia que otimizem o uso da GPU

## Conclusão

A implementação de um sistema de evolução cognitiva autônoma local como o A³X representa um avanço significativo no campo da inteligência artificial. Através da combinação de técnicas de aprendizado contínuo, otimização para hardware limitado, modularidade de conhecimento e mecanismos robustos de autoavaliação, é possível criar um sistema que genuinamente evolui através de suas próprias experiências.

As recomendações apresentadas neste relatório oferecem um caminho viável para transformar o A³X de um sistema de execução de comandos para um agente cognitivo genuinamente autônomo e adaptativo. A abordagem modular, com foco na eficiência de recursos e treinamento incremental, permite maximizar o potencial mesmo com as limitações de hardware impostas.

O sucesso deste projeto pode redefinir como concebemos sistemas de IA locais, demonstrando que a inteligência significativa não depende necessariamente de recursos computacionais massivos ou dados centralizados, mas pode emergir através de ciclos cuidadosamente projetados de experiência, aprendizado e adaptação.

Citations:
[1] https://www.semanticscholar.org/paper/070c39a6f308e2a0caa8321b8e90c1f2955571e1
[2] https://github.com/maylad31/vector_sqlite
[3] https://www.servicenow.com/br/ai/what-is-rlhf.html
[4] https://www.reddit.com/r/LocalLLaMA/comments/1amjx77/how_to_convert_my_finetuned_model_to_gguf/
[5] https://www.substratus.ai/blog/converting-hf-model-gguf-model
[6] https://awari.com.br/machine-learning-pipeline-desenvolvimento-de-pipelines-de-machine-learning-2/
[7] https://www.ibm.com/br-pt/topics/machine-learning-pipeline
[8] https://www.cloudflare.com/pt-br/learning/ai/what-is-quantization/
[9] https://forum.level1techs.com/t/radeon-rx-6400-for-home-assistant-ai-acceleration/220383
[10] https://exame.com/inteligencia-artificial/google-lanca-colecao-de-ia-aberta-gemma-3-com-ate-27-bilhoes-de-parametros/
[11] https://cheatsheet.md/pt/llm-leaderboard/how-to-run-mistral-locally
[12] https://www.semanticscholar.org/paper/e7af38691f09e541e9df16a7ca60f0120ea1de5c
[13] https://www.reddit.com/r/Oobabooga/comments/1d432y2/merge_trained_lora_to_original_model_and_then_to/
[14] https://pt.docs.gaianet.ai/tutorial/llamacpp/
[15] https://www.semanticscholar.org/paper/077ed544f2b213c44860bdec3e98b7f41d6125d0
[16] https://kaitchup.substack.com/p/fast-inference-with-gguf-lora-adapters
[17] https://www.semanticscholar.org/paper/ae86677cfd483b48b44d91de202e730c34a350cd
[18] https://huggingface.co/TheBloke/Llama-2-7B-LoRA-Assemble-GGUF
[19] https://www.semanticscholar.org/paper/ec4918ed50abb3510795f389f780da31bea0e3f8
[20] https://www.datacamp.com/pt/blog/what-is-continuous-learning
[21] https://www.semanticscholar.org/paper/e4d135e243d05b6a09fab6071d7e5f46b9918af5
[22] https://www.semanticscholar.org/paper/069a941388b93a42b52fa52276092671979e2804
[23] https://www.semanticscholar.org/paper/7a91dfcdb190edfd7f0e7c5c5158f23e59946e6c
[24] https://www.semanticscholar.org/paper/8e13cf0fe553b8652ffdc3d91f97aabd3aef2961
[25] https://www.reddit.com/r/concursospublicos/comments/1ew7evt/a_quest%C3%A3o_de_intelig%C3%AAncia_artificial/
[26] https://www.reddit.com/r/LocalLLaMA/comments/1jeoocb/technical_discussion_local_ai_deployment_market/?tl=pt-br
[27] https://www.reddit.com/r/LocalLLaMA/comments/193362r/new_model_openchat_35_update_0106/?tl=pt-br
[28] https://www.reddit.com/r/cscareerquestions/comments/1fatrae/how_much_continuous_learning_is_actually_required/?tl=pt-br
[29] https://www.reddit.com/r/LocalLLaMA/comments/1bdzw87/which_quantization_method_you_useprefer/?tl=pt-br
[30] https://www.reddit.com/r/homeschool/comments/1j31cq1/are_these_the_most_structured_traditional_programs/?tl=pt-br
[31] https://www.reddit.com/r/brdev/comments/1inqe4n/staff_software_engineer_com_muito_tempo_livre/
[32] https://www.reddit.com/r/ArtificialInteligence/comments/1j0jgow/is_it_possible_that_large_language_models_learned/?tl=pt-br
[33] https://www.reddit.com/r/cscareerquestions/comments/1i3om00/is_having_a_always_learning_mind_set_in_a_mid/?tl=pt-br
[34] https://www.reddit.com/r/LocalLLaMA/comments/1agbf5s/gpu_requirements_for_llms/?tl=pt-br
[35] https://www.reddit.com/r/intj/comments/1h9i9xi/collaborative_intj_manifesto/?tl=pt-br
[36] https://www.reddit.com/r/LocalLLaMA/comments/1ehlazq/introducing_sqlitevec_v010_a_vector_search_sqlite/?tl=pt-br
[37] https://www.reddit.com/r/selfhosted/comments/1juel4k/build_yout_local_custom_ai_self_with_second_me/?tl=pt-br
[38] https://www.reddit.com/r/StableDiffusion/comments/1enuib1/i_trained_an_anime_aesthetic_lora_for_flux/?tl=pt-br
[39] https://www.reddit.com/r/ChatGPTPro/comments/1jcxa6w/does_chatgpt_know_your_iq_based_off_of_your/?tl=pt-br
[40] https://www.reddit.com/r/LocalLLaMA/comments/1isiyl1/stop_overengineering_ai_apps_just_use_postgres/?tl=pt-br
[41] https://www.reddit.com/r/github/comments/1juek8y/build_a_local_custom_ai_self_with_second_me_now/?tl=pt-br
[42] https://www.reddit.com/r/LocalLLaMA/comments/1j7t18m/framework_and_digits_suddenly_seem_underwhelming/?tl=pt-br
[43] https://www.redhat.com/pt-br/topics/ai/lora-vs-qlora
[44] https://pepsic.bvsalud.org/pdf/psie/n46/n46a09.pdf
[45] https://python.langchain.com/v0.1/docs/integrations/vectorstores/sqlitevss/
[46] https://datascience.eu/pt/wiki-pt/o-que-e-rlhf/
[47] https://wadhwanifoundation.org/pt/modelo-para-o-sucesso-organizacional-como-criar-uma-cultura-de-aprendizado-continuo-em-sua-organizacao/
[48] https://lume.ufrgs.br/bitstream/handle/10183/196818/001096611.pdf?sequence=1
[49] https://www.hashtagtreinamentos.com/banco-de-dados-em-python
[50] https://www.unite.ai/pt/what-is-reinforcement-learning-from-human-feedback-rlhf/
[51] https://www.youtube.com/watch?v=gSiicHeuAGs
[52] https://roxpartner.com/llms-on-premises-vale-a-pena-rodar-modelos-de-ia-localmente/
[53] https://www.scielo.br/j/epsic/a/BZ3L3cthHHxht7VPt3ccPLJ/?lang=pt
[54] https://aws.amazon.com/pt/what-is/reinforcement-learning-from-human-feedback/
[55] https://www.reddit.com/r/LocalLLaMA/comments/1b0p646/how_do_i_convert_my_pytorch_model_to_gguf_format/?tl=pt-br
[56] https://www.reddit.com/r/pcgaming/comments/1i52ofi/inside_dlss_4_nvidia_machine_learning_the_bryan/?tl=pt-br
[57] https://www.reddit.com/r/LocalLLaMA/comments/1cx6ozp/llamacpp_gguf_wrapper/?tl=pt-br
[58] https://www.reddit.com/r/cpp_questions/comments/14lut75/is_c_a_good_language_for_ai_machine_learning/?tl=pt-br
[59] https://www.reddit.com/r/LocalLLaMA/comments/1h3fxey/convert_multimodal_model_to_gguf_to_run_locally/?tl=pt-br
[60] https://www.reddit.com/r/LocalLLaMA/comments/18ptf9t/how_to_convert_lora_tuned_stablelm_zephyr_3b_to/?tl=pt-br
[61] https://www.reddit.com/r/LocalLLaMA/comments/1icwys9/berkley_ai_research_team_claims_to_reproduce/?tl=pt-br
[62] https://www.reddit.com/r/LocalLLaMA/comments/1d4rica/llamacpp_removes_convertpy_in_favor_of/?tl=pt-br
[63] https://www.reddit.com/r/LocalLLaMA/comments/1amjx77/how_to_convert_my_finetuned_model_to_gguf/?tl=pt-br
[64] https://www.reddit.com/r/java/comments/1c4gkll/java_use_in_machine_learning/?tl=pt-br
[65] https://www.reddit.com/r/LocalLLaMA/comments/16m5ciz/how_to_use_and_reliable_is_the_ggml_to_gguf/?tl=pt-br
[66] https://www.reddit.com/r/LocalLLaMA/comments/1e1rhuu/llama_cpp_lora_adapter_swap/?tl=pt-br
[67] https://www.reddit.com/r/dataanalysis/comments/1hyca66/are_we_also_going_to_be_expected_to_work_on/?tl=pt-br
[68] https://www.reddit.com/r/ollama/comments/1alx0hz/how_to_convert_to_gguf_my_finetuned_model_or_make/?tl=pt-br
[69] https://www.reddit.com/r/ollama/comments/1dorjxf/how_can_i_apply_lora_adapters_to_models_in_ollama/?tl=pt-br
[70] https://www.reddit.com/r/LocalLLaMA/comments/1ilfhyl/is_nvidia_becoming_a_bottleneck_for_ai_advancement/?tl=pt-br
[71] https://learn.microsoft.com/pt-br/azure/architecture/ai-ml/
[72] https://www.ibm.com/br-pt/think/topics/gguf-versus-ggml
[73] https://github.com/ggerganov/llama.cpp/issues/3953
[74] https://sevenpublicacoes.com.br/editora/article/download/6487/11737/25844
[75] https://huggingface.co/google/madlad400-3b-mt/discussions/7
[76] https://www.ibm.com/br-pt/think/topics/rlhf
[77] https://blog.dsacademy.com.br/a-inteligencia-artificial-pode-aprender-a-aprender/
[78] https://github.com/ggml-org/llama.cpp/discussions/2948
[79] https://www.datacamp.com/pt/blog/what-is-reinforcement-learning-from-human-feedback
[80] https://blog.nvidia.com.br/blog/processamento-acelerado-de-dados-inovacao-ia-todos-os-setores/
[81] https://www.reddit.com/r/LocalLLaMA/comments/1fqwler/show_me_your_ai_rig/?tl=pt-br
[82] https://www.reddit.com/r/LocalLLaMA/comments/1hgmh9u/what_doesnt_exist_but_should_and_you_wish_did_and/?tl=pt-br
[83] https://www.reddit.com/r/compsci/comments/1fvar2t/revolutionizing_ai_hardware_ultrascalable_1bit/?tl=pt-br
[84] https://www.reddit.com/r/htpc/comments/10qcs3c/added_a_passivelycooled_rx_6400_to_my_completely/?tl=pt-br
[85] https://www.reddit.com/r/LocalLLaMA/comments/1cuq3gf/are_you_building_a_rig_as_a_hobbyist/?tl=es-419
[86] https://www.reddit.com/r/LocalLLaMA/comments/1hvj4wn/nvidia_announces_3000_personal_ai_supercomputer/?tl=pt-br
[87] https://www.reddit.com/r/LocalLLaMA/comments/1cpel7z/can_we_update_this_llm_gpu_buying_guide_new/?tl=pt-br
[88] https://www.reddit.com/r/LocalLLaMA/comments/16xq65o/about_to_buy_hardware_for_7k/?tl=es-419
[89] https://www.reddit.com/r/ollama/comments/1icyl2l/will_there_ever_be_uncensored_self_hosted_ai/?tl=pt-br
[90] https://www.reddit.com/r/LocalLLaMA/comments/1d5axvx/while_nvidia_crushes_the_ai_data_center_space/?tl=pt-br
[91] https://www.reddit.com/r/LocalLLaMA/comments/1cfdbpf/rag_is_all_you_need/?tl=pt-br
[92] https://www.reddit.com/r/MachineLearning/comments/11w03sy/r_unlock_the_power_of_personal_ai_introducing/?tl=es-419
[93] https://www.reddit.com/r/MachineLearning/comments/1hpg91o/d_why_mamba_did_not_catch_on/?tl=pt-br
[94] https://www.reddit.com/r/LocalLLaMA/comments/1drnbq7/advice_on_building_a_gpu_pc_for_llm_with_a_1500/?tl=pt-br
[95] https://www.reddit.com/r/MachineLearning/comments/1iupnet/d_have_we_hit_a_scaling_wall_in_base_models_non/?tl=pt-br
[96] https://www.engenhariahibrida.com.br/post/fim-das-limitacoes-inteligencia-artificial
[97] https://iot-labs.io/portfolio/modulo-lorawan-smart-modular-technologies/
[98] https://repositorio.animaeducacao.com.br/bitstreams/c1dcfd40-f6c7-4173-8d3d-9b770cff9e89/download
[99] https://www.amd.com/pt/products/graphics/radeon-ai.html
[100] https://www.robocore.net/lorawan/modulo-lorawan-bee-v2-chip-antenna?newlang=english
[101] https://www.ime.usp.br/~vwsetzer/IAtrad.html
[102] https://www.gigabyte.com/Graphics-Card/GV-R64EAGLE-4GD
[103] https://www.gov.br/mcti/pt-br/acompanhe-o-mcti/lei-de-tics/lei-de-tics-ppi-projetos
[104] https://inatel.br/brasil6g/documents/brasil6g-meta-2-atividade-2-3-ia.pdf
[105] https://www.cloudskillsboost.google/paths/17/course_templates/1036/video/513736?locale=pt_BR
[106] https://www.reddit.com/r/LocalLLaMA/comments/1j9dkvh/gemma_3_release_a_google_collection/?tl=pt-br
[107] https://www.reddit.com/r/LocalLLaMA/comments/1ig2cm2/mistralsmall24binstruct2501_is_simply_the_best/?tl=pt-br
[108] https://www.reddit.com/r/LocalLLaMA/comments/1jsq1so/smaller_gemma3_qat_versions_12b_in_8gb_and_27b_in/?tl=pt-br
[109] https://www.reddit.com/r/LocalLLaMA/comments/1igpedw/mistral_small_3_redefining_expectations/?tl=pt-br
[110] https://www.reddit.com/r/LocalLLaMA/comments/1jhl6jp/gemma3_is_outperforming_a_ton_of_models_on/?tl=pt-br
[111] https://www.reddit.com/r/LocalLLaMA/comments/1idny3w/mistral_small_3/?tl=pt-br
[112] https://www.reddit.com/r/LocalLLaMA/comments/1j9reim/lm_studio_updated_with_gemma_3_gguf_support/?tl=pt-br
[113] https://www.reddit.com/r/LocalLLaMA/comments/1iqmwsl/i_pay_for_chatgpt_20_usd_i_specifically_use_the/?tl=pt-br
[114] https://www.reddit.com/r/LocalLLaMA/comments/1j9hsfc/gemma_3_ggufs_recommended_settings/?tl=pt-br
[115] https://www.reddit.com/r/LocalLLaMA/comments/1glw1rs/computer_spec_for_running_large_ai_model_70b/?tl=pt-br
[116] https://www.reddit.com/r/LocalLLaMA/comments/1jgau52/gemma_3_27b_vs_mistral_24b_vs_qwq_32b_i_tested_on/?tl=pt-br
[117] https://www.reddit.com/r/ollama/comments/1idqxto/why_are_all_local_ai_models_so_bad_no_one_talks/?tl=pt-br
[118] https://www.reddit.com/r/LocalLLaMA/comments/1jhwr2p/next_gemma_versions_wishlist/?tl=pt-br
[119] https://www.reddit.com/r/LocalLLaMA/comments/1ixvlop/do_you_think_that_mistral_worked_to_develop_saba/?tl=pt-br
[120] https://www.reddit.com/r/LocalLLaMA/comments/1jdasng/heads_up_if_youre_using_gemma_3_vision/?tl=pt-br
[121] https://www.reddit.com/r/LocalLLaMA/comments/1hqak1f/whats_your_primary_local_llm_at_the_end_of_2024/?tl=pt-br
[122] https://www.reddit.com/r/Bard/comments/1jqmbq5/gemma_3_qat_3x_less_memory_same_performance/?tl=pt-br
[123] https://www.reddit.com/r/selfhosted/comments/1b0wqgu/building_my_own_ai_server_for_machine_learning/?tl=pt-br
[124] https://www.reddit.com/r/LocalLLaMA/comments/1j9wgv2/gemma_3_appreciation_post/?tl=pt-br
[125] https://www.reddit.com/r/MistralAI/comments/1j4017j/training_data_of_mistral/?tl=pt-br
[126] https://ai.google.dev/gemma
[127] https://rockcontent.com/br/blog/o-que-e-o-mistral-ai/
[128] https://www.robertodiasduarte.com.br/deepseek-r1-a-revolucao-da-ia-de-codigo-aberto/
[129] https://developers.googleblog.com/pt-br/gemma-explained-paligemma-architecture/
[130] https://meetcody.ai/pt-br/blog/genai-da-empresa-francesa-mistral-ai-o-melhor-assistente-comercial-de-ia/
[131] https://www.youtube.com/watch?v=wdnbxrPqbO0
[132] https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora
[133] https://www.onlyoffice.com/blog/pt-br/2025/01/como-usar-o-mistral-ai-no-onlyoffice
[134] https://neuronup.com/br/novidades-neuronup/inteligencia-artificial-na-reabilitacao-cognitiva-o-futuro-da-neuropsicologia/
[135] https://mistral.ai
[136] https://www.abinee.org.br/wp-content/uploads/2024/09/IBM.pdf
[137] https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/mistral
[138] https://www.semanticscholar.org/paper/385876c5cd12cbb669cc839ea14f7769200a9c33
[139] https://www.reddit.com/r/Twitter_Brasil/comments/17erjuu/e_ainda_tem_que_achar_solu%C3%A7%C3%A3o/
[140] https://www.reddit.com/r/brdev/comments/16r67l8/que_sideprojects_voc%C3%AAs_est%C3%A3o_fazendo_no_momento/
[141] https://www.youtube.com/watch?v=VPuR5C_-uW8
[142] https://repositorio.fgv.br/bitstreams/9d6f598b-ee51-4bed-a878-214c60371a0e/download
[143] https://www.adrenaline.com.br/amd/amd-libera-cpus-ryzen-e-gpus-radeon-executarem-chatbot-local-com-ia/
[144] https://iot-labs.io/ecossistema-lorawan/modulos-chipsets/
[145] https://pt.linkedin.com/pulse/rlhf-reinforcement-learning-from-human-feedback-o-segredo-oeiras-amhqf
[146] https://www.ibm.com/br-pt/topics/mlops
[147] https://www.youtube.com/watch?v=adhF0CHYFQ8
[148] https://www.datacamp.com/pt/tutorial/mistral-7b-tutorial
[149] https://www.cognifit.com/br/habilidade-cognitiva/velocidade-de-processamento/ 