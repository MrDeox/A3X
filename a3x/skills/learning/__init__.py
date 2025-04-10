# This file makes the 'learning' directory a Python package 

# Explicitly import skill modules to ensure registration
from . import learn_from_reflection_logs
from . import refine_decision_prompt
from . import synthesize_learning_insights # Re-enabled import
from . import apply_prompt_refinement_from_logs # Added import
from . import auto_improve_simulation_prompt # Added import

# Opcional: Adicionar um log ou print para confirmar que o __init__ foi executado (bom para debug)
print("DEBUG: Imported learning skills package.")

# NÃO é necessário chamar register_skill aqui. O decorador @skill faz isso. 