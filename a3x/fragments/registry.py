import logging
import inspect
import pkgutil
import importlib
import re
from typing import Dict, Optional, Type, List, Callable
from pathlib import Path

# Importações base
from .base import BaseFragment, FragmentDef, ManagerFragment
from a3x.core.llm_interface import LLMInterface # Assumindo que LLMInterface está aqui
from a3x.core.config import PROJECT_ROOT # Para path discovery

logger = logging.getLogger(__name__)

# <<< DECORATOR to mark fragment classes >>>
def fragment(
    name: str,
    description: str,
    category: str = "Execution",
    skills: Optional[List[str]] = None,
    managed_skills: Optional[List[str]] = None
) -> Callable[[Type[BaseFragment]], Type[BaseFragment]]:
    """
    Decorator to mark and provide metadata for Fragment classes.

    This decorator should be applied to any class inheriting from BaseFragment or ManagerFragment.
    It attaches a `_fragment_metadata` dictionary to the class, which is used by the
    FragmentRegistry during the discovery process.

    Args:
        name: The unique name for this fragment, used for identification and invocation.
        description: A brief description of the fragment's purpose, used in prompts.
        category: The category of the fragment, typically "Execution" (for direct task
            completion) or "Management" (for coordinating other skills/fragments).
        skills: A list of skill names that this fragment directly uses (for Execution fragments).
        managed_skills: A list of skill names that this fragment coordinates or manages
            (primarily for Management fragments).

    Returns:
        A decorator function that attaches metadata to the decorated class.
    """
    def decorator(cls: Type[BaseFragment]) -> Type[BaseFragment]:
        if not inspect.isclass(cls) or not issubclass(cls, BaseFragment):
            raise TypeError(f"@{fragment.__name__} can only decorate subclasses of BaseFragment.")

        # Store metadata directly on the class for later discovery
        cls._fragment_metadata = {
            "name": name,
            "description": description,
            "category": category,
            "skills": skills or [],
            "managed_skills": managed_skills or []
        }
        logger.debug(f"Metadata attached to fragment class {cls.__name__}: {cls._fragment_metadata}")
        return cls
    return decorator


class FragmentRegistry:
    """
    Manages the discovery, registration, instantiation, and access of Fragments and Managers.

    This class acts as the central hub for all fragment-related operations. It automatically
    discovers fragments decorated with `@fragment` within the `a3x.fragments` package upon
    initialization and provides methods to access fragment definitions, instances, and
    trigger dynamic reloads.
    """
    def __init__(self, llm_interface: Optional[LLMInterface] = None, skill_registry=None, config: Optional[Dict] = None):
        """
        Initializes the FragmentRegistry.

        Scans the `a3x.fragments` package for classes decorated with `@fragment` and
        registers their definitions automatically.

        Args:
            llm_interface: An instance of LLMInterface, passed to fragments requiring LLM calls.
            skill_registry: A dictionary mapping skill names to their implementation details,
                              passed to fragments for skill execution.
            config: A global configuration dictionary, parts of which might be passed to fragments.
        """
        # Dependencies
        self.llm_interface = llm_interface # Crucial for Managers/Fragments needing LLM
        self.skill_registry = skill_registry or {} # Ensure it's a dict
        self.config = config or {}
        self.logger = logger # Use module logger

        # Internal state (Instance Attributes)
        self._fragment_defs: Dict[str, FragmentDef] = {} # Stores metadata {name: FragmentDef}
        self._fragment_classes: Dict[str, Type[BaseFragment]] = {} # Stores loaded classes {name: ClassObject}
        self._fragments: Dict[str, BaseFragment] = {} # Stores instantiated fragments {name: Instance}

        self.logger.info("FragmentRegistry initialized.")
        self.discover_and_register_fragments() # Discover fragments on init

    def register_fragment_definition(
        self,
        name: str,
        fragment_class: Type[BaseFragment],
        description: str,
        category: str = "Execution",
        skills: Optional[List[str]] = None,
        managed_skills: Optional[List[str]] = None
    ):
        """Registers the metadata (definition) of a discovered or explicitly provided fragment class.

        This method is primarily called internally by the discovery process but can be used
        to register fragments manually if needed.

        Args:
            name: The name under which to register the fragment.
            fragment_class: The actual class object of the fragment.
            description: Description of the fragment.
            category: Category ('Execution' or 'Management').
            skills: List of skills used by the fragment.
            managed_skills: List of skills managed by the fragment.
        """
        if not inspect.isclass(fragment_class) or not issubclass(fragment_class, BaseFragment):
            self.logger.error(f"Attempted to register invalid class for fragment '{name}'. Must be a subclass of BaseFragment.")
            return

        # Use name from metadata if available, otherwise fall back
        registration_name = name # Use name from metadata/decorator
        if not registration_name:
             # Fallback to class name if decorator somehow didn't provide one (shouldn't happen)
             registration_name = fragment_class.__name__
             self.logger.warning(f"Fragment class {fragment_class.__name__} registered using class name as fallback.")

        fragment_def = FragmentDef(
            name=registration_name,
            fragment_class=fragment_class, # Store the class itself in the definition
            description=description,
            category=category,
            skills=skills or [],
            managed_skills=managed_skills or []
        )

        if registration_name in self._fragment_defs:
            self.logger.warning(f"Fragment definition '{registration_name}' is being redefined in the registry.")

        self._fragment_defs[registration_name] = fragment_def
        self._fragment_classes[registration_name] = fragment_class # Cache the class directly
        self.logger.info(f"Registered Fragment/Manager Definition: '{registration_name}' (Category: {category})")

    # --- MODIFIED: Now uses decorator metadata --- 
    def discover_and_register_fragments(self, force_reload: bool = False):
        """
        Discovers fragment classes within the `a3x.fragments` package
        that are decorated with @fragment and registers their definitions.

        This method scans all modules in the `a3x.fragments` directory (excluding
        `registry.py`, `base.py`, and `definitions.py`), looks for classes decorated
        with `@fragment`, and registers them using the metadata provided in the decorator.

        Args:
            force_reload: If True, clears all existing fragment definitions, classes, and
                          instances from the registry and reloads all modules within the
                          `a3x.fragments` package before performing discovery. This allows
                          picking up changes in fragment code or adding new fragment files
                          without restarting the agent, enabling dynamic updates.
                          Note: Module reloading can have side effects in complex scenarios.
        """
        if force_reload:
            self.logger.info("Forcing fragment reload. Clearing existing definitions...")
            self._fragment_defs.clear()
            self._fragment_classes.clear()
            self._fragments.clear() # Clear instances too

        self.logger.info("Starting fragment discovery...")
        fragments_pkg_name = "a3x.fragments"
        try:
            package = importlib.import_module(fragments_pkg_name)
            # Ensure package.__path__ is usable
            if not hasattr(package, '__path__'):
                 self.logger.error(f"Package {fragments_pkg_name} has no __path__, cannot discover modules.")
                 return

            prefix = package.__name__ + "."
            modules_processed = set() # Track processed modules to avoid double processing

            # Use walk_packages for potentially deeper discovery if needed
            for importer, modname, ispkg in pkgutil.walk_packages(package.__path__, prefix):
                # Skip the registry, base, and definitions modules themselves
                if modname.endswith((".registry", ".base", ".definitions")):
                     continue
                # Skip if already processed (walk_packages can yield duplicates sometimes)
                if modname in modules_processed:
                     continue
                modules_processed.add(modname)

                try:
                    # --- Reload Module if force_reload is True --- 
                    if force_reload and modname in importlib.sys.modules:
                         self.logger.debug(f"Attempting to reload module: {modname}")
                         module = importlib.reload(importlib.sys.modules[modname])
                         self.logger.info(f"Reloaded module: {modname}")
                    else:
                         self.logger.debug(f"Attempting to import module: {modname}")
                         module = importlib.import_module(modname)
                         self.logger.debug(f"Successfully imported module: {modname}")
                    # --- End Reload Logic --- 

                    self.logger.debug(f"Inspecting members of module: {modname}")
                    for attribute_name, attribute in inspect.getmembers(module):
                        # Check if it's a class AND has the _fragment_metadata attribute set by our decorator
                        if inspect.isclass(attribute) and hasattr(attribute, '_fragment_metadata'):
                            metadata = attribute._fragment_metadata
                            self.logger.info(f"Discovered @fragment decorated class: {attribute.__name__} in {modname} with metadata: {metadata}")
                            
                            # Register using the metadata from the decorator
                            self.register_fragment_definition(
                                name=metadata["name"],
                                fragment_class=attribute,
                                description=metadata["description"],
                                category=metadata["category"],
                                skills=metadata["skills"],
                                managed_skills=metadata["managed_skills"]
                            )

                except ImportError as e:
                    # Log gracefully if a module within the package fails to import/reload
                    self.logger.warning(f"Could not import/reload module {modname} during fragment discovery: {e}")
                except Exception as e:
                    # Log other errors during module inspection
                     self.logger.error(f"Error inspecting module {modname}: {e}", exc_info=True)
        except ImportError as e:
            # Error importing the main fragments package
            self.logger.error(f"Could not import base fragments package '{fragments_pkg_name}': {e}")

        self.logger.info(f"Fragment discovery completed. {len(self._fragment_defs)} definitions registered.")


    def register_fragment_class(self, name: str, fragment_cls: Type[BaseFragment]):
        """Dynamically registers a new Fragment class (e.g., generated at runtime)."""
        # This method might become less necessary if discovery handles everything,
        # but could be useful for explicitly adding programmatically generated fragments.
        # It should ensure the class has the required metadata or extract it.
        if not inspect.isclass(fragment_cls) or not issubclass(fragment_cls, BaseFragment):
            self.logger.error(f"Attempted to register invalid class for fragment '{name}'. Must be a subclass of BaseFragment.")
            return False
        if name in self._fragment_classes and self._fragment_classes[name] != fragment_cls:
            self.logger.warning(f"Dynamically registering fragment class '{name}', overwriting existing registration.")
        elif name in self._fragment_classes:
            self.logger.debug(f"Fragment class '{name}' already registered.")
            return True # Already exists

        # Extract metadata (assuming it follows the decorator pattern or has attributes)
        metadata = getattr(fragment_cls, '_fragment_metadata', None)
        if metadata:
             self.register_fragment_definition(
                 name=metadata["name"], # Use name from metadata
                 fragment_class=fragment_cls,
                 description=metadata["description"],
                 category=metadata["category"],
                 skills=metadata["skills"],
                 managed_skills=metadata["managed_skills"]
             )
             self.logger.info(f"Dynamically registered fragment class and definition: '{metadata['name']}'")
             return True
        else:
             # Fallback if no decorator metadata (less ideal)
             self.logger.warning(f"Dynamically registering fragment '{name}' without decorator metadata. Attempting fallback extraction.")
             description = fragment_cls.__doc__.strip().split('\n')[0] if fragment_cls.__doc__ else "Auto-generated fragment"
             category = "Management" if issubclass(fragment_cls, ManagerFragment) else "Execution"
             skills = getattr(fragment_cls, 'DEFAULT_SKILLS', [])
             managed_skills = getattr(fragment_cls, 'MANAGED_SKILLS', [])
             self.register_fragment_definition(name, fragment_cls, description, category, skills, managed_skills)
             self.logger.info(f"Dynamically registered fragment (fallback): '{name}'")
             return True


    def load_fragment(self, name: str) -> Optional[BaseFragment]:
        """
        Instantiates a fragment based on its registered definition.

        If the fragment is already instantiated, returns the existing instance.
        Otherwise, it finds the class definition, prepares necessary dependencies
        (like LLMInterface, SkillRegistry, etc., based on the fragment's needs),
        and instantiates it.

        Args:
            name: The name of the fragment to load.

        Returns:
            An instance of the requested fragment, or None if instantiation fails.
        """
        if name in self._fragments:
            self.logger.debug(f"Returning cached instance for fragment '{name}'.")
            return self._fragments[name]

        if name not in self._fragment_classes or name not in self._fragment_defs:
            self.logger.error(f"Fragment '{name}' class or definition not found in registry. Cannot load.")
            return None

        fragment_cls = self._fragment_classes[name]
        fragment_def = self._fragment_defs[name]
        self.logger.info(f"Attempting to instantiate Fragment '{name}' (Class: {fragment_cls.__name__}).")

        # --- Prepare dependencies --- 
        # Gather potential dependencies the fragment might need
        # The fragment's __init__ signature will determine which ones are actually passed
        all_possible_kwargs = {
            "fragment_def": fragment_def,
            "llm_interface": self.llm_interface,
            "skill_registry": self.skill_registry, 
            "tool_registry": self.skill_registry, # Pass skill registry as tool_registry
            "config": self.config, 
            # Add other potential dependencies here
        }

        try:
            # --- Inspect __init__ and Filter kwargs --- 
            init_sig = inspect.signature(fragment_cls.__init__)
            init_params = init_sig.parameters
            
            actual_init_kwargs = {}
            for param_name, param in init_params.items():
                if param_name == 'self': # Skip self
                    continue
                if param_name in all_possible_kwargs:
                    actual_init_kwargs[param_name] = all_possible_kwargs[param_name]
                # If param has a default, it's optional, don't raise error if not provided
                elif param.default == inspect.Parameter.empty:
                    self.logger.warning(f"Required __init__ parameter '{param_name}' for {fragment_cls.__name__} not found in registry dependencies.")
                    # Decide if this should be a hard error or just a warning
                    # raise ValueError(f"Missing dependency for {fragment_cls.__name__}: {param_name}")
            
            self.logger.debug(f"Filtered kwargs for {fragment_cls.__name__}.__init__: {list(actual_init_kwargs.keys())}")

            # --- Instantiate with filtered args --- 
            instance = fragment_cls(**actual_init_kwargs)
            
            self._fragments[name] = instance
            self.logger.info(f"Successfully instantiated and cached Fragment '{name}'.")
            return instance
        except TypeError as e:
            # Catch TypeErrors specifically related to instantiation (wrong args)
            self.logger.error(f"Failed to instantiate Fragment '{name}' ({fragment_cls.__name__}): {e}", exc_info=True)
        except Exception as e:
            # Catch other potential errors during instantiation
            self.logger.error(f"An unexpected error occurred during instantiation of Fragment '{name}' ({fragment_cls.__name__}): {e}", exc_info=True)
        
        return None # Return None if instantiation failed

    def get_fragment(self, name: str) -> Optional[BaseFragment]:
        """
        Returns an instantiated fragment, loading it if necessary.

        This is the primary method for accessing fragment instances.

        Args:
            name: The name of the fragment to retrieve.

        Returns:
            The cached or newly loaded instance of the fragment, or None if not found/loadable.
        """
        fragment = self._fragments.get(name)
        if not fragment:
            self.logger.warning(f"Fragment '{name}' not found in loaded instances. Attempting to load...")
            fragment = self.load_fragment(name) # load_fragment handles logging on failure
            if not fragment:
                 # Keep error minimal as load_fragment already logged details
                 self.logger.error(f"Failed to get or load fragment '{name}'.")
                 return None
        return fragment

    def get_fragment_definition(self, name: str) -> Optional[FragmentDef]:
         """Retrieves the FragmentDef (metadata) for a given fragment name.

         Provides access to the stored definition (class, description, skills, etc.)
         without needing to instantiate the fragment.

         Args:
            name: The name of the fragment definition to retrieve.

         Returns:
             The FragmentDef object containing metadata, or None if not found.
         """
         fragment_def = self._fragment_defs.get(name)
         if not fragment_def:
              self.logger.warning(f"Fragment definition for '{name}' not found.")
         return fragment_def

    def get_all_definitions(self) -> Dict[str, FragmentDef]:
         """Returns a dictionary copy of all registered fragment definitions.

         Useful for inspecting the currently registered fragments and their metadata.

         Returns:
             A dictionary mapping fragment names to their FragmentDef objects.
         """
         return self._fragment_defs.copy()

    # <<< ADDED: Method to get all instantiated fragments >>>
    def get_all_fragments(self) -> Dict[str, BaseFragment]:
        """Returns a dictionary of all currently instantiated fragment instances."""
        # Ensure all defined fragments are loaded/instantiated first?
        # For now, just return the current dictionary.
        # Consider calling self.load_all_fragments() here if needed, but beware of re-instantiation.
        self.logger.debug(f"Returning {len(self._fragments)} instantiated fragments: {list(self._fragments.keys())}")
        return self._fragments.copy()
    # <<< END ADDED >>>

    def get_available_fragments_description(self) -> str:
         """
         Formats the names, categories, descriptions, and managed skills
         of all *registered* fragments for inclusion in the orchestrator prompt.

         Generates a string suitable for inserting into the Orchestrator's system prompt
         to inform it about the available components (fragments/managers) it can delegate to.

         Returns:
             A formatted string describing available fragments, or a message if none are registered.
         """
         lines = ["Available Components (Workers):"]
         definitions = self.get_all_definitions() # Get current definitions
         if not definitions:
              return "No fragments registered."

         for name, fragment_def in definitions.items():
              managed_info = ""
              if fragment_def.category == "Management" and fragment_def.managed_skills:
                  managed_info = f" Manages Skills: {fragment_def.managed_skills}"
              lines.append(f"- {name} ({fragment_def.category}): {fragment_def.description}{managed_info}")

         return "\n".join(lines)

    def load_all_fragments(self):
         """Attempts to instantiate all registered fragment classes."""
         self.logger.info("Attempting to load instances for all registered fragments...")
         count = 0
         for name in list(self._fragment_classes.keys()): # Iterate over a copy of keys
             if name not in self._fragments:
                  if self.load_fragment(name):
                       count += 1
         self.logger.info(f"Finished loading fragments. {count} new instances created.")

    # <<< ADDED Method >>>
    async def select_fragment_for_task(self, objective: str, context: Optional[Dict] = None) -> Optional[BaseFragment]:
        """
        Selects the most appropriate Fragment based on the objective.
        Currently uses simple keyword matching for file operations.
        """
        self.logger.info(f"Attempting to select fragment for objective: '{objective[:100]}...'")

        # Simple keyword matching for file operations
        file_op_keywords = ['list', 'read', 'write', 'delete', 'file', 'directory', 'arquivo', 'diretorio', 'listar', 'ler', 'escrever', 'apagar', 'deletar']
        objective_lower = objective.lower()
        if any(keyword in objective_lower for keyword in file_op_keywords):
            self.logger.info("Objective seems related to file operations. Selecting FileOpsManager.")
            # Return the *instance* of the FileOpsManager
            file_ops_manager = self.get_fragment("FileOpsManager")
            if file_ops_manager:
                return file_ops_manager
            else:
                self.logger.error("FileOpsManager fragment is registered but failed to load/instantiate.")
                return None

        # --- Default Fallback ---
        # TODO: Implement more sophisticated selection (LLM call, semantic search, etc.)
        # For now, if no specific match, return None
        self.logger.warning(f"No specific fragment matched objective via keyword search: '{objective[:100]}...'")
        return None
    # <<< END ADDED Method >>>

# --- Standalone functions are removed --- 