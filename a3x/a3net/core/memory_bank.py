import torch
from typing import Dict, Optional, List, Type, Union
import os
import json
import logging
import importlib
import zipfile
import tempfile
from pathlib import Path

# Assuming FragmentCell is defined nearby
from .fragment_cell import FragmentCell 

# <<< Add import for get_embedding_model >>>
from ..trainer.dataset_builder import get_embedding_model

logger = logging.getLogger(__name__)

class MemoryBank:
    """Stores FragmentCell instances in memory and persists/exports their state."""
    
    DEFAULT_SAVE_DIR = "a3net_memory"
    DEFAULT_EXPORT_DIR = "a3x_repo"

    def __init__(self, 
                 save_dir: str = DEFAULT_SAVE_DIR, 
                 export_dir: str = DEFAULT_EXPORT_DIR):
        """Initializes the memory bank.

        Args:
            save_dir: The directory to save/load internal fragment state files (.pt).
            export_dir: The default directory to save exported fragment packages (.a3xfrag).
        """
        self.fragments: Dict[str, torch.nn.Module] = {}
        self.save_dir = Path(save_dir)
        self.export_dir = Path(export_dir)

        # Create directories if they don't exist
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[MemoryBank] Using save directory: {self.save_dir.resolve()}")
        print(f"[MemoryBank] Using default export directory: {self.export_dir.resolve()}")
        print("[MemoryBank] Initialized.")

    def exists(self, fragment_id: str) -> bool:
        """Checks if a fragment exists in the cache or on disk."""
        # 1. Check memory cache
        if fragment_id in self.fragments:
            return True
        
        # 2. Check disk (.pt file)
        state_file = self._get_save_path(fragment_id, "pt")
        if state_file.exists():
            return True
            
        return False

    def _get_save_path(self, fragment_id: str, ext: str) -> Path:
        """Helper to get the full path for a fragment file."""
        return self.save_dir / f"{fragment_id}.{ext}"

    def save(self, fragment_id: str, fragment: torch.nn.Module):
        """Saves or updates a fragment instance in memory and persists its state to disk.
        
        Args:
            fragment_id: A unique identifier for the fragment.
            fragment: The nn.Module instance to save (e.g., FragmentCell, NeuralLanguageFragment).
        """
        if not isinstance(fragment, torch.nn.Module):
            raise TypeError("Only torch.nn.Module instances can be saved in the MemoryBank.")
            
        self.fragments[fragment_id] = fragment
        print(f"[MemoryBank] Fragment '{fragment_id}' cached in memory.")

        state_file = self._get_save_path(fragment_id, "pt")
        save_data = {
            'class_name': fragment.__class__.__name__,
            'module': fragment.__class__.__module__,
            'state_dict': fragment.state_dict(),
            'init_args': { 
                attr: getattr(fragment, attr) 
                for attr in ['input_dim', 'hidden_dim', 'num_classes', 'fragment_id', 'description', 'id_to_label'] 
                if hasattr(fragment, attr)
            }
        }
        if hasattr(fragment, 'linear2') and not hasattr(fragment, 'num_classes'):
            if 'output_dim' not in save_data['init_args']:
                save_data['init_args']['output_dim'] = fragment.linear2.out_features
            save_data['init_args'].pop('num_classes', None)
            save_data['init_args'].pop('id_to_label', None)

        try:
            torch.save(save_data, state_file)
            print(f"[MemoryBank] Fragment '{fragment_id}' state saved to {state_file}")
        except Exception as e:
            logger.error(f"Error saving fragment '{fragment_id}' state to {state_file}: {e}", exc_info=True)
            print(f"[MemoryBank] Error saving fragment '{fragment_id}' state to disk: {e}")

    def load(self, fragment_id: str) -> Optional[torch.nn.Module]:
        """Loads a fragment instance from memory cache or disk.
        
        Args:
            fragment_id: The identifier of the fragment to load.
            
        Returns:
            The nn.Module instance if found, otherwise None.
        """
        # 1. Check memory cache first
        if fragment_id in self.fragments:
            print(f"[MemoryBank] Fragment '{fragment_id}' found in memory cache.")
            return self.fragments[fragment_id]

        # 2. If not in cache, try loading from disk
        state_file = self._get_save_path(fragment_id, "pt")
        print(f"[MemoryBank] Fragment '{fragment_id}' not in cache. Attempting to load from {state_file}...")

        if not state_file.exists():
            print(f"[MemoryBank] State file not found: {state_file}")
            return None

        try:
            # Load the saved data dictionary
            load_data = torch.load(state_file)
            class_name = load_data['class_name']
            module_path = load_data['module']
            state_dict = load_data['state_dict']
            init_args = load_data['init_args'] # Dictionary of __init__ args

            # Dynamically import the module and get the class
            try:
                module = importlib.import_module(module_path)
                FragmentClass = getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                logger.error(f"Could not import class {class_name} from {module_path}: {e}", exc_info=True)
                print(f"[MemoryBank] Error importing class {class_name} from {module_path}: {e}")
                return None

            # Re-instantiate the fragment
            try:
                # Filter init_args based on the expected signature of the target class
                constructor_args = {}
                original_saved_input_dim = init_args.get('input_dim') # Store original for comparison

                # --- Correction: Verify and potentially override input_dim before instantiation --- 
                if class_name in ["NeuralLanguageFragment", "ReflectiveLanguageFragment"]:
                    try:
                        embedding_model = get_embedding_model() # Assumes this is accessible
                        if embedding_model:
                            correct_input_dim = embedding_model.get_sentence_embedding_dimension()
                            saved_input_dim = init_args.get('input_dim')
                            if saved_input_dim is not None and saved_input_dim != correct_input_dim:
                                logger.warning(f"[MemoryBank Load] Correcting input_dim for '{fragment_id}'. Saved: {saved_input_dim}, Current Embedding Dim: {correct_input_dim}. USING {correct_input_dim}.")
                                init_args['input_dim'] = correct_input_dim # Override before instantiation
                            elif saved_input_dim is None:
                                logger.warning(f"[MemoryBank Load] 'input_dim' missing in saved args for '{fragment_id}'. Setting from embedding model: {correct_input_dim}.")
                                init_args['input_dim'] = correct_input_dim
                        else:
                            logger.error(f"[MemoryBank Load] Cannot verify input_dim for '{fragment_id}'. Embedding model failed to load. Using saved value: {original_saved_input_dim}")
                    except Exception as emb_err:
                        logger.error(f"[MemoryBank Load] Error getting embedding dimension for '{fragment_id}': {emb_err}. Using saved value: {original_saved_input_dim}", exc_info=True)
                # --- End Correction --- 

                if class_name == "FragmentCell":
                    required_keys = ['input_dim', 'hidden_dim', 'output_dim']
                    if not all(k in init_args for k in required_keys):
                         raise ValueError(f"Missing dimensions {required_keys} in saved init_args for FragmentCell {fragment_id}")
                    constructor_args = {k: init_args[k] for k in required_keys}
                elif class_name in ["NeuralLanguageFragment", "ReflectiveLanguageFragment"]:
                     required_keys = ['fragment_id', 'description', 'input_dim', 'hidden_dim', 'num_classes']
                     optional_keys = ['id_to_label']
                     if not all(k in init_args for k in required_keys):
                          raise ValueError(f"Missing arguments {required_keys} in saved init_args for {class_name} {fragment_id}")
                     constructor_args = {k: init_args[k] for k in required_keys}
                     # Add optional args if they exist in the saved data
                     for k in optional_keys:
                         if k in init_args:
                             constructor_args[k] = init_args[k]
                else:
                    # Attempt generic instantiation if type is unknown, might fail
                    logger.warning(f"Attempting generic instantiation for unknown fragment type: {class_name}")
                    # Pass only args likely relevant based on what we save, needs refinement
                    possible_keys = ['fragment_id', 'description', 'input_dim', 'hidden_dim', 'num_classes', 'output_dim', 'id_to_label']
                    constructor_args = {k: v for k, v in init_args.items() if k in possible_keys}

                fragment = FragmentClass(**constructor_args)
                print(f"[MemoryBank] Re-instantiated {class_name} with args: {constructor_args}")
            except Exception as e:
                logger.error(f"Error re-instantiating {class_name} for {fragment_id} with filtered args {constructor_args}: {e}", exc_info=True)
                print(f"[MemoryBank] Error re-instantiating {class_name} for {fragment_id}: {e}")
                return None
                
            # Load the saved state dictionary
            fragment.load_state_dict(state_dict)
            print(f"[MemoryBank] Loaded state dict into re-instantiated {class_name}.")

            # Add to memory cache
            self.fragments[fragment_id] = fragment
            print(f"[MemoryBank] Fragment '{fragment_id}' loaded from disk and cached.")
            return fragment

        except Exception as e:
            logger.error(f"Error loading fragment '{fragment_id}' from {state_file}: {e}", exc_info=True)
            print(f"[MemoryBank] Error loading fragment '{fragment_id}' from disk: {e}")
            # Attempt to delete potentially corrupted file?
            # try:
            #     os.remove(state_file)
            # except OSError:
            #     pass
            return None

    def list(self) -> List[str]:
        """Returns a list of fragment IDs currently cached in memory."""
        # Note: This doesn't list fragments only present on disk but not loaded.
        fragment_ids = list(self.fragments.keys())
        print(f"[MemoryBank] Listing {len(fragment_ids)} fragments currently in memory cache.")
        return fragment_ids

    def __len__(self) -> int:
        """Returns the number of fragments currently cached in memory."""
        return len(self.fragments)

    def export(self, fragment_id: str, export_filename: Optional[Union[str, Path]] = None) -> bool:
        """Exports a given fragment into a .a3xfrag package (zip file).

        The package contains the model state dictionary (fragment.pt) and 
        metadata (metadata.json).

        Args:
            fragment_id: The ID of the fragment to export.
            export_filename: The full desired path for the output .a3xfrag file.
                             If None, defaults to '[fragment_id].a3xfrag' inside
                             the MemoryBank's export_dir.

        Returns:
            True if the export was successful, False otherwise.
        """
        print(f"[MemoryBank] Attempting to export fragment '{fragment_id}'...")
        
        # 1. Load the fragment (from cache or disk)
        fragment = self.load(fragment_id)
        if fragment is None:
            print(f"[MemoryBank] Export failed: Fragment '{fragment_id}' could not be loaded.")
            return False

        # 2. Determine final export path
        if export_filename is None:
            export_path = self.export_dir / f"{fragment_id}.a3xfrag"
        else:
            export_path = Path(export_filename)
            # Ensure parent directory exists if a full path is given
            export_path.parent.mkdir(parents=True, exist_ok=True)
            # Ensure correct extension
            if export_path.suffix != ".a3xfrag":
                 export_path = export_path.with_suffix(".a3xfrag")

        print(f"[MemoryBank] Exporting to: {export_path.resolve()}")

        try:
            # 3. Create a temporary directory
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                
                # 4. Save state_dict to fragment.pt in temp dir
                state_dict_path = tmpdir_path / "fragment.pt"
                torch.save(fragment.state_dict(), state_dict_path)
                print(f"[MemoryBank] Saved state_dict to temporary file: {state_dict_path}")

                # 5. Gather and save metadata to metadata.json in temp dir
                metadata = {
                    "fragment_id": fragment_id,
                    "class_name": fragment.__class__.__name__,
                    "module": fragment.__class__.__module__,
                }
                # Add optional attributes if they exist
                for attr in ['input_dim', 'hidden_dim', 'num_classes', 'output_dim', 'description', 'id_to_label']:
                    if hasattr(fragment, attr):
                        value = getattr(fragment, attr)
                        # Special handling for non-JSON serializable types if needed (e.g., Path objects)
                        # For now, assuming basic types or dicts like id_to_label
                        metadata[attr] = value
                # Handle output_dim specifically if num_classes isn't present (like FragmentCell)
                if 'num_classes' not in metadata and 'output_dim' not in metadata and hasattr(fragment, 'linear2'):
                     metadata['output_dim'] = fragment.linear2.out_features

                metadata_path = tmpdir_path / "metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
                print(f"[MemoryBank] Saved metadata to temporary file: {metadata_path}")
                print(f"  Metadata contents: {metadata}")

                # 6. Create the zip archive (.a3xfrag)
                with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    zipf.write(state_dict_path, arcname="fragment.pt")
                    zipf.write(metadata_path, arcname="metadata.json")
                
                print(f"[MemoryBank] Successfully created export package: {export_path.resolve()}")
            
            # Temporary directory is automatically cleaned up here
            return True

        except Exception as e:
            logger.error(f"Error exporting fragment '{fragment_id}' to {export_path}: {e}", exc_info=True)
            print(f"[MemoryBank] Export failed for fragment '{fragment_id}': {e}")
            # Clean up potentially incomplete export file
            if export_path.exists():
                try:
                    os.remove(export_path)
                except OSError:
                     logger.warning(f"Could not remove incomplete export file: {export_path}")
            return False 

    def import_a3xfrag(self, a3xfrag_path: Union[str, Path]) -> bool:
        """Imports a fragment from an .a3xfrag package file.

        Loads the metadata and state dictionary, reconstructs the fragment, 
        caches it in memory, and saves its internal state file.

        Args:
            a3xfrag_path: The path to the .a3xfrag file.

        Returns:
            True if the import was successful, False otherwise.
        """
        a3xfrag_file = Path(a3xfrag_path)
        print(f"[MemoryBank] Attempting to import fragment from: {a3xfrag_file.resolve()}")

        # 1. Validate path and extension
        if not a3xfrag_file.exists() or not a3xfrag_file.is_file():
            print(f"[MemoryBank] Import failed: File not found or is not a file: {a3xfrag_file}")
            logger.error(f"Import failed: File not found or not a file: {a3xfrag_file}")
            return False
        
        if a3xfrag_file.suffix != ".a3xfrag":
            print(f"[MemoryBank] Import failed: File does not have .a3xfrag extension: {a3xfrag_file}")
            logger.error(f"Import failed: Invalid extension for {a3xfrag_file}")
            return False

        try:
            # 2. Extract zip to temporary directory
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                print(f"[MemoryBank] Extracting {a3xfrag_file.name} to temporary directory...")
                with zipfile.ZipFile(a3xfrag_file, 'r') as zipf:
                    zipf.extractall(tmpdir_path)
                
                # 3. Verify expected files
                state_dict_path = tmpdir_path / "fragment.pt"
                metadata_path = tmpdir_path / "metadata.json"

                if not state_dict_path.exists() or not metadata_path.exists():
                    print("[MemoryBank] Import failed: Zip archive missing fragment.pt or metadata.json.")
                    logger.error(f"Import failed: Missing required files in {a3xfrag_file.name}")
                    return False
                
                print("[MemoryBank] Found fragment.pt and metadata.json.")

                # 4. Load metadata
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                print(f"[MemoryBank] Loaded metadata: {metadata}")

                # Extract required info from metadata
                fragment_id = metadata.get("fragment_id")
                class_name = metadata.get("class_name")
                module_path = metadata.get("module")
                init_args = metadata # Use the whole metadata dict for init args initially
                
                if not all([fragment_id, class_name, module_path]):
                     print("[MemoryBank] Import failed: Metadata missing fragment_id, class_name, or module.")
                     logger.error(f"Import failed: Incomplete metadata in {a3xfrag_file.name}")
                     return False

                # 5. Reconstruct Fragment (similar to load method logic)
                try:
                    module = importlib.import_module(module_path)
                    FragmentClass = getattr(module, class_name)
                except (ImportError, AttributeError) as e:
                    logger.error(f"Could not import class {class_name} from {module_path}: {e}", exc_info=True)
                    print(f"[MemoryBank] Import failed: Error importing class {class_name} from {module_path}: {e}")
                    return False
                
                try:
                    # Filter init_args from metadata based on expected class signature
                    constructor_args = {}
                    if class_name == "FragmentCell":
                        required_keys = ['input_dim', 'hidden_dim', 'output_dim']
                        if not all(k in init_args for k in required_keys):
                             raise ValueError(f"Metadata missing dimensions {required_keys}")
                        constructor_args = {k: init_args[k] for k in required_keys}
                    elif class_name in ["NeuralLanguageFragment", "ReflectiveLanguageFragment"]:
                        required_keys = ['fragment_id', 'description', 'input_dim', 'hidden_dim', 'num_classes']
                        optional_keys = ['id_to_label']
                        if not all(k in init_args for k in required_keys):
                             raise ValueError(f"Metadata missing args {required_keys}")
                        constructor_args = {k: init_args[k] for k in required_keys}
                        for k in optional_keys:
                            if k in init_args:
                                constructor_args[k] = init_args[k]
                    else:
                        logger.warning(f"Attempting generic instantiation from metadata for unknown fragment type: {class_name}")
                        possible_keys = ['fragment_id', 'description', 'input_dim', 'hidden_dim', 'num_classes', 'output_dim', 'id_to_label']
                        constructor_args = {k: v for k, v in init_args.items() if k in possible_keys}

                    fragment = FragmentClass(**constructor_args)
                    print(f"[MemoryBank] Re-instantiated {class_name} '{fragment_id}' with args: {constructor_args}")
                except Exception as e:
                    logger.error(f"Error re-instantiating {class_name} '{fragment_id}' from metadata with args {constructor_args}: {e}", exc_info=True)
                    print(f"[MemoryBank] Import failed: Error re-instantiating {class_name} '{fragment_id}': {e}")
                    return False
                
                # Load state dict
                state_dict = torch.load(state_dict_path)
                fragment.load_state_dict(state_dict)
                print(f"[MemoryBank] Loaded state dict into {class_name} '{fragment_id}'.")

                # 6. Cache in memory
                self.fragments[fragment_id] = fragment
                print(f"[MemoryBank] Fragment '{fragment_id}' imported and cached in memory.")

                # 7. Save internal .pt state file for future direct loads (Optional but recommended)
                internal_save_path = self._get_save_path(fragment_id, "pt")
                try:
                    # We need to reconstruct the save_data structure used by save/load
                    save_data_for_internal = {
                        'class_name': class_name,
                        'module': module_path,
                        'state_dict': state_dict, # Use the already loaded state_dict
                        'init_args': init_args # Save the original metadata init_args
                    }
                    torch.save(save_data_for_internal, internal_save_path)
                    print(f"[MemoryBank] Saved internal state file: {internal_save_path}")
                except Exception as e:
                     logger.warning(f"Could not save internal state file for imported fragment {fragment_id}: {e}", exc_info=True)
                     print(f"[MemoryBank] Warning: Could not save internal state file for imported fragment {fragment_id}: {e}")
                     # Proceed even if internal save fails, fragment is in memory.
                
                # Temporary directory is automatically cleaned up here
                return True
        
        except zipfile.BadZipFile:
             print(f"[MemoryBank] Import failed: File is not a valid zip archive: {a3xfrag_file}")
             logger.error(f"Import failed: BadZipFile for {a3xfrag_file}")
             return False
        except Exception as e:
            logger.error(f"Error importing fragment from {a3xfrag_file}: {e}", exc_info=True)
            print(f"[MemoryBank] Import failed: An unexpected error occurred: {e}")
            return False 