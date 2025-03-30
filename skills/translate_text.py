"""
Skill to translate text between languages using Argos Translate.
"""

import os
import logging
import argostranslate.package
import argostranslate.translate
import nltk # Needed for sentence tokenization

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK data if not present
try:
    logger.debug("Checking for NLTK 'punkt' resource...")
    nltk.data.find('tokenizers/punkt')
    logger.debug("NLTK 'punkt' resource found.")
except LookupError:
    logger.info("NLTK 'punkt' tokenizer not found. Attempting download...")
    try:
        nltk.download('punkt') # Attempt download inside the except block
        logger.info("NLTK 'punkt' tokenizer downloaded successfully.")
    except Exception as download_exc:
        # Catch potential errors during download (network issues, permissions)
        logger.error(f"Failed to download NLTK 'punkt' resource: {download_exc}", exc_info=True)
        # Depending on criticality, might want to raise an error or return a failure state

# --- Constants for Language Models ---
# Define paths relative to the project root or use absolute paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "argos-translate")

# Store loaded models to avoid reloading
LOADED_MODELS = {}
INSTALLED_PACKAGES = False # Flag to track if packages are installed

def ensure_models_available(source_lang: str, target_lang: str) -> bool:
    """Checks if required Argos Translate models are installed and installs them if not."""
    global INSTALLED_PACKAGES
    if INSTALLED_PACKAGES:
        return True # Assume they are still installed within the session

    try:
        # Update package index
        logger.info(f"Updating Argos Translate package index from: {DEFAULT_MODEL_DIR}")
        argostranslate.package.update_package_index()

        # Check installed packages
        available_packages = argostranslate.package.get_installed_packages()
        available_package_paths = {pkg.package_path for pkg in available_packages}
        logger.info(f"Available locally installed Argos Translate packages: {available_package_paths}")

        # Determine required package direction
        required_from_code = source_lang
        required_to_code = target_lang

        # Find the required translation package
        installed_translation = None
        for pkg in available_packages:
            if pkg.from_code == required_from_code and pkg.to_code == required_to_code:
                installed_translation = pkg
                break

        if installed_translation:
            logger.info(f"Required translation model {source_lang}->{target_lang} is installed: {installed_translation.package_path}")
            # Load the model into cache if not already loaded
            if (source_lang, target_lang) not in LOADED_MODELS:
                logger.info(f"Loading model {source_lang}->{target_lang} into memory...")
                LOADED_MODELS[(source_lang, target_lang)] = installed_translation.get_translation()
                logger.info(f"Model {source_lang}->{target_lang} loaded.")
            INSTALLED_PACKAGES = True # Mark as installed for this session
            return True
        else:
            logger.warning(f"Required translation model {source_lang}->{target_lang} not found locally.")
            # Attempt to find and install the package
            available_remote_packages = argostranslate.package.get_available_packages()
            package_to_install = None
            for pkg in available_remote_packages:
                if pkg.from_code == required_from_code and pkg.to_code == required_to_code:
                    package_to_install = pkg
                    break

            if package_to_install:
                logger.info(f"Found available package: {package_to_install}. Attempting to download and install...")
                # Ensure the target directory exists
                install_dir = os.path.join(argostranslate.package.get_package_path(), package_to_install.filename)
                if not os.path.exists(os.path.dirname(install_dir)):
                     os.makedirs(os.path.dirname(install_dir), exist_ok=True)
                     logger.info(f"Created directory for Argos Translate packages: {os.path.dirname(install_dir)}")

                # Correctly initiate download and install
                download_path = package_to_install.download()
                if download_path:
                     logger.info(f"Package downloaded to: {download_path}")
                     argostranslate.package.install_from_path(download_path)
                     logger.info(f"Package {package_to_install.filename} installed successfully.")
                     # Verify installation and load
                     available_packages = argostranslate.package.get_installed_packages() # Refresh list
                     installed_translation = None
                     for pkg in available_packages:
                         if pkg.from_code == required_from_code and pkg.to_code == required_to_code:
                             installed_translation = pkg
                             break
                     if installed_translation:
                        logger.info(f"Loading model {source_lang}->{target_lang} into memory after install...")
                        LOADED_MODELS[(source_lang, target_lang)] = installed_translation.get_translation()
                        logger.info(f"Model {source_lang}->{target_lang} loaded.")
                        INSTALLED_PACKAGES = True
                        return True
                     else:
                         logger.error(f"Failed to find translation {source_lang}->{target_lang} even after installation attempt.")
                         return False
                else:
                     logger.error(f"Failed to download package: {package_to_install.filename}")
                     return False
            else:
                logger.error(f"Could not find an available package for {source_lang}->{target_lang}.")
                return False

    except argostranslate.package.PackageException as e:
        logger.error(f"Argos Translate Package Error: {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error during Argos Translate model check/install: {e}", exc_info=True)
        return False

def skill_translate_text(text: str, source_lang: str, target_lang: str) -> dict:
    """Translates text from source_lang to target_lang using Argos Translate.

    Args:
        text (str): The text to translate.
        source_lang (str): The ISO 639-1 code of the source language (e.g., 'en').
        target_lang (str): The ISO 639-1 code of the target language (e.g., 'pt').

    Returns:
        dict: Result dictionary with status and translated text or error message.
    """
    logger.info(f"Attempting to translate text from '{source_lang}' to '{target_lang}'.")

    if not text or not source_lang or not target_lang:
        logger.error("Missing required arguments: text, source_lang, or target_lang.")
        return {"status": "error", "data": {"message": "Missing required arguments."}}

    # Ensure models are available (installs if necessary)
    if not ensure_models_available(source_lang, target_lang):
        logger.error(f"Translation models for {source_lang}->{target_lang} are not available or couldn't be installed.")
        return {"status": "error", "data": {"message": f"Models for {source_lang}->{target_lang} unavailable."}}

    try:
        # Get the loaded translation object
        translation = LOADED_MODELS.get((source_lang, target_lang))
        if not translation:
             # This should ideally not happen if ensure_models_available succeeded
             logger.error(f"Model {source_lang}->{target_lang} not found in loaded cache even after check.")
             return {"status": "error", "data": {"message": "Internal error: Model cache issue."}}

        # Perform translation
        # Handle potentially long text by splitting into sentences
        sentences = nltk.sent_tokenize(text)
        translated_sentences = []
        logger.info(f"Translating {len(sentences)} sentences...")
        for i, sentence in enumerate(sentences):
            translated_sentence = translation.translate(sentence)
            translated_sentences.append(translated_sentence)
            if (i + 1) % 10 == 0:
                 logger.debug(f"Translated {i+1}/{len(sentences)} sentences...")

        translated_text = " ".join(translated_sentences)
        logger.info(f"Translation successful. Input length: {len(text)}, Output length: {len(translated_text)}")
        logger.debug(f"Original: {text}")
        logger.debug(f"Translated: {translated_text}")
        return {"status": "success", "data": {"translated_text": translated_text}}

    except argostranslate.translate.TranslateError as e:
        logger.error(f"Argos Translate Error: {e}", exc_info=True)
        return {"status": "error", "data": {"message": f"Translation engine error: {e}"}}
    except Exception as e:
        logger.error(f"Unexpected error during translation: {e}", exc_info=True)
        return {"status": "error", "data": {"message": f"An unexpected error occurred: {e}"}}

# Example Usage (for testing):
# if __name__ == '__main__':
#     # Ensure logging is configured if running standalone
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#     # Make sure models are downloaded (first run might take time)
#     # You might need to run `argospm update` and `argospm install translate-en_pt` in your venv
#     print("Ensuring English to Portuguese model...")
#     ensure_models_available('en', 'pt')
#     print("Ensuring Portuguese to English model...")
#     ensure_models_available('pt', 'en')

#     text_to_translate_en = "Hello world! This is a test of the translation system. It should handle multiple sentences."
#     print(f"\nTranslating from English to Portuguese: '{text_to_translate_en}'")
#     result_en_pt = skill_translate_text(text_to_translate_en, 'en', 'pt')
#     print(f"Result: {result_en_pt}")

#     text_to_translate_pt = "Olá Mundo! Este é um teste do sistema de tradução. Ele deve lidar com múltiplas frases."
#     print(f"\nTranslating from Portuguese to English: '{text_to_translate_pt}'")
#     result_pt_en = skill_translate_text(text_to_translate_pt, 'pt', 'en')
#     print(f"Result: {result_pt_en}")

#     print("\nTesting unsupported language pair (expecting error):")
#     result_err = skill_translate_text("Test", 'xx', 'yy')
#     print(f"Result: {result_err}")