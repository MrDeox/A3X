import sqlite3
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the path relative to this script's location or use an absolute path
# Assuming the lib dir is ../a3x/lib relative to this script if run from project root
LIB_DIR = os.path.join(os.path.dirname(__file__), "a3x", "lib")
VECTOR_EXTENSION_PATH = os.path.join(LIB_DIR, "vector0.so")

logger.info(f"Attempting to test SQLite extension loading.")
logger.info(f"Python sqlite3 module version: {sqlite3.sqlite_version}")
logger.info(f"Extension path: {VECTOR_EXTENSION_PATH}")

conn = None
try:
    conn = sqlite3.connect(':memory:') # Use in-memory DB for test
    logger.info("Connected to in-memory database.")

    # Check if the method exists before calling PRAGMA or load
    if not hasattr(conn, 'enable_load_extension'):
         logger.warning("Connection object does NOT initially have 'enable_load_extension' method.")
         # If this ^ is true, it confirms the build issue theory

    if not hasattr(conn, 'load_extension'):
         logger.warning("Connection object does NOT initially have 'load_extension' method.")
         # If this ^ is true, it confirms the build issue theory

    # Add trusted_schema pragma
    conn.execute("PRAGMA trusted_schema = OFF;")
    logger.info("Executed PRAGMA trusted_schema=OFF;")

    # Try enabling extensions via PRAGMA
    conn.execute("PRAGMA enable_load_extension = 1;")
    logger.info("Executed PRAGMA enable_load_extension=1;")

    # Verify if the method exists *after* PRAGMA
    if hasattr(conn, 'load_extension'):
        logger.info("Connection object HAS 'load_extension' method after PRAGMA.")
        # Now try calling it
        if os.path.exists(VECTOR_EXTENSION_PATH):
            conn.load_extension(VECTOR_EXTENSION_PATH)
            logger.info("Successfully called conn.load_extension().")

            # Further test: try creating the virtual table
            try:
                cursor = conn.cursor()
                cursor.execute("CREATE VIRTUAL TABLE vec_test USING vec0(v float[4]);")
                logger.info("Successfully created vec0 virtual table.")
                cursor.execute("DROP TABLE vec_test;")
                logger.info("Successfully dropped vec0 virtual table.")
            except sqlite3.OperationalError as op_err:
                 logger.error(f"Failed to use vec0 module: {op_err}")

        else:
            logger.warning(f"Extension file not found at {VECTOR_EXTENSION_PATH}, skipping load test.")
    else:
         logger.error("Connection object STILL does not have 'load_extension' method after PRAGMA.")

except sqlite3.Error as e:
    logger.error(f"SQLite error during test: {e}", exc_info=True)
except Exception as e:
    logger.error(f"An unexpected error occurred: {e}", exc_info=True)
finally:
    if conn:
        conn.close()
        logger.info("Database connection closed.") 