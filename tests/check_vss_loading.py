import sqlite3
import os

db_path = os.path.join(os.getcwd(), 'memory.db')
vector_lib_path = os.path.abspath('lib/vector0.so')
vss_lib_path = os.path.abspath('lib/vss0.so')

try:
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    
    # Primeiro carrega a extensão vector0
    conn.load_extension(vector_lib_path)
    print("Extensão vector0 carregada com sucesso!")
    
    # Depois carrega a extensão vss0
    conn.load_extension(vss_lib_path)
    cursor = conn.cursor()
    cursor.execute("SELECT vss_version();")
    version = cursor.fetchone()[0]
    print(f"SUCESSO! Extensão sqlite-vss carregada. Versão: {version}")
    conn.enable_load_extension(False)
except Exception as e:
    print(f"ERRO ao carregar as extensões: {e}")
    print(f"Verifique se os arquivos existem em:")
    print(f"- vector0.so: {vector_lib_path}")
    print(f"- vss0.so: {vss_lib_path}")
    print("Verifique também as permissões e dependências das extensões.")
finally:
    if 'conn' in locals() and conn:
        conn.close() 