import logging
import requests
from a3x.core.skills import skill
from a3x.core.config import GUMROAD_API_BASE_URL, GUMROAD_API_KEY
from typing import Optional

logger = logging.getLogger(__name__)

@skill(
    name="gumroad_api",
    description=(
        "Permite ao agente interagir com a API da Gumroad para buscar produtos, vendas, tendências, criar ou atualizar ofertas, "
        "e analisar oportunidades de monetização digital de forma autônoma e segura."
    ),
    parameters={
        "endpoint": {"type": str, "description": "O endpoint da API a ser chamado (ex: 'products', 'sales')."},
        "method": {"type": str, "description": "Método HTTP (GET, POST, PUT, DELETE)."},
        "params": {"type": Optional[dict], "description": "Parâmetros de query (GET) ou corpo da requisição (POST/PUT)."},
        "api_key": {"type": Optional[str], "description": "Chave de API da Gumroad (usa config se None)."},
        "base_url": {"type": Optional[str], "description": "URL base da API (usa config se None)."},
    },
)
async def gumroad_api_skill(
    endpoint: str,
    method: str = "GET",
    params: Optional[dict] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
):
    """
    Skill para integração autônoma com a API da Gumroad.
    """
    url = (base_url or GUMROAD_API_BASE_URL).rstrip("/") + "/" + endpoint.lstrip("/")
    key = api_key or GUMROAD_API_KEY
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        logger.info(f"[GumroadAPI] {method} {url} | params: {params}")
        if method.upper() == "GET":
            resp = requests.get(url, headers=headers, params=params)
        elif method.upper() == "POST":
            resp = requests.post(url, headers=headers, json=params)
        elif method.upper() == "PUT":
            resp = requests.put(url, headers=headers, json=params)
        elif method.upper() == "DELETE":
            resp = requests.delete(url, headers=headers, json=params)
        else:
            return {"status": "error", "data": {"message": f"Método HTTP não suportado: {method}"}}
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"[GumroadAPI] Resposta: {data}")
        return {"status": "success", "data": data}
    except Exception as e:
        logger.error(f"[GumroadAPI] Erro ao acessar API: {e}")
        return {"status": "error", "data": {"message": str(e)}}