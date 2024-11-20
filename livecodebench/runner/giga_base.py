import logging
import base64
import json
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, model_validator

import yaml
import requests


logger = logging.getLogger(__name__)


class GigaParams(BaseModel):
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def check_zero_temperature(cls, data: Any) -> Any:
        if data["temperature"] == 0.0:
            logger.warning("Updated temperate value. Before: %s", data)
            data["temperature"] = 1.0
            data["top_p"] = 0
            logger.warning("Updated temperate value. After: %s", data)
        return data


class GigaApiConfig(BaseModel):
    base_url: str
    auth_url: str
    route_models: str = "/models"
    route_chat: str = "/chat/completions"

class GigaConfig(BaseModel):
    model: str
    api: GigaApiConfig
    params: GigaParams
    login_password: Optional[str] = None
    need_auth: bool = False

    @classmethod
    def load_from_config(cls, path: str):
        with open(path, encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))

class GigaChat:
    def __init__(self):
        self.config = None
        self.token = None

    def update_config(self, config: GigaConfig):
        self.config = config
        logger.info("GigaChat config: %s", config)
        self.get_token()

    def get_token(self):
        key = base64.b64encode(self.config.login_password.encode()).decode("utf-8")
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid4()),
            "Authorization": f"Basic {key}",
        }
        api = self.config.api
        self.token = json.loads(requests.post(url=api.auth_url, headers=headers).text)["tok"]

    def create_chat_completion(self, prompt, **kwargs):
        logger.info(prompt)
        api = self.config.api
        url = api.base_url + api.route_chat
        headers = {"Content-Type": "application/json", "Accept": "application/json", "Authorization": f"Bearer {self.token}"}
        if not isinstance(prompt, list):
            prompt = [prompt]
        data_raw = {
            "model": self.config.model,
            "messages": prompt,
        }
        for param in ["temperature", "top_p"]:
            if param in kwargs:
                data_raw[param] = float(kwargs[param])

        data = json.dumps(data_raw)

        response_raw = requests.post(url=url, headers=headers, data=data)
        logger.info(response_raw.status_code)
        response = json.loads(response_raw.text)
        # logger.info(response)
        result = [choice["message"]["content"] for choice in response["choices"]]
        logger.info(result)
        return result
