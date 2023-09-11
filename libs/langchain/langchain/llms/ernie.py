"""Wrapper around Ernie APIs."""
from __future__ import annotations

import logging
import json

from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import requests

from langchain.callbacks.manager import (
    CallbackManagerForLLMRun,
)
from langchain.llms.base import LLM
from langchain.pydantic_v1 import BaseModel, Extra, Field, PrivateAttr, root_validator
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class _ErnieEndpointClient(BaseModel):
    """An API client that talks to a Ernie llm endpoint."""

    host: str
    oauth_url: str
    api_key: str
    secret_key: str

    def _get_access_token(self):
        url = "{0}?&grant_type=client_credentials&client_id={1}&client_secret={2}".format(self.oauth_url, self.api_key, self.secret_key)

        payload = json.dumps("")
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        response = requests.request("POST", url, headers=headers, data=payload)
        return response.json().get("access_token")   

    def post(self, request: Any) -> Any:

        url = "{0}?access_token={1}".format(self.host, self._get_access_token())

        headers = {
            'Content-Type': 'application/json'
        }

        logger.debug(json.dumps(request["messages"]))

        response = requests.post(url, headers=headers, json=request)
        # TODO: error handling and automatic retries

        logger.debug(json.dumps(response.json()))

        if not response.ok:
            raise ValueError(f"HTTP {response.status_code} error: {response.text}")

        return response.json()["result"]


class Ernie(LLM):
    """Wrapper around Ernie large language models.
    To use, you should have the environment variable
    ``ERNIE_API_KEY`` and ``ERNIE_GROUP_ID`` set with your API key,
    or pass them as a named parameter to the constructor.
    Example:
     .. code-block:: python
         from langchain.llms.ernie import Ernie
         ernie = Ernie(model="<model_name>", ernie_api_key="my-api-key",
          ernie_group_id="my-group-id")
    """

    _client: _ErnieEndpointClient = PrivateAttr()
    model: str = "ERNIE-Bot-turbo"
    """Model name to use."""
    max_tokens: int = 11200
    """Denotes the number of tokens to predict per generation."""
    temperature: float = 0.2
    """A non-negative float that tunes the degree of randomness in generation."""
    top_p: float = 0.8
    """Total probability mass of tokens to consider at each step."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    ernie_api_host: Optional[str] = None
    ernie_oauth_url: Optional[str] = None
    ernie_secret_key: Optional[str] = None
    ernie_api_key: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["ernie_api_key"] = get_from_dict_or_env(
            values, "ernie_api_key", "ERNIE_API_KEY"
        )
        values["ernie_secret_key"] = get_from_dict_or_env(
            values, "ernie_secret_key", "ERNIE_SECRET_KEY"
        )
        # Get custom api url from environment.
        values["ernie_api_host"] = get_from_dict_or_env(
            values,
            "ernie_api_host",
            "ERNIE_API_HOST",
            default="https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant",
        )

        # Get custom api url from environment.
        values["ernie_oauth_url"] = get_from_dict_or_env(
            values,
            "ernie_oauth_url",
            "ERNIE_OAUTH_URL",
            default="https://aip.baidubce.com/oauth/2.0/token",
        )
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        return {
            "model": self.model,
            "tokens_to_generate": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            **self.model_kwargs,
        }

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{"model": self.model}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "ernie"

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._client = _ErnieEndpointClient(
            host=self.ernie_api_host,
            oauth_url=self.ernie_oauth_url,
            api_key=self.ernie_api_key,
            secret_key=self.ernie_secret_key,
        )

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        r"""Call out to Ernie's completion endpoint to chat
        Args:
            prompt: The prompt to pass into the model.
        Returns:
            The string generated by the model.
        Example:
            .. code-block:: python
                response = ernie("Tell me a joke.")
        """
        request = self._default_params
        request["messages"] = [{"role": "user", "content": prompt}]
        request.update(kwargs)
        response = self._client.post(request)

        return response
