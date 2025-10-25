from pydantic_settings import BaseSettings


class Secret(BaseSettings):
    huggingface_token: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


secret = Secret()