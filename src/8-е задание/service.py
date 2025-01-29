# python
import contextlib
import json
import logging
import threading
import time

# 3rdparty
import uvicorn
from pydantic import TypeAdapter

# project
from tools.logging_tools import configure_service_logger
from schemas.service_config import ServiceConfig


class Server(uvicorn.Server):
    """Обертка над uvicorn.Server, не блокирующая основной поток"""

    @contextlib.contextmanager
    def run_in_thread(self):
        """Метод для запуска сервиса в потоке"""
        thread = threading.Thread(target=self.run)
        thread.start()
        try:
            while not self.started and thread.is_alive():
                time.sleep(1e-3)
            yield
        finally:
            self.should_exit = True
            thread.join()


def get_service(
    service_config: ServiceConfig,
    num_workers: int = 0,
    reload: bool = True,
) -> Server:
    """Функция для инициализации FastAPI-сервиса в рамках uvicorn

    Параметры:
        * `service_config` (`ServiceConfig`): путь к конфигурации сервиса
        * `num_workers` (`int`, optional): число обработчиков
        * `reload` (`bool`, `optional`): перезагружать ил сервиса

    Возвращает:
        * `Server`: объект Server
    """
    config = uvicorn.Config(
        "app:app",
        host=service_config.common_params.host,
        port=service_config.common_params.port,
        log_level=logging.INFO,
        log_config=service_config.logging_params.logging_config,
        workers=num_workers,
        reload=reload,
        use_colors=True,
    )
    configure_service_logger(service_config, logging.INFO, "log_file")
    return Server(config)


def main() -> None:
    """Точка инициализации сервиса"""

    service_config = r".\configs\service_config.json"

    with open(service_config, "r") as json_service_config:
        service_config_dict = json.load(json_service_config)

    service_config_adapter = TypeAdapter(ServiceConfig)
    service_config_python = service_config_adapter.validate_python(
        service_config_dict)

    get_service(service_config_python).run()


if __name__ == "__main__":
    main()
