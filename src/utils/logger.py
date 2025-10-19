"""Sistema de logging centralizado para a aplicação."""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class Logger:
    """Classe para gerenciar logging da aplicação."""

    _loggers = {}

    @staticmethod
    def get_logger(
            name: str,
            log_file: Optional[Path] = None,
            level: int = logging.INFO
    ) -> logging.Logger:
        """
        Retorna ou cria um logger configurado.

        Args:
            name: Nome do logger
            log_file: Caminho para arquivo de log (opcional)
            level: Nível de logging

        Returns:
            Logger configurado
        """
        if name in Logger._loggers:
            return Logger._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Formato do log
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Handler para console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Handler para arquivo (se especificado)
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        Logger._loggers[name] = logger
        return logger


# Logger padrão da aplicação
default_logger = Logger.get_logger('llm_app')