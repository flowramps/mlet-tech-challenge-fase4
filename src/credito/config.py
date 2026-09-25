"""Configuração central da aplicação, resolvida por variáveis de ambiente."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Parâmetros de execução. Prefixo de ambiente: ``CREDITO_``."""

    model_config = SettingsConfigDict(
        env_prefix="CREDITO_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"
    metrics_dir: Path = PROJECT_ROOT / "metrics"
    reports_dir: Path = PROJECT_ROOT / "reports"

    # Redistribuição curada pelo OpenML do dataset da competição Kaggle "Give Me Some
    # Credit" (2011). O Kaggle exige autenticação, o que quebraria o clone limpo e o job
    # de build do CI; o OpenML serve o mesmo arquivo por URL pública com checksum.
    dataset_url: str = "https://openml.org/data/v1/download/22125240/GiveMeSomeCredit.arff"
    dataset_md5: str = "6d013a631b97b13d5372f097697e25a1"
    dataset_filename: str = "GiveMeSomeCredit.arff"

    model_filename: str = "model.joblib"

    random_seed: int = 42
    test_size: float = 0.2
    validation_size: float = 0.2

    # Pisos do gate de promoção. Ficam em zero até a Task 10, que os calibra a partir do
    # primeiro treino real — um piso chutado antes da medição é enfeite que sempre passa.
    min_auc_pr: float = 0.0
    min_recall_positivo: float = 0.0

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def dataset_path(self) -> Path:
        return self.raw_dir / self.dataset_filename

    @property
    def model_path(self) -> Path:
        return self.models_dir / self.model_filename


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
