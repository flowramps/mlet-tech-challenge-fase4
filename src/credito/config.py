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
    mlruns_dir: Path = PROJECT_ROOT / "mlruns"
    # Registro de decisão de crédito: guarda dado pessoal, por isso fora do git e com
    # retenção executável (ver `credito.governanca.decisoes`).
    decisoes_dir: Path = PROJECT_ROOT / "decisoes"

    # Nome do experimento MLflow que agrupa os runs do monitoramento — conjunto fechado de
    # um elemento só, mas nomeado em vez de literal espalhado pelo chamador, pela mesma
    # razão que `model_filename` já é campo em vez de string solta em `model.train`.
    mlflow_experimento: str = "credito-monitoramento"
    # E o que agrupa os runs do treino: o log de execução de cada `make train`, inclusive
    # os que falham (ver `credito.tracking.execucao`).
    mlflow_experimento_treino: str = "credito-treino"

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

    # Pisos do gate de promoção, calibrados sobre medição e não sobre chute — um piso
    # inventado antes de existir modelo é enfeite que sempre passa. A margem de 0,05 em
    # ambos absorve a variação natural entre retreinos (reordenação de partição, versão
    # de biblioteca) sem abrir espaço para uma queda real de qualidade passar batido.
    #
    # Piso de AUC-PR. Calibrado a partir do AUC-PR do campeão (xgboost) no conjunto de
    # teste da Referência, 0,3716, menos a margem.
    min_auc_pr: float = 0.3216

    # Piso de recall da classe positiva — a métrica de negócio: aprovar um inadimplente
    # custa o valor emprestado, recusar um bom pagador custa a margem. Calibrado a partir
    # do recall do mesmo campeão no mesmo conjunto de teste, 0,6915, menos a margem.
    min_recall_positivo: float = 0.6415

    # Convenção de risco de crédito para PSI. Não são medições deste dataset: são os cortes
    # que a indústria usa há décadas para decidir se um modelo de score ainda vale. Abaixo
    # de 0,10 a variação é ruído amostral no tamanho de lote que usamos; acima de 0,25 a
    # distribuição mudou o bastante para a decisão do modelo não se sustentar.
    #
    # A nula empírica do PSI neste dado JÁ FOI MEDIDA — `credito.drift.calibration.
    # distribuicao_nula_psi`, com a tabela das dez features publicada no README. Rodada
    # contra a Referência real (117.917 linhas), 10 bins, semente 42, 5.000 reamostras por
    # feature, em lotes do tamanho que o detector de fato vê (23.584 linhas): o pior p95
    # entre as dez features é 0,001610 (`NumberRealEstateLoansOrLines`) e o pior p99 é
    # 0,001978. Ou seja, `psi_atencao` está ~62x acima do PSI que o acaso produz no p95 e
    # `psi_critico` ~126x acima do p99; em 50.000 medições sob a nula, o maior PSI que o
    # acaso produziu foi 0,002714. Para este dado, com estes bins e este tamanho de lote,
    # a convenção é conservadora por cerca de duas ordens de grandeza.
    #
    # E mesmo assim estes dois valores continuam sendo a convenção, não a medição: a nula
    # medida não alimenta limiar nenhum no código. Isso é dívida declarada, com a razão
    # escrita no README — cortar em 0,0016 transformaria variação amostral irrelevante em
    # alarme diário, o que é pior que o problema. O que a medição mudou foi o status da
    # escolha: a banda entre 0,0016 e 0,10 é margem de tolerância deliberada, não zona de
    # incerteza estatística.
    psi_atencao: float = 0.10
    psi_critico: float = 0.25

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

    @property
    def decisoes_path(self) -> Path:
        return self.decisoes_dir / "decisoes.jsonl"

    @property
    def mlflow_tracking_uri(self) -> str:
        # SQLite, nunca o backend de arquivo do MLflow 3.x (em modo de manutenção) — ver
        # o docstring de `credito.tracking.mlflow_client`.
        return f"sqlite:///{self.mlruns_dir}/mlflow.db"

    @property
    def mlflow_artifact_location(self) -> str:
        # Explícito, nunca implícito: sem isto o MLflow grava em `./mlruns/<id>/...`
        # relativo ao cwd do processo, mesmo com o tracking apontando para outro lugar
        # (ver o mesmo docstring).
        return str(self.mlruns_dir / "artifacts")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
