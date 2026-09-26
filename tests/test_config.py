"""A configuração é a única fonte de caminhos e limiares do projeto."""

from __future__ import annotations

from pathlib import Path

from credito.config import Settings, get_settings


def test_prefixo_de_ambiente_e_credito(monkeypatch):
    monkeypatch.setenv("CREDITO_RANDOM_SEED", "7")
    assert Settings().random_seed == 7


def test_diretorios_derivam_do_data_dir():
    settings = Settings(data_dir=Path("/tmp/x"))
    assert settings.raw_dir == Path("/tmp/x/raw")
    assert settings.interim_dir == Path("/tmp/x/interim")


def test_get_settings_devolve_a_mesma_instancia():
    assert get_settings() is get_settings()


def test_md5_do_dataset_esta_fixado():
    # O download verifica integridade contra este valor; se ele virar vazio, a verificação
    # passa a não verificar nada.
    assert Settings().dataset_md5 == "6d013a631b97b13d5372f097697e25a1"


def test_pisos_do_gate_estao_calibrados():
    # Um piso em zero é um gate desligado que continua parecendo ligado no código. Estes
    # valores vêm do campeão medido no conjunto de teste (auc_pr 0,3716, recall positivo
    # 0,6915) menos a margem de 0,05; o teste existe para que zerá-los seja uma decisão
    # visível, não um efeito colateral de alguém "destravando o pipeline".
    settings = Settings()
    assert settings.min_auc_pr == 0.3216
    assert settings.min_recall_positivo == 0.6415
