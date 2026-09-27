"""A configuração é a única fonte de caminhos e limiares do projeto."""

from __future__ import annotations

from pathlib import Path

from credito.config import PROJECT_ROOT, Settings, get_settings


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


def test_limiares_de_drift_seguem_a_convencao_do_setor():
    # Estes dois números não vêm de medição deste dataset — são a convenção de risco de
    # crédito que a indústria usa para PSI (ver o comentário em `config.py`). O teste fixa
    # os valores e a ordem entre eles: uma troca acidental deixaria `psi_atencao` acima de
    # `psi_critico`, e `classificar()` nunca devolveria CRITICO.
    #
    # Fixar os valores também é o que mantém honesto o comentário de `config.py`, que
    # agora cita a nula empírica medida contra a Referência real e diz o quanto cada corte
    # está acima dela (~62x o pior p95, ~126x o pior p99). Essas razões são derivadas
    # destes dois números: mudar um deles sem rever o comentário publicaria uma folga que
    # não é mais a real — o mesmo tipo de apodrecimento que este comentário já sofreu uma
    # vez, quando afirmava que a medição da nula "ainda não existe" depois de ela ter sido
    # implementada nesta mesma branch.
    settings = Settings()
    assert settings.psi_atencao == 0.10
    assert settings.psi_critico == 0.25
    assert settings.psi_atencao < settings.psi_critico


def _entradas_do_env_example() -> dict[str, str]:
    caminho = Path(__file__).resolve().parents[1] / ".env.example"
    entradas: dict[str, str] = {}
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        limpa = linha.strip()
        if not limpa or limpa.startswith("#") or "=" not in limpa:
            continue
        chave, valor = limpa.split("=", 1)
        entradas[chave.strip()] = valor.strip()
    return entradas


def test_env_example_reproduz_os_defaults_que_declara():
    # O arquivo existe para dizer quais são os defaults; se ele mentir, copiá-lo para .env
    # muda o comportamento em silêncio, que é o pior desfecho possível para um arquivo de
    # exemplo. Este teste compara cada valor **ativo** com o default de verdade.
    padrao = Settings()
    declarados = _entradas_do_env_example()

    assert declarados, ".env.example não declara nenhuma variável ativa"
    for chave, valor in declarados.items():
        campo = chave.removeprefix("CREDITO_").lower()
        assert hasattr(padrao, campo), chave
        assert str(getattr(padrao, campo)) == valor, chave


def test_defaults_de_diretorio_sao_absolutos_e_ancorados_no_repositorio():
    # A razão de os quatro diretórios estarem comentados no .env.example: o default é
    # absoluto, ancorado no repositório, e nenhum caminho relativo o reproduz a não ser
    # por coincidência de diretório de trabalho. Declará-los como valor relativo era
    # exatamente essa coincidência escrita como se fosse o default.
    padrao = Settings()
    declarados = _entradas_do_env_example()

    for campo in ("data_dir", "models_dir", "metrics_dir", "reports_dir"):
        valor = getattr(padrao, campo)
        assert valor.is_absolute(), campo
        assert valor.parent == PROJECT_ROOT, campo
        assert f"CREDITO_{campo.upper()}" not in declarados, campo
