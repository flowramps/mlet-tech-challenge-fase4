"""O registro de decisão é o que dá objeto ao direito de revisão (LGPD, art. 20): sem ele,
uma recusa passada não pode ser reconstruída. Estes testes travam as três propriedades que
fazem dele um registro de auditoria, e não um log: nada se perde, nada se sobrescreve, e o
que venceu o prazo de retenção sai — e só o que venceu."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta

import pytest

from credito.governanca.decisoes import (
    RETENCAO_ANOS,
    Decisao,
    RegistroJsonl,
    limite_de_retencao,
    sha256_do_arquivo,
)


def _decisao(id_decisao: str = "d-1", *, registrada_em: datetime | None = None) -> Decisao:
    return Decisao(
        id_decisao=id_decisao,
        registrada_em=(registrada_em or datetime(2026, 10, 3, 12, 0, tzinfo=UTC)).isoformat(),
        features={"age": 45.0, "MonthlyIncome": 5000.0},
        probabilidade_inadimplencia=0.18,
        aprovado=True,
        limiar=0.5,
        candidato="xgboost",
        modelo_sha256="abc123",
    )


def test_registrar_e_buscar_devolvem_a_mesma_decisao(tmp_path):
    registro = RegistroJsonl(tmp_path / "decisoes.jsonl")
    original = _decisao()

    registro.registrar(original)

    assert registro.buscar("d-1") == original


def test_buscar_id_inexistente_devolve_none_em_vez_de_inventar(tmp_path):
    registro = RegistroJsonl(tmp_path / "decisoes.jsonl")
    registro.registrar(_decisao("d-1"))

    assert registro.buscar("nao-existe") is None


def test_buscar_sem_arquivo_ainda_devolve_none(tmp_path):
    # O arquivo só nasce na primeira decisão: consultar antes disso não é erro.
    assert RegistroJsonl(tmp_path / "decisoes.jsonl").buscar("d-1") is None


def test_registro_e_append_only_uma_linha_por_decisao(tmp_path):
    caminho = tmp_path / "decisoes.jsonl"
    registro = RegistroJsonl(caminho)

    registro.registrar(_decisao("d-1"))
    registro.registrar(_decisao("d-2"))

    linhas = caminho.read_text(encoding="utf-8").splitlines()
    assert [json.loads(linha)["id_decisao"] for linha in linhas] == ["d-1", "d-2"]


def test_registro_cria_o_diretorio_na_primeira_decisao(tmp_path):
    caminho = tmp_path / "ainda" / "nao" / "existe" / "decisoes.jsonl"

    RegistroJsonl(caminho).registrar(_decisao())

    assert caminho.exists()


def test_decisao_gravada_durante_um_expurgo_nao_se_perde(tmp_path, monkeypatch):
    # O risco de concorrência que importa: o expurgo lê o arquivo, escreve um novo e o
    # troca pelo antigo. Uma decisão gravada no meio disso iria para o arquivo antigo e
    # sumiria na troca — uma decisão emitida e sem registro, em silêncio. A corrida é
    # forçada: a troca do expurgo espera até a gravação concorrente já ter sido tentada.
    #
    # (Escritas simultâneas de decisões não precisam de teste próprio: linhas curtas em
    # modo append já são atômicas no sistema operacional — medido, um teste disso passava
    # com ou sem a trava, e por isso não está aqui.)
    import credito.governanca.decisoes as modulo

    caminho = tmp_path / "decisoes.jsonl"
    registro = RegistroJsonl(caminho)
    limite = datetime(2021, 10, 3, tzinfo=UTC)
    registro.registrar(_decisao("velha", registrada_em=limite - timedelta(days=1)))

    troca_iniciada = threading.Event()
    gravacao_tentada = threading.Event()
    troca_original = modulo.os.replace

    def troca_atrasada(origem, destino):
        troca_iniciada.set()  # o expurgo já leu o arquivo e vai trocá-lo
        gravacao_tentada.wait(timeout=2)
        troca_original(origem, destino)

    monkeypatch.setattr(modulo.os, "replace", troca_atrasada)

    expurgo = threading.Thread(target=registro.expurgar, kwargs={"antes_de": limite})
    expurgo.start()
    assert troca_iniciada.wait(timeout=2)
    nova = threading.Thread(target=registro.registrar, args=(_decisao("chegou-no-meio"),))
    nova.start()
    nova.join(timeout=0.3)
    gravacao_tentada.set()
    expurgo.join()
    nova.join()

    assert registro.buscar("chegou-no-meio") is not None
    assert registro.buscar("velha") is None


def test_limite_de_retencao_e_cinco_anos_antes_de_agora():
    agora = datetime(2026, 10, 3, tzinfo=UTC)

    assert RETENCAO_ANOS == 5
    assert limite_de_retencao(agora) == datetime(2021, 10, 3, tzinfo=UTC)


def test_limite_de_retencao_em_29_de_fevereiro_nao_quebra():
    # 2028-02-29 menos cinco anos cai num ano sem 29 de fevereiro.
    assert limite_de_retencao(datetime(2028, 2, 29, tzinfo=UTC)) == datetime(
        2023, 2, 28, tzinfo=UTC
    )


def test_expurgar_remove_so_o_que_venceu_e_conta_quantas(tmp_path):
    registro = RegistroJsonl(tmp_path / "decisoes.jsonl")
    limite = datetime(2021, 10, 3, tzinfo=UTC)
    registro.registrar(_decisao("velha", registrada_em=limite - timedelta(seconds=1)))
    registro.registrar(_decisao("no-limite", registrada_em=limite))
    registro.registrar(_decisao("nova", registrada_em=limite + timedelta(days=1)))

    removidas = registro.expurgar(antes_de=limite)

    # Exatamente no limite ainda está dentro do prazo: sai só o que é estritamente
    # anterior. Um `<=` apagaria uma decisão no último dia em que ela pode ser revista.
    assert removidas == 1
    assert registro.buscar("velha") is None
    assert registro.buscar("no-limite") is not None
    assert registro.buscar("nova") is not None


def test_expurgar_sem_arquivo_nao_remove_nada(tmp_path):
    assert (
        RegistroJsonl(tmp_path / "decisoes.jsonl").expurgar(
            antes_de=datetime(2021, 1, 1, tzinfo=UTC)
        )
        == 0
    )


def test_decisao_recusa_data_sem_fuso():
    # Comparar uma data sem fuso com o limite de retenção (com fuso) quebraria o expurgo
    # — ou, pior, compararia horários de fusos diferentes como se fossem o mesmo.
    with pytest.raises(ValueError, match="fuso"):
        _decisao(registrada_em=datetime(2026, 10, 3, 12, 0))


def test_sha256_do_arquivo_muda_quando_o_modelo_muda(tmp_path):
    caminho = tmp_path / "model.joblib"
    caminho.write_bytes(b"campeao-1")
    antes = sha256_do_arquivo(caminho)
    caminho.write_bytes(b"campeao-2")

    assert sha256_do_arquivo(caminho) != antes
    assert len(antes) == 64
