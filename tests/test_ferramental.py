"""Invariantes do ferramental que nenhum teste de comportamento alcança.

São afirmações que os arquivos de configuração fazem sobre si mesmos. Um comentário
pedindo "mantenha estes três valores iguais" não mantém nada igual; o teste mantém.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from importlib.metadata import packages_distributions
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

# Código que roda fora da suíte: `src/` e `scripts/` rodam com `main` + `pipeline` (o que
# `poetry install` traz); `tests/` não entra, porque pytest vive no grupo `dev`. A imagem
# instala só `main` e roda só a API — o fecho de imports dela é cobrado à parte, abaixo.
FONTES_DE_PRODUCAO = ("src", "scripts")
API = RAIZ / "src" / "credito" / "api"


def _versao(caminho: Path, padrao: str) -> str:
    texto = (RAIZ / caminho).read_text(encoding="utf-8")
    casamento = re.search(padrao, texto)
    assert casamento, f"versão do ruff não encontrada em {caminho}"
    return casamento.group(1)


def test_versao_do_ruff_e_a_mesma_nos_tres_lugares():
    # Três pinos separados para a mesma ferramenta: o hook de pre-commit, o job de lint do
    # CI e o grupo `dev`. Divergência entre eles produz o clássico "passa local, falha no
    # CI" por regra nova ou formatação diferente. O `.pre-commit-config.yaml` já pedia que
    # os três ficassem iguais — em prosa, o que não impediu o `pyproject.toml` de usar uma
    # faixa `^0.7.4` enquanto os outros dois fixavam a versão exata.
    pyproject = tomllib.loads((RAIZ / "pyproject.toml").read_text(encoding="utf-8"))
    do_poetry = pyproject["tool"]["poetry"]["group"]["dev"]["dependencies"]["ruff"]
    do_hook = _versao(Path(".pre-commit-config.yaml"), r"ruff-pre-commit\s*\n\s*rev:\s*v(\S+)")
    do_ci = _versao(Path(".github/workflows/ci.yml"), r"pip install ruff==(\S+)")

    assert do_poetry == do_hook == do_ci


def test_versao_do_ruff_no_pyproject_e_exata():
    # Uma faixa aqui deixaria o ambiente local subir uma minor sozinho e sair de sincronia
    # com os outros dois pinos sem ninguém editar nada — o teste acima continuaria verde
    # até a próxima release do ruff. Este fecha essa porta.
    pyproject = tomllib.loads((RAIZ / "pyproject.toml").read_text(encoding="utf-8"))
    declarada = pyproject["tool"]["poetry"]["group"]["dev"]["dependencies"]["ruff"]

    assert re.fullmatch(r"\d+\.\d+\.\d+", declarada), declarada


def _nome_normalizado(bruto: str) -> str:
    return re.sub(r"[-_.]+", "-", bruto).lower()


def _dependencias_do_grupo_main() -> set[str]:
    pyproject = tomllib.loads((RAIZ / "pyproject.toml").read_text(encoding="utf-8"))
    declaradas = pyproject["project"]["dependencies"]
    # PEP 508: "uvicorn[standard] (>=0.32,<1.0)" — o nome é o que vem antes do extra e da
    # especificação de versão.
    return {
        _nome_normalizado(re.split(r"[\s\[(<>=!~;]", linha, maxsplit=1)[0]) for linha in declaradas
    }


def _dependencias_do_grupo_pipeline() -> set[str]:
    pyproject = tomllib.loads((RAIZ / "pyproject.toml").read_text(encoding="utf-8"))
    grupo = pyproject["tool"]["poetry"]["group"]["pipeline"]["dependencies"]
    return {_nome_normalizado(nome) for nome in grupo}


def _modulos_importados_em_producao() -> set[str]:
    modulos: set[str] = set()
    for diretorio in FONTES_DE_PRODUCAO:
        for arquivo in (RAIZ / diretorio).rglob("*.py"):
            arvore = ast.parse(arquivo.read_text(encoding="utf-8"))
            for no in ast.walk(arvore):
                if isinstance(no, ast.Import):
                    modulos.update(alias.name.split(".")[0] for alias in no.names)
                elif isinstance(no, ast.ImportFrom) and no.level == 0 and no.module:
                    modulos.add(no.module.split(".")[0])
    return modulos - set(sys.stdlib_module_names) - {"credito"}


def _distribuicao_de_cada_modulo() -> dict[str, set[str]]:
    origem: dict[str, set[str]] = {}
    for modulo, distribuicoes in packages_distributions().items():
        origem[modulo] = {_nome_normalizado(nome) for nome in distribuicoes}
    return origem


def test_toda_dependencia_do_grupo_main_e_importada_em_producao():
    # O grupo `main` existe para manter a superfície da imagem pequena; quatro pacotes
    # (`fastapi`, `uvicorn`, `prometheus-client`, `scipy`) estavam declarados sem nenhum
    # import correspondente, cada um carregando a própria árvore de transitivas e a própria
    # janela de CVE. A etapa que precisar de um deles volta a declará-lo junto com o código
    # que o usa — e este teste fica vermelho no dia em que alguém declarar antes.
    importados = _modulos_importados_em_producao()
    origem = _distribuicao_de_cada_modulo()
    fornecidas = {
        distribuicao for modulo in importados for distribuicao in origem.get(modulo, set())
    }

    assert _dependencias_do_grupo_main() <= fornecidas


def test_toda_dependencia_do_grupo_pipeline_e_importada_em_producao():
    # A mesma disciplina do teste acima, para o grupo que só o pipeline usa: declarar sem
    # importar é peso e janela de CVE sem uso.
    importados = _modulos_importados_em_producao()
    origem = _distribuicao_de_cada_modulo()
    fornecidas = {
        distribuicao for modulo in importados for distribuicao in origem.get(modulo, set())
    }

    assert _dependencias_do_grupo_pipeline() <= fornecidas


def test_todo_import_de_producao_tem_dependencia_declarada():
    # A outra direção, que é o mesmo defeito de sinal trocado: `numpy` era importado
    # direto por `adulterate.py` e `evaluate.py` sem estar declarado, chegando de carona
    # pelo pandas e pelo scikit-learn. Funciona até o dia em que uma dessas duas troque de
    # versão principal — e aí quebra num módulo que nunca pediu nada ao numpy.
    declaradas = _dependencias_do_grupo_main() | _dependencias_do_grupo_pipeline()
    origem = _distribuicao_de_cada_modulo()

    nao_declarados = {
        modulo
        for modulo in _modulos_importados_em_producao()
        if not (origem.get(modulo, set()) & declaradas)
    }

    assert nao_declarados == set()


def _arquivo_do_modulo(nome: str) -> Path | None:
    """`credito.a.b` -> o arquivo `src/credito/a/b.py` ou `src/credito/a/b/__init__.py`."""
    base = RAIZ / "src" / Path(*nome.split("."))
    for candidato in (base.with_suffix(".py"), base / "__init__.py"):
        if candidato.exists():
            return candidato
    return None


def _fecho_de_imports(inicio: list[Path]) -> set[str]:
    """Módulos de terceiros alcançados a partir de `inicio`, seguindo os imports internos
    (`credito.*`) arquivo a arquivo — estaticamente, sem importar nada."""
    pendentes, visitados, terceiros = list(inicio), set(), set()
    while pendentes:
        arquivo = pendentes.pop()
        if arquivo in visitados:
            continue
        visitados.add(arquivo)
        for no in ast.walk(ast.parse(arquivo.read_text(encoding="utf-8"))):
            nomes: list[str] = []
            if isinstance(no, ast.Import):
                nomes = [alias.name for alias in no.names]
            elif isinstance(no, ast.ImportFrom) and no.level == 0 and no.module:
                # `from credito.a import b` pode trazer o submódulo `b`, não só um nome.
                nomes = [no.module] + [f"{no.module}.{alias.name}" for alias in no.names]
            for nome in nomes:
                if nome.split(".")[0] == "credito":
                    if (alvo := _arquivo_do_modulo(nome)) is not None:
                        pendentes.append(alvo)
                elif nome.split(".")[0] not in sys.stdlib_module_names:
                    terceiros.add(nome.split(".")[0])
    return terceiros


def test_a_api_so_alcanca_dependencias_que_a_imagem_instala():
    # A imagem instala `--only main`. Se algum módulo que a API importa — direta ou
    # transitivamente, via `credito.*` — passar a importar MLflow, Evidently ou Pandera, o
    # container sobe e quebra no primeiro import, e só o build da CI perceberia, minutos
    # depois. Este teste percebe na hora, sem Docker.
    alcancados = _fecho_de_imports(sorted(API.glob("*.py")))
    origem = _distribuicao_de_cada_modulo()
    main = _dependencias_do_grupo_main()

    fora_da_imagem = {modulo for modulo in alcancados if not (origem.get(modulo, set()) & main)}

    assert fora_da_imagem == set()
    # Sanidade do próprio fecho: um fecho vazio passaria na asserção acima por vacuidade.
    assert {"fastapi", "prometheus_client", "pydantic"} <= alcancados
    # O que um fecho estático não vê: `joblib.load` exige `xgboost` e `sklearn` para
    # desserializar o campeão, sem import nenhum no código. Os dois estão em `main` por
    # isso, e o smoke test de `/health` no job de build da CI é quem cobre esse caminho.
