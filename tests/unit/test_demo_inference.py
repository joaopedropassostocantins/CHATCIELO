"""
tests/unit/test_demo_inference.py
==================================
Testes unitários para demo_inference.py.

Cobertura obrigatória (CLAUDE.md §3.1):
  - Mascaramento LGPD: scrub_pii() — CPF, cartão, email, telefone
  - Roteamento de intents: detect_intent()
  - Construção de prompts: build_cot_prompt()
  - Dataset sintético: build_synthetic_dataset()
  - Extração de texto: _extract_between(), _extract_after()
  - Salvamento CSV: save_submission() — formato e conteúdo
  - Pipeline main(): smoke test via --dry-run (sem GPU)

Todos os fixtures usam dados sintéticos — nenhum CPF/cartão real (CLAUDE.md rule #3).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Importações do módulo alvo
from demo_inference import (
    MerchantInteraction,
    _extract_after,
    _extract_between,
    build_cot_prompt,
    build_synthetic_dataset,
    detect_intent,
    load_config,
    main,
    save_submission,
    scrub_pii,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_cpf() -> str:
    """CPF sintético (formato válido, mas falso — não pertence a ninguém)."""
    return "123.456.789-00"


@pytest.fixture()
def synthetic_card() -> str:
    """Número de cartão sintético de 16 dígitos."""
    return "4111 1111 1111 1111"


@pytest.fixture()
def synthetic_email() -> str:
    """E-mail sintético."""
    return "lojista.teste@exemplo.com.br"


@pytest.fixture()
def synthetic_phone() -> str:
    """Telefone sintético no padrão BR."""
    return "(11) 98765-4321"


@pytest.fixture()
def basic_interaction() -> MerchantInteraction:
    """Interação sintética básica para testes."""
    return MerchantInteraction(
        id="test_001",
        segment="MEI",
        message="Minha taxa está muito alta.",
        expected_intent="reclamacao_taxa",
    )


# ---------------------------------------------------------------------------
# Testes: scrub_pii() — Mascaramento LGPD
# ---------------------------------------------------------------------------


class TestScrubPii:
    """Testes para a função de mascaramento de PII (LGPD Art. 46)."""

    def test_scrub_cpf_formatted(self, synthetic_cpf: str) -> None:
        """CPF com pontuação deve ser removido."""
        result = scrub_pii(f"Meu CPF é {synthetic_cpf} para consulta.")
        assert synthetic_cpf not in result
        assert "[CPF_REMOVIDO]" in result

    def test_scrub_cpf_unformatted(self) -> None:
        """CPF sem pontuação (11 dígitos) deve ser removido."""
        raw_cpf = "12345678900"
        result = scrub_pii(f"CPF: {raw_cpf}")
        assert raw_cpf not in result

    def test_scrub_card_number(self, synthetic_card: str) -> None:
        """Número de cartão de crédito deve ser removido."""
        result = scrub_pii(f"Cartão: {synthetic_card}")
        assert synthetic_card not in result
        assert "[CARTAO_REMOVIDO]" in result

    def test_scrub_email(self, synthetic_email: str) -> None:
        """E-mail deve ser removido."""
        result = scrub_pii(f"Me contate em {synthetic_email}.")
        assert synthetic_email not in result
        assert "[EMAIL_REMOVIDO]" in result

    def test_scrub_phone(self, synthetic_phone: str) -> None:
        """Número de telefone brasileiro deve ser removido."""
        result = scrub_pii(f"Ligue para {synthetic_phone}.")
        assert synthetic_phone not in result
        assert "[TELEFONE_REMOVIDO]" in result

    def test_scrub_multiple_pii_in_one_string(
        self, synthetic_cpf: str, synthetic_email: str
    ) -> None:
        """Múltiplos tipos de PII no mesmo texto devem ser todos removidos."""
        text = f"CPF: {synthetic_cpf}, email: {synthetic_email}"
        result = scrub_pii(text)
        assert synthetic_cpf not in result
        assert synthetic_email not in result

    def test_scrub_no_pii_unchanged(self) -> None:
        """Texto sem PII não deve ser alterado."""
        clean = "Olá, gostaria de saber sobre taxas de MDR."
        result = scrub_pii(clean)
        assert result == clean

    def test_scrub_empty_string(self) -> None:
        """String vazia deve retornar string vazia."""
        assert scrub_pii("") == ""

    def test_scrub_type_error(self) -> None:
        """Input não-string deve levantar TypeError."""
        with pytest.raises(TypeError):
            scrub_pii(12345)  # type: ignore[arg-type]

    def test_scrub_returns_string(self) -> None:
        """scrub_pii deve sempre retornar str."""
        result = scrub_pii("texto qualquer")
        assert isinstance(result, str)

    def test_scrub_does_not_leave_cpf_pattern(self) -> None:
        """Após scrub, nenhum padrão CPF deve existir no resultado."""
        text = "CPF 111.222.333-44 e CPF 99988877766"
        result = scrub_pii(text)
        cpf_pattern = re.compile(r"\d{3}\.?\d{3}\.?\d{3}-?\d{2}")
        assert not cpf_pattern.search(result)


# ---------------------------------------------------------------------------
# Testes: detect_intent() — Roteamento
# ---------------------------------------------------------------------------


class TestDetectIntent:
    """Testes para o roteador de intents por palavra-chave."""

    @pytest.mark.parametrize(
        "message,expected",
        [
            ("Minha taxa está muito alta", "reclamacao_taxa"),
            ("O MDR que aparece no extrato é diferente", "duvida_taxa_mdr"),
            ("Como funciona o parcelamento?", "duvida_parcelamento"),
            ("A maquininha está com problema", "problema_terminal"),
            ("Quero cancelar o contrato", "cancelamento"),
            ("Não sei o que aconteceu", "outros"),
        ],
    )
    def test_intent_detection(self, message: str, expected: str) -> None:
        """Intent correto deve ser detectado para cada palavra-chave."""
        assert detect_intent(message) == expected

    def test_case_insensitive(self) -> None:
        """Detecção deve ser case-insensitive."""
        assert detect_intent("TAXA ALTA") == "reclamacao_taxa"
        assert detect_intent("Cancelar") == "cancelamento"

    def test_unknown_returns_outros(self) -> None:
        """Mensagem sem palavras-chave conhecidas retorna 'outros'."""
        assert detect_intent("olá tudo bem?") == "outros"

    def test_empty_message(self) -> None:
        """String vazia deve retornar 'outros' sem erro."""
        assert detect_intent("") == "outros"

    def test_returns_string(self) -> None:
        """detect_intent sempre retorna str."""
        result = detect_intent("qualquer mensagem")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Testes: build_cot_prompt() — Construção de prompts CoT
# ---------------------------------------------------------------------------


class TestBuildCotPrompt:
    """Testes para a construção do prompt Chain-of-Thought."""

    def test_prompt_contains_think_tag(self, basic_interaction: MerchantInteraction) -> None:
        """Prompt deve conter marcador <think> para Chain-of-Thought."""
        prompt = build_cot_prompt(basic_interaction, "reclamacao_taxa")
        assert "<think>" in prompt

    def test_prompt_contains_segment(self, basic_interaction: MerchantInteraction) -> None:
        """Prompt deve mencionar o segmento do lojista."""
        prompt = build_cot_prompt(basic_interaction, "reclamacao_taxa")
        assert basic_interaction.segment in prompt

    def test_prompt_contains_message(self, basic_interaction: MerchantInteraction) -> None:
        """Prompt deve incluir a mensagem do lojista."""
        prompt = build_cot_prompt(basic_interaction, "reclamacao_taxa")
        assert basic_interaction.message in prompt

    def test_prompt_contains_intent(self, basic_interaction: MerchantInteraction) -> None:
        """Prompt deve mencionar o intent detectado."""
        prompt = build_cot_prompt(basic_interaction, "reclamacao_taxa")
        assert "reclamacao_taxa" in prompt

    def test_prompt_all_segments(self) -> None:
        """Prompt deve ser gerado sem erro para MEI, VAREJO e CORPORATE."""
        for segment in ("MEI", "VAREJO", "CORPORATE"):
            interaction = MerchantInteraction(
                id=f"seg_{segment}",
                segment=segment,
                message="Teste",
                expected_intent="outros",
            )
            prompt = build_cot_prompt(interaction, "outros")
            assert isinstance(prompt, str)
            assert len(prompt) > 0

    def test_prompt_unknown_segment_fallback(self) -> None:
        """Segmento desconhecido não deve causar erro — usa tom padrão."""
        interaction = MerchantInteraction(
            id="seg_unknown",
            segment="UNKNOWN",
            message="Teste",
            expected_intent="outros",
        )
        prompt = build_cot_prompt(interaction, "outros")
        assert isinstance(prompt, str)

    def test_prompt_is_string(self, basic_interaction: MerchantInteraction) -> None:
        """build_cot_prompt sempre retorna str."""
        result = build_cot_prompt(basic_interaction, "reclamacao_taxa")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Testes: build_synthetic_dataset() — Dataset sem PII
# ---------------------------------------------------------------------------


class TestBuildSyntheticDataset:
    """Testes para o dataset sintético de lojistas Cielo."""

    def test_returns_five_interactions(self) -> None:
        """Dataset deve conter exatamente 5 interações."""
        dataset = build_synthetic_dataset()
        assert len(dataset) == 5

    def test_all_have_unique_ids(self) -> None:
        """Todos os IDs devem ser únicos."""
        dataset = build_synthetic_dataset()
        ids = [i.id for i in dataset]
        assert len(ids) == len(set(ids))

    def test_all_have_valid_segments(self) -> None:
        """Todos os segmentos devem ser MEI, VAREJO ou CORPORATE."""
        valid_segments = {"MEI", "VAREJO", "CORPORATE"}
        dataset = build_synthetic_dataset()
        for interaction in dataset:
            assert interaction.segment in valid_segments

    def test_no_real_cpf_in_messages(self) -> None:
        """Nenhuma mensagem deve conter padrão de CPF real."""
        cpf_pattern = re.compile(r"\d{3}\.?\d{3}\.?\d{3}-?\d{2}")
        dataset = build_synthetic_dataset()
        for interaction in dataset:
            assert not cpf_pattern.search(interaction.message), (
                f"PII detectado em {interaction.id}: {interaction.message}"
            )

    def test_no_card_numbers_in_messages(self) -> None:
        """Nenhuma mensagem deve conter número de cartão."""
        dataset = build_synthetic_dataset()
        for interaction in dataset:
            # Sequências de 16 dígitos contínuos são suspeitas
            assert not re.search(r"\b\d{16}\b", interaction.message), (
                f"Possível cartão em {interaction.id}"
            )

    def test_all_messages_nonempty(self) -> None:
        """Todas as mensagens devem ser não-vazias."""
        dataset = build_synthetic_dataset()
        for interaction in dataset:
            assert len(interaction.message.strip()) > 0

    def test_expected_intents_present(self) -> None:
        """Os 5 intents esperados devem estar presentes no dataset."""
        dataset = build_synthetic_dataset()
        intents = {i.expected_intent for i in dataset}
        assert "reclamacao_taxa" in intents
        assert "cancelamento" in intents
        assert "problema_terminal" in intents


# ---------------------------------------------------------------------------
# Testes: _extract_between() e _extract_after()
# ---------------------------------------------------------------------------


class TestTextExtraction:
    """Testes para funções auxiliares de extração de texto."""

    def test_extract_between_basic(self) -> None:
        """Extrai conteúdo entre marcadores."""
        text = "antes <think>conteúdo pensamento</think> depois"
        result = _extract_between(text, "<think>", "</think>")
        assert result == "conteúdo pensamento"

    def test_extract_between_not_found(self) -> None:
        """Retorna string vazia se marcadores não existirem."""
        result = _extract_between("texto sem marcadores", "<think>", "</think>")
        assert result == ""

    def test_extract_between_multiline(self) -> None:
        """Extrai conteúdo multiline entre marcadores."""
        text = "<think>\nlinha 1\nlinha 2\n</think>"
        result = _extract_between(text, "<think>", "</think>")
        assert "linha 1" in result
        assert "linha 2" in result

    def test_extract_after_basic(self) -> None:
        """Extrai tudo após o marcador."""
        text = "</think>\nEsta é a resposta final."
        result = _extract_after(text, "</think>")
        assert "resposta final" in result

    def test_extract_after_not_found(self) -> None:
        """Retorna string vazia se marcador não encontrado."""
        result = _extract_after("texto", "</think>")
        assert result == ""

    def test_extract_after_at_start(self) -> None:
        """Funciona quando marcador está no início."""
        text = "</think>resposta"
        result = _extract_after(text, "</think>")
        assert result == "resposta"


# ---------------------------------------------------------------------------
# Testes: save_submission() — Formato Kaggle
# ---------------------------------------------------------------------------


class TestSaveSubmission:
    """Testes para o salvamento do submission.csv."""

    def test_creates_csv_file(self, tmp_path: Path) -> None:
        """submission.csv deve ser criado no caminho especificado."""
        output = tmp_path / "submission.csv"
        save_submission([{"id": "001", "prediction": "resposta"}], output)
        assert output.exists()

    def test_csv_has_correct_headers(self, tmp_path: Path) -> None:
        """CSV deve ter colunas 'id' e 'prediction'."""
        output = tmp_path / "submission.csv"
        save_submission([{"id": "001", "prediction": "teste"}], output)
        with open(output, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ["id", "prediction"]

    def test_csv_row_count(self, tmp_path: Path) -> None:
        """CSV deve ter tantas linhas quanto registros de entrada."""
        rows = [{"id": f"id_{i}", "prediction": f"pred_{i}"} for i in range(5)]
        output = tmp_path / "submission.csv"
        save_submission(rows, output)
        with open(output, encoding="utf-8") as f:
            data = list(csv.DictReader(f))
        assert len(data) == 5

    def test_csv_prediction_truncated_to_500(self, tmp_path: Path) -> None:
        """Predictions longas devem ser truncadas a 500 caracteres."""
        long_pred = "x" * 1000
        output = tmp_path / "submission.csv"
        save_submission([{"id": "001", "prediction": long_pred}], output)
        with open(output, encoding="utf-8") as f:
            row = list(csv.DictReader(f))[0]
        assert len(row["prediction"]) <= 500

    def test_csv_empty_results(self, tmp_path: Path) -> None:
        """Lista vazia deve criar CSV apenas com header."""
        output = tmp_path / "submission.csv"
        save_submission([], output)
        with open(output, encoding="utf-8") as f:
            content = f.read()
        assert "id,prediction" in content

    def test_csv_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Diretórios pai devem ser criados automaticamente."""
        output = tmp_path / "nested" / "dir" / "submission.csv"
        save_submission([{"id": "001", "prediction": "resp"}], output)
        assert output.exists()

    def test_csv_id_preserved(self, tmp_path: Path) -> None:
        """IDs devem ser preservados sem alteração."""
        output = tmp_path / "submission.csv"
        save_submission(
            [{"id": "cielo_001", "prediction": "ok"}, {"id": "cielo_002", "prediction": "ok"}],
            output,
        )
        with open(output, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["id"] == "cielo_001"
        assert rows[1]["id"] == "cielo_002"


# ---------------------------------------------------------------------------
# Testes: load_config()
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """Testes para carregamento do config.yaml."""

    def test_loads_successfully(self) -> None:
        """config.yaml deve carregar sem erros."""
        config = load_config()
        assert isinstance(config, dict)

    def test_has_model_key(self) -> None:
        """Config deve ter chave 'model'."""
        config = load_config()
        assert "model" in config

    def test_has_intent_routing(self) -> None:
        """Config deve ter chave 'intent_routing'."""
        config = load_config()
        assert "intent_routing" in config

    def test_has_security_section(self) -> None:
        """Config deve ter seção 'security' com regras LGPD."""
        config = load_config()
        assert "security" in config
        assert "lgpd" in config["security"]

    def test_has_monitoring_kpis(self) -> None:
        """Config deve ter KPIs de monitoramento."""
        config = load_config()
        assert "monitoring" in config
        assert "kpis" in config["monitoring"]

    def test_p99_sla_is_300ms(self) -> None:
        """SLA P99 deve ser 300ms (alinhado com CLAUDE.md §3.3)."""
        config = load_config()
        p99 = config["monitoring"]["kpis"]["latency"]["p99_sla_ms"]
        assert p99 == 300

    def test_lgpd_enabled(self) -> None:
        """LGPD deve estar habilitada por padrão."""
        config = load_config()
        assert config["security"]["lgpd"]["enabled"] is True

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """Arquivo inexistente deve levantar FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_intent_rules_have_required_keys(self) -> None:
        """Cada regra de intent deve ter 'intent', 'action' e 'priority'."""
        config = load_config()
        for rule in config["intent_routing"]["rules"]:
            assert "intent" in rule
            assert "action" in rule
            assert "priority" in rule


# ---------------------------------------------------------------------------
# Testes: main() — Pipeline completo (dry-run, sem GPU)
# ---------------------------------------------------------------------------


class TestMainPipeline:
    """Testes de smoke para o pipeline principal em modo dry-run."""

    def test_main_dry_run_completes(self, tmp_path: Path) -> None:
        """Pipeline dry-run deve completar sem erros."""
        with patch("demo_inference.OUTPUT_FILE", tmp_path / "submission.csv"):
            main(dry_run=True)  # não deve levantar exceção

    def test_main_dry_run_creates_submission(self, tmp_path: Path) -> None:
        """Pipeline dry-run deve criar submission.csv."""
        output = tmp_path / "submission.csv"
        with patch("demo_inference.OUTPUT_FILE", output):
            main(dry_run=True)
        assert output.exists()

    def test_main_dry_run_submission_has_5_rows(self, tmp_path: Path) -> None:
        """submission.csv deve ter exatamente 5 predições."""
        output = tmp_path / "submission.csv"
        with patch("demo_inference.OUTPUT_FILE", output):
            main(dry_run=True)
        with open(output, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 5

    def test_main_dry_run_all_ids_present(self, tmp_path: Path) -> None:
        """Todos os IDs do dataset sintético devem estar no CSV."""
        output = tmp_path / "submission.csv"
        with patch("demo_inference.OUTPUT_FILE", output):
            main(dry_run=True)
        with open(output, encoding="utf-8") as f:
            ids = {row["id"] for row in csv.DictReader(f)}
        assert "cielo_001" in ids
        assert "cielo_005" in ids

    def test_main_dry_run_no_pii_in_submission(self, tmp_path: Path) -> None:
        """submission.csv não deve conter padrões de CPF após scrub."""
        output = tmp_path / "submission.csv"
        with patch("demo_inference.OUTPUT_FILE", output):
            main(dry_run=True)
        content = output.read_text(encoding="utf-8")
        cpf_pattern = re.compile(r"\d{3}\.\d{3}\.\d{3}-\d{2}")
        assert not cpf_pattern.search(content), "PII detectado no submission.csv!"
