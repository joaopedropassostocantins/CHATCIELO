#!/usr/bin/env python3
"""
demo_inference.py — Cielo Adaptive Core | Kaggle NVIDIA Reasoning Challenge
============================================================================
Modelo : nvidia/Nemotron-3-Nano-30B-A3B-BF16
Ambiente: Kaggle Notebook GPU T4 x2
Objetivo: Demonstrar inferência com thinking mode (CoT), roteamento de intents,
          mascaramento LGPD e geração de submission.csv.

Arquitetura de segurança ativada neste script:
  - LGPD: regex scrub de CPF/cartão ANTES de qualquer chamada ao modelo
  - Faker: dados sintéticos — sem PII real em nenhum fixture
  - Prometheus: contador de inferências e eventos de mascaramento
  - BitsAndBytes NF4: quantização 4-bit — reduz VRAM de ~60 GB para ~18 GB

Formato de saída: submission.csv  (id, prediction) — padrão Kaggle.

Uso:
    python demo_inference.py                # execução completa com GPU
    python demo_inference.py --dry-run      # smoke test sem carregar o modelo
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Logging estruturado — sem PII nos logs (LGPD rule #6 do CLAUDE.md)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("cielo.demo")

# ---------------------------------------------------------------------------
# Prometheus — métricas de observabilidade
# Ativado quando prometheus-client está disponível (prod/Kaggle com internet)
# ---------------------------------------------------------------------------
try:
    from prometheus_client import Counter, Histogram, start_http_server  # type: ignore[import]

    # Contador total de chamadas de inferência com labels de intent e segmento
    INFERENCE_COUNTER = Counter(
        "chatcielo_demo_inferences_total",
        "Total de inferências no demo Kaggle",
        ["intent", "segment"],
    )
    # Histograma de latência (alinhado com SLA P99 < 300ms do CLAUDE.md §3.3)
    LATENCY_HISTOGRAM = Histogram(
        "chatcielo_demo_latency_seconds",
        "Latência de inferência (segundos)",
        buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0],
    )
    # Eventos de mascaramento LGPD — auditoria de PII detectado e removido
    PII_SCRUB_COUNTER = Counter(
        "chatcielo_pii_scrubbed_total",
        "Total de eventos de mascaramento PII (LGPD)",
        ["pii_type"],
    )
    PROMETHEUS_AVAILABLE = True
    logger.info("Prometheus client carregado — métricas ativas.")
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus-client não encontrado — métricas desativadas.")

# ---------------------------------------------------------------------------
# Faker — dados sintéticos LGPD-safe
# Nenhum CPF, cartão ou nome real é usado em qualquer fixture (CLAUDE.md rule #3)
# ---------------------------------------------------------------------------
try:
    from faker import Faker  # type: ignore[import]

    _fake = Faker("pt_BR")
    FAKER_AVAILABLE = True
except ImportError:
    FAKER_AVAILABLE = False
    logger.warning("faker não encontrado — usando fixtures estáticas sintéticas.")

# ---------------------------------------------------------------------------
# Constantes & configuração
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).parent / "config" / "config.yaml"
MODEL_ID = "nvidia/Nemotron-3-Nano-30B-A3B-BF16"
OUTPUT_FILE = Path("submission.csv")

# Padrões LGPD de PII — definidos também em config.yaml §security.pii_masking
_PII_PATTERNS: dict[str, tuple[str, str]] = {
    "cpf": (
        r"(\d{3}\.?\d{3}\.?\d{3}-?\d{2})",
        "[CPF_REMOVIDO]",
    ),
    "card": (
        r"(\b(?:\d[ -]?){13,19}\b)",
        "[CARTAO_REMOVIDO]",
    ),
    "email": (
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
        "[EMAIL_REMOVIDO]",
    ),
    "phone": (
        r"(\(?\d{2}\)?\s?\d{4,5}-?\d{4})",
        "[TELEFONE_REMOVIDO]",
    ),
}


# ---------------------------------------------------------------------------
# LGPD — mascaramento de PII
# Aplicado a TODO texto antes de chegar ao modelo (linha de defesa #1)
# ---------------------------------------------------------------------------
def scrub_pii(text: str) -> str:
    """Remove PII do texto antes de enviar ao modelo (LGPD Art. 46).

    Args:
        text: String de entrada que pode conter CPF, cartão, email ou telefone.

    Returns:
        String sanitizada com PII substituída por placeholders.

    Raises:
        TypeError: Se text não for uma string.

    Validation Metrics:
        - Zero ocorrências de padrões PII na saída (garantido pelos asserts de teste).
    """
    if not isinstance(text, str):
        raise TypeError(f"scrub_pii esperava str, recebeu {type(text)}")

    for pii_type, (pattern, replacement) in _PII_PATTERNS.items():
        original = text
        text = re.sub(pattern, replacement, text)
        if text != original and PROMETHEUS_AVAILABLE:
            # Incrementa contador de auditoria LGPD
            PII_SCRUB_COUNTER.labels(pii_type=pii_type).inc()  # type: ignore[union-attr]

    return text


# ---------------------------------------------------------------------------
# Roteamento de intents
# Lógica espelhada de config.yaml §intent_routing
# ---------------------------------------------------------------------------
_INTENT_RULES: dict[str, str] = {
    "taxa": "reclamacao_taxa",
    "caro": "reclamacao_taxa",
    "mdr": "duvida_taxa_mdr",
    "parcel": "duvida_parcelamento",
    "maquininha": "problema_terminal",
    "terminal": "problema_terminal",
    "cancelar": "cancelamento",
    "cancelamento": "cancelamento",
}


def detect_intent(message: str) -> str:
    """Detecta o intent do lojista usando correspondência por palavras-chave.

    Args:
        message: Mensagem do lojista (já sanitizada de PII).

    Returns:
        String com o intent detectado (ex: 'reclamacao_taxa').
    """
    lower = message.lower()
    for keyword, intent in _INTENT_RULES.items():
        if keyword in lower:
            return intent
    return "outros"


# ---------------------------------------------------------------------------
# Dataset sintético — 5 interações simuladas de lojistas Cielo
# Faker garante dados realistas sem PII real (CLAUDE.md rule #3)
# ---------------------------------------------------------------------------
@dataclass
class MerchantInteraction:
    """Representa uma interação simulada de lojista com o suporte Cielo.

    Attributes:
        id: Identificador único da interação.
        segment: Segmento do lojista (MEI, VAREJO, CORPORATE).
        message: Mensagem do lojista (pré-sanitizada).
        expected_intent: Intent esperado para validação.
        context: Contexto adicional da interação.
    """

    id: str
    segment: str
    message: str
    expected_intent: str
    context: dict[str, Any] = field(default_factory=dict)


def build_synthetic_dataset() -> list[MerchantInteraction]:
    """Constrói dataset sintético de 5 interações de lojistas Cielo.

    Usa Faker para gerar nomes de estabelecimentos fictícios.
    Nenhum CPF, CNPJ real ou dado pessoal é incluído.

    Returns:
        Lista de 5 MerchantInteraction com dados sintéticos.
    """
    # Nomes sintéticos de estabelecimentos — sem PII real
    if FAKER_AVAILABLE:
        company_a = _fake.company()
        company_b = _fake.company()
    else:
        company_a = "Loja Sintética A"
        company_b = "Empresa Fictícia B"

    return [
        MerchantInteraction(
            id="cielo_001",
            segment="MEI",
            message=(
                "Olá, minha taxa está muito alta. "
                f"Sou dono do {company_a} e estou pagando mais de 3% no débito. "
                "Tem como reduzir ou tem alguma promoção?"
            ),
            expected_intent="reclamacao_taxa",
            context={"monthly_revenue_brl": 15_000, "terminal_count": 1},
        ),
        MerchantInteraction(
            id="cielo_002",
            segment="VAREJO",
            message=(
                "Não estou entendendo o extrato. "
                "A MDR que aparece é diferente do que foi acordado no contrato. "
                "Podem explicar como é calculado?"
            ),
            expected_intent="duvida_taxa_mdr",
            context={"monthly_revenue_brl": 80_000, "terminal_count": 3},
        ),
        MerchantInteraction(
            id="cielo_003",
            segment="MEI",
            message=(
                "Quero saber como funciona o parcelamento. "
                "Se o cliente parcelar em 10x, quando eu recebo? "
                "E qual é a taxa de parcelamento?"
            ),
            expected_intent="duvida_parcelamento",
            context={"monthly_revenue_brl": 22_000, "terminal_count": 1},
        ),
        MerchantInteraction(
            id="cielo_004",
            segment="CORPORATE",
            message=(
                f"A maquininha da unidade do {company_b} está apresentando erro E05 "
                "desde ontem às 14h. Já reiniciamos mas o problema persiste. "
                "Precisamos de suporte técnico urgente."
            ),
            expected_intent="problema_terminal",
            context={"monthly_revenue_brl": 500_000, "terminal_count": 12},
        ),
        MerchantInteraction(
            id="cielo_005",
            segment="VAREJO",
            message=(
                "Estou pensando em cancelar o contrato com a Cielo. "
                "Recebi uma proposta de um concorrente com taxa menor. "
                "Vocês conseguem fazer uma contraproposta?"
            ),
            expected_intent="cancelamento",
            context={"monthly_revenue_brl": 95_000, "terminal_count": 4},
        ),
    ]


# ---------------------------------------------------------------------------
# Prompt builder — Thinking Mode (Chain of Thought)
# O modelo raciocina entre <think>...</think> antes de formular a resposta final.
# Isso é central para o NVIDIA Reasoning Challenge (LMSYS SOTA).
# ---------------------------------------------------------------------------
def build_cot_prompt(interaction: MerchantInteraction, intent: str) -> str:
    """Constrói o prompt com Chain-of-Thought para o modelo Nemotron.

    O template forçar o modelo a:
      1. Pensar sobre o problema dentro de <think>...</think>
      2. Decidir a ação com base no intent roteado
      3. Formular uma resposta empática e orientada a solução

    Args:
        interaction: Dados da interação do lojista.
        intent: Intent detectado pelo roteador.

    Returns:
        String com o prompt completo no formato CoT esperado pelo Nemotron.
    """
    segment_tone = {
        "MEI": "simples, direto e acolhedor",
        "VAREJO": "consultivo e profissional",
        "CORPORATE": "técnico, executivo e preciso",
    }.get(interaction.segment, "profissional")

    action_hint = {
        "reclamacao_taxa": "ofereça uma simulação de antecipação de recebíveis",
        "duvida_taxa_mdr": "explique detalhadamente a composição do MDR",
        "duvida_parcelamento": "explique os prazos de recebimento e taxas de parcelamento",
        "problema_terminal": "abra um chamado técnico e forneça ETA de resolução",
        "cancelamento": "ative o script de retenção, destaque benefícios exclusivos",
        "outros": "direcione ao atendimento geral com empatia",
    }.get(intent, "responda com empatia")

    return (
        f"<|system|>\n"
        f"Você é o assistente de suporte da Cielo Brasil. "
        f"Responda em português, com tom {segment_tone}. "
        f"Segmento do lojista: {interaction.segment}.\n"
        f"<|user|>\n"
        f"{interaction.message}\n"
        f"<|assistant|>\n"
        f"<think>\n"
        f"Vou analisar a mensagem do lojista passo a passo:\n"
        f"1. Intent detectado: {intent}\n"
        f"2. Segmento: {interaction.segment} — tom adequado: {segment_tone}\n"
        f"3. Ação recomendada pelo roteador: {action_hint}\n"
        f"4. Devo ser empático, resolver o problema e {action_hint}.\n"
        f"</think>\n"
    )


# ---------------------------------------------------------------------------
# Carregamento do modelo — BitsAndBytes 4-bit NF4
# Reduz VRAM: ~60 GB (BF16 full) → ~18 GB (NF4 4-bit) — cabe em T4 x2 (32 GB)
# ---------------------------------------------------------------------------
def load_model_4bit(
    model_id: str,
    dry_run: bool = False,
) -> tuple[Any, Any]:
    """Carrega o modelo Nemotron em quantização 4-bit NF4 via BitsAndBytes.

    Segurança de memória:
      - NF4 (NormalFloat4): melhor fidelidade estatística vs INT4 simples
      - double_quant=True: segunda quantização das constantes de escala (~0.4 bit/param extra)
      - compute_dtype=bfloat16: operações de atenção em BF16 (estabilidade numérica)
      - device_map="auto": distribui automaticamente entre as 2 GPUs T4

    Args:
        model_id: Identificador HuggingFace do modelo.
        dry_run: Se True, retorna mocks sem carregar pesos (CI/smoke test).

    Returns:
        Tupla (model, tokenizer) prontos para inferência.

    Raises:
        ImportError: Se transformers ou bitsandbytes não estiverem instalados.
        RuntimeError: Se nenhuma GPU CUDA estiver disponível (em modo não-dry-run).
    """
    if dry_run:
        logger.info("[DRY-RUN] Pulando carregamento do modelo — retornando mocks.")

        class _MockTokenizer:
            def __call__(self, text: str, **kw: Any) -> dict[str, Any]:
                return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}

            def decode(self, ids: Any, **kw: Any) -> str:
                return "<think>\nAnálise mock.\n</think>\nResposta sintética de teste."

        class _MockModel:
            def generate(self, **kw: Any) -> list[list[int]]:
                return [[1, 2, 3, 4, 5]]

        return _MockModel(), _MockTokenizer()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError(
            "Instale: pip install transformers bitsandbytes accelerate"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU CUDA não detectada. Este script requer T4 x2 (Kaggle P100/T4)."
        )

    logger.info("Configurando BitsAndBytes NF4 4-bit...")
    # Configuração de quantização — espelha config.yaml §model.quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",        # NormalFloat4 — melhor fidelidade
        bnb_4bit_use_double_quant=True,    # double quantization ~0.4 bit/param
        bnb_4bit_compute_dtype=torch.bfloat16,  # compute em BF16
    )

    logger.info("Carregando tokenizer de %s ...", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    logger.info("Carregando modelo %s em 4-bit NF4 (pode levar ~5 min) ...", model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",          # distribui automaticamente entre T4 x2
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    gpu_mem = sum(
        p.element_size() * p.nelement() for p in model.parameters()
    ) / 1024**3
    logger.info("Modelo carregado. Footprint estimado: %.2f GB", gpu_mem)

    return model, tokenizer


# ---------------------------------------------------------------------------
# Inferência com thinking mode
# ---------------------------------------------------------------------------
def run_inference(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 512,
    dry_run: bool = False,
) -> tuple[str, str, float]:
    """Executa inferência com Chain-of-Thought (thinking mode).

    O modelo gera texto após o bloco <think>...</think> do prompt,
    completando o raciocínio e formulando a resposta final.

    Args:
        model: Modelo carregado (real ou mock).
        tokenizer: Tokenizador correspondente.
        prompt: Prompt CoT completo com <think> aberto.
        max_new_tokens: Limite de tokens gerados.
        dry_run: Se True, usa mock sem GPU.

    Returns:
        Tupla (thinking_content, final_response, latency_ms).

    Raises:
        RuntimeError: Se a geração falhar inesperadamente.
    """
    t0 = time.perf_counter()

    if dry_run:
        raw_output = tokenizer.decode([], skip_special_tokens=True)
        thinking = "Análise mock do intent. Ação: resposta sintética."
        response = "Olá! Entendemos sua situação e vamos ajudá-lo. [RESPOSTA SINTÉTICA]"
        latency_ms = (time.perf_counter() - t0) * 1000
        return thinking, response, latency_ms

    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch não disponível") from exc

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    latency_ms = (time.perf_counter() - t0) * 1000

    # Decodifica apenas os tokens NOVOS (não o prompt)
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(new_ids, skip_special_tokens=True)

    # Extrai thinking e resposta do output
    thinking = _extract_between(raw_output, "<think>", "</think>")
    response = _extract_after(raw_output, "</think>").strip()
    if not response:
        response = raw_output.strip()

    return thinking, response, latency_ms


def _extract_between(text: str, start: str, end: str) -> str:
    """Extrai conteúdo entre dois marcadores.

    Args:
        text: Texto completo de busca.
        start: Marcador de início.
        end: Marcador de fim.

    Returns:
        Conteúdo entre os marcadores, ou string vazia se não encontrado.
    """
    pattern = re.compile(
        re.escape(start) + r"(.*?)" + re.escape(end), re.DOTALL
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def _extract_after(text: str, marker: str) -> str:
    """Extrai tudo após um marcador.

    Args:
        text: Texto completo.
        marker: Marcador de corte.

    Returns:
        Texto após o marcador, ou string vazia se não encontrado.
    """
    idx = text.find(marker)
    return text[idx + len(marker):] if idx != -1 else ""


# ---------------------------------------------------------------------------
# Salvar submission.csv — formato padrão Kaggle (id, prediction)
# ---------------------------------------------------------------------------
def save_submission(
    results: list[dict[str, str]],
    output_path: Path = OUTPUT_FILE,
) -> None:
    """Salva os resultados no formato submission.csv do Kaggle.

    Formato: duas colunas — id (string) e prediction (texto da resposta).
    A prediction é truncada a 500 chars para compatibilidade com limites Kaggle.

    Args:
        results: Lista de dicts com chaves 'id' e 'prediction'.
        output_path: Caminho do arquivo de saída.

    Raises:
        OSError: Se não for possível escrever no caminho especificado.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "prediction"])
        writer.writeheader()
        for row in results:
            # Trunca prediction para evitar problemas com limites de upload Kaggle
            writer.writerow(
                {"id": row["id"], "prediction": row["prediction"][:500]}
            )
    logger.info("submission.csv salvo em: %s (%d linhas)", output_path, len(results))


# ---------------------------------------------------------------------------
# Carregar config.yaml
# ---------------------------------------------------------------------------
def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Carrega a configuração do projeto a partir do config.yaml.

    Args:
        config_path: Caminho para o arquivo YAML.

    Returns:
        Dicionário com a configuração completa.

    Raises:
        FileNotFoundError: Se o config.yaml não for encontrado.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml não encontrado em: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Main — pipeline completo
# ---------------------------------------------------------------------------
def main(dry_run: bool = False) -> None:
    """Pipeline principal do demo Cielo Adaptive Core.

    Fluxo:
      1. Carrega config.yaml (intent routing, segurança, KPIs)
      2. Carrega modelo Nemotron em 4-bit NF4 (ou mock em dry-run)
      3. Itera sobre 5 interações sintéticas de lojistas
      4. Aplica mascaramento LGPD (scrub_pii) em cada mensagem
      5. Detecta intent e constrói prompt CoT
      6. Executa inferência com thinking mode
      7. Exibe thinking + resposta final no console
      8. Salva submission.csv no formato Kaggle

    Args:
        dry_run: Se True, executa sem GPU para smoke tests.
    """
    logger.info("=" * 70)
    logger.info("CIELO ADAPTIVE CORE — Kaggle NVIDIA Reasoning Challenge")
    logger.info("Modelo: %s", MODEL_ID)
    logger.info("Modo: %s", "DRY-RUN (mock)" if dry_run else "GPU T4 x2 (4-bit NF4)")
    logger.info("=" * 70)

    # 1. Configuração
    config = load_config()
    logger.info("config.yaml carregado. %d regras de intent.", len(config["intent_routing"]["rules"]))

    # 2. Iniciar servidor Prometheus (prod/Kaggle com porta disponível)
    if PROMETHEUS_AVAILABLE and not dry_run:
        prom_port = config.get("monitoring", {}).get("prometheus", {}).get("port", 9090)
        try:
            start_http_server(prom_port)  # type: ignore[union-attr]
            logger.info("Prometheus HTTP server iniciado na porta %d", prom_port)
        except OSError:
            logger.warning("Porta %d ocupada — Prometheus não iniciado.", prom_port)

    # 3. Carregar modelo
    model, tokenizer = load_model_4bit(MODEL_ID, dry_run=dry_run)

    # 4. Dataset sintético
    interactions = build_synthetic_dataset()
    logger.info("Dataset sintético: %d interações (sem PII real).", len(interactions))

    submission_rows: list[dict[str, str]] = []

    # 5. Loop de inferência
    for i, interaction in enumerate(interactions, start=1):
        logger.info("\n%s", "─" * 60)
        logger.info("[%d/%d] ID: %s | Segmento: %s", i, len(interactions), interaction.id, interaction.segment)

        # LGPD: mascarar PII ANTES de qualquer processamento
        safe_message = scrub_pii(interaction.message)
        logger.info("Mensagem (sanitizada): %s", safe_message)

        # Detectar intent
        intent = detect_intent(safe_message)
        logger.info("Intent detectado: %s (esperado: %s)", intent, interaction.expected_intent)

        # Construir prompt CoT
        prompt = build_cot_prompt(
            MerchantInteraction(
                id=interaction.id,
                segment=interaction.segment,
                message=safe_message,
                expected_intent=interaction.expected_intent,
                context=interaction.context,
            ),
            intent,
        )

        # Inferência
        thinking, response, latency_ms = run_inference(
            model, tokenizer, prompt, dry_run=dry_run
        )

        # Registrar métricas Prometheus
        if PROMETHEUS_AVAILABLE:
            INFERENCE_COUNTER.labels(intent=intent, segment=interaction.segment).inc()  # type: ignore[union-attr]
            LATENCY_HISTOGRAM.observe(latency_ms / 1000.0)  # type: ignore[union-attr]

        # Exibir resultado no console (formato legível para Kaggle notebook)
        print(f"\n{'═'*60}")
        print(f"  ID: {interaction.id} | Segmento: {interaction.segment}")
        print(f"  Intent: {intent} | Latência: {latency_ms:.1f}ms")
        print(f"{'─'*60}")
        print(f"  [THINKING MODE — Cadeia de Pensamento]")
        print(f"  {thinking[:300]}{'...' if len(thinking) > 300 else ''}")
        print(f"{'─'*60}")
        print(f"  [RESPOSTA FINAL]")
        print(f"  {response[:400]}{'...' if len(response) > 400 else ''}")
        print(f"{'═'*60}\n")

        # Verificar SLA P99 (CLAUDE.md §3.3: P99 < 300ms)
        sla_ms = (
            config.get("monitoring", {})
            .get("kpis", {})
            .get("latency", {})
            .get("p99_sla_ms", 300)
        )
        if latency_ms > sla_ms:
            logger.warning(
                "ALERTA SLA: latência %.1fms > %.0fms (P99 SLA). Otimização necessária.",
                latency_ms, sla_ms,
            )

        submission_rows.append({"id": interaction.id, "prediction": response})

    # 6. Salvar submission.csv
    save_submission(submission_rows, OUTPUT_FILE)

    logger.info("\n✓ Pipeline concluído. %d predições salvas em submission.csv", len(submission_rows))
    logger.info("  → Faça upload de submission.csv na página de submit do Kaggle.")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cielo Adaptive Core — Kaggle NVIDIA Reasoning Challenge"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Executa sem carregar o modelo (smoke test / CI).",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
