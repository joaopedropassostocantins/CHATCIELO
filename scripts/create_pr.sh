#!/usr/bin/env bash
# scripts/create_pr.sh
#
# Cria um Pull Request via gh CLI com validação de título.
# Erro corrigido: "Pull request creation failed. Validation failed: Title can't be blank"
#
# USO:
#   ./scripts/create_pr.sh [--title "Título"] [--base main] [--body "Corpo"]
#
# Se --title não for passado, usa o último commit como título (com fallback).

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
TITLE=""
BASE="main"
BODY=""
BRANCH=""

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --title)  TITLE="$2";  shift 2 ;;
        --base)   BASE="$2";   shift 2 ;;
        --body)   BODY="$2";   shift 2 ;;
        --branch) BRANCH="$2"; shift 2 ;;
        *) echo "Argumento desconhecido: $1" >&2; exit 1 ;;
    esac
done

# ── Detectar branch atual se não fornecido ────────────────────────────────────
if [[ -z "$BRANCH" ]]; then
    BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")"
fi

if [[ -z "$BRANCH" || "$BRANCH" == "HEAD" ]]; then
    echo "ERRO: Não foi possível determinar a branch atual." >&2
    exit 1
fi

# ── CORREÇÃO: Garantir que o título nunca seja vazio ─────────────────────────
# Estratégia:
#   1. Usa --title se fornecido e não-vazio
#   2. Tenta usar a mensagem do último commit (primeira linha)
#   3. Fallback: "chore: update" (nunca ficará em branco)

if [[ -z "$TITLE" ]]; then
    # Tenta extrair título do último commit
    TITLE="$(git log -1 --pretty=format='%s' 2>/dev/null || echo "")"
fi

# Strip espaços em branco
TITLE="$(echo "$TITLE" | xargs)"

# Fallback final — garante que o campo nunca chegue em branco na chamada da API
if [[ -z "$TITLE" ]]; then
    TITLE="chore: update"
    echo "AVISO: Título estava vazio, usando fallback: '${TITLE}'" >&2
fi

# ── Truncar título para 70 chars (boas práticas de PR) ───────────────────────
if [[ ${#TITLE} -gt 70 ]]; then
    TITLE="${TITLE:0:67}..."
fi

# ── Body padrão se não fornecido ──────────────────────────────────────────────
if [[ -z "$BODY" ]]; then
    BODY="$(cat <<EOF
## Resumo

- Branch: \`${BRANCH}\`
- Base: \`${BASE}\`

## Checklist

- [ ] Testes passando (\`pytest tests/ -v\`)
- [ ] Lint OK (\`ruff check . && black --check .\`)
- [ ] Sem PII em logs/outputs
EOF
)"
fi

# ── Criação do PR ─────────────────────────────────────────────────────────────
echo "Criando PR:"
echo "  Branch : ${BRANCH}"
echo "  Base   : ${BASE}"
echo "  Título : ${TITLE}"
echo ""

gh pr create \
    --title "${TITLE}" \
    --base "${BASE}" \
    --head "${BRANCH}" \
    --body "${BODY}"
