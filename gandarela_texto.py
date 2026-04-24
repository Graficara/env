"""
gandarela_texto.py — Interface com linguagem real
==================================================
Traduz texto em cadeias de blocos Gandarela.

Princípio: palavras raras carregam mais energia (mais informação).
Palavras comuns carregam menos energia — mas nunca zero.

Depende de: gandarela.py (deve estar na mesma pasta)
"""

import math
import re
from collections import Counter
from gandarela import (
    BlocoGandarela, CadeiaGandarela,
    compor, compor_cadeia, energia, EPSILON
)


# ─────────────────────────────────────────────
# Tokenização
# ─────────────────────────────────────────────

def tokenizar(texto: str) -> list[str]:
    """
    Quebra texto em tokens — palavras em minúsculo, sem pontuação.
    Preserva ordem original.
    """
    tokens = re.findall(r'\b[a-záéíóúâêîôûãõàüç]+\b', texto.lower())
    return tokens


# ─────────────────────────────────────────────
# Energia semântica
# ─────────────────────────────────────────────

# Palavras funcionais do português — alta frequência, baixa informação
PALAVRAS_FUNCIONAIS = {
    'a', 'o', 'e', 'é', 'de', 'do', 'da', 'dos', 'das', 'em', 'no',
    'na', 'nos', 'nas', 'um', 'uma', 'uns', 'umas', 'que', 'se', 'por',
    'para', 'com', 'como', 'mas', 'ou', 'nem', 'seu', 'sua', 'seus',
    'suas', 'ele', 'ela', 'eles', 'elas', 'eu', 'tu', 'nós', 'vós',
    'ao', 'à', 'pelo', 'pela', 'pelos', 'pelas', 'já', 'não', 'mais',
    'só', 'até', 'este', 'essa', 'isso', 'aqui', 'ali', 'lá', 'me',
    'te', 'nos', 'lhe', 'lhes', 'ser', 'ter', 'foi', 'são', 'está',
    'tem', 'há', 'vai', 'sobre', 'entre', 'quando', 'então', 'também'
}


def energia_semantica(token: str, frequencias: dict[str, float],
                       v_base: float = 2.0) -> float:
    """
    Calcula a energia de um token baseada na sua raridade no contexto.

    Tokens raros → alta energia (mais informação, v próximo de c)
    Tokens comuns → baixa energia (menos informação, v alto)

    Fórmula: v = v_base + frequência_relativa * escala
    Quanto mais frequente, maior v, menor E — mas nunca zero.
    """
    freq = frequencias.get(token, 0.0)
    
    # Palavras funcionais recebem frequência máxima artificialmente
    if token in PALAVRAS_FUNCIONAIS:
        freq = max(freq, 0.8)

    # v cresce com a frequência — palavras comuns propagam mais rápido
    # v mínimo é 1.01 (logo acima de c=1) para palavras raríssimas
    escala = 50.0
    v = 1.01 + freq * escala
    
    return energia(v)


def calcular_frequencias(tokens: list[str]) -> dict[str, float]:
    """Frequência relativa de cada token no texto."""
    contagem = Counter(tokens)
    total = len(tokens)
    return {tok: count / total for tok, count in contagem.items()}


# ─────────────────────────────────────────────
# Processador de texto
# ─────────────────────────────────────────────

class ProcessadorGandarela:
    """
    Converte texto em cadeia de blocos Gandarela.
    Cada token vira um bloco com energia proporcional à sua raridade.
    """

    def __init__(self, v_base: float = 2.0):
        self.v_base = v_base
        self.historico: list[dict] = []

    def processar(self, texto: str) -> CadeiaGandarela:
        """
        Processa um texto completo.
        Retorna CadeiaGandarela com um bloco por token.
        """
        tokens = tokenizar(texto)
        if not tokens:
            raise ValueError("Texto vazio ou sem tokens válidos")

        frequencias = calcular_frequencias(tokens)

        # Complexidade C = log do vocabulário único
        # Mais palavras distintas → contexto mais complexo
        vocab = len(set(tokens))
        C = max(math.log(vocab + 1) * 2, 0.5)

        cadeia = CadeiaGandarela(C=C, v=self.v_base)

        blocos_info = []
        for token in tokens:
            e_sem = energia_semantica(token, frequencias, self.v_base)
            # Garante que E está dentro do contexto C
            e_sem = min(e_sem, C - EPSILON)
            e_sem = max(e_sem, EPSILON)

            bloco = BlocoGandarela(
                energia_valor=e_sem,
                contexto=C,
                velocidade=self.v_base,
                rotulo=token
            )
            cadeia.adicionar_bloco(bloco)
            blocos_info.append({
                'token': token,
                'energia': round(e_sem, 6),
                'frequencia': round(frequencias[token], 4),
                'funcional': token in PALAVRAS_FUNCIONAIS
            })

        self.historico.append({
            'texto': texto,
            'tokens': tokens,
            'C': C,
            'blocos': blocos_info
        })

        return cadeia

    def relatorio(self, texto: str, top_n: int = 10) -> None:
        """Processa e imprime relatório completo."""
        print("=" * 56)
        print("  PROCESSADOR GANDARELA — análise de texto")
        print("=" * 56)
        print(f"\nTexto: \"{texto}\"")

        cadeia = self.processar(texto)
        ultimo = self.historico[-1]

        print(f"\nContexto C = {cadeia.C:.4f}  "
              f"(vocabulário: {len(set(ultimo['tokens']))} palavras únicas)")
        print(f"Tokens: {len(ultimo['tokens'])}")
        print(f"N teórico de blocos: {cadeia.n_teorico():.2f}")

        # Ordena por energia — mais raras primeiro
        ordenados = sorted(ultimo['blocos'],
                           key=lambda x: x['energia'], reverse=True)

        print(f"\n{'─'*56}")
        print(f"  {'TOKEN':<20} {'ENERGIA':>10}  {'FREQ':>6}  TIPO")
        print(f"{'─'*56}")
        for item in ordenados[:top_n]:
            tipo = "funcional" if item['funcional'] else "conteúdo"
            barra = "█" * int(item['energia'] / cadeia.C * 20)
            print(f"  {item['token']:<20} {item['energia']:>10.6f}  "
                  f"{item['frequencia']:>6.4f}  {tipo}")
            print(f"  {'':20} {barra}")

        # Composição da cadeia inteira
        print(f"\n{'─'*56}")
        bloco_final = cadeia.compor_tudo()
        print(f"  Energia composta (cadeia inteira): "
              f"{bloco_final.energia_valor:.6f}")
        print(f"  Energia relativa ao contexto:      "
              f"{bloco_final.energia_relativa():.4f}")
        print(f"  Invariante N·E = C:                "
              f"{cadeia.n_teorico() * cadeia.energia_teorica():.4f}")
        print(f"\n  ✓ Nenhum token foi anulado — energia residual preservada.")
        print("=" * 56)


# ─────────────────────────────────────────────
# Comparador de textos
# ─────────────────────────────────────────────

def comparar(texto1: str, texto2: str) -> None:
    """
    Compara dois textos pela sua energia Gandarela.
    Texto com mais conteúdo semântico → energia composta maior.
    """
    proc = ProcessadorGandarela()

    print("\n" + "=" * 56)
    print("  COMPARAÇÃO GANDARELA")
    print("=" * 56)

    resultados = []
    for txt in [texto1, texto2]:
        cadeia = proc.processar(txt)
        bloco = cadeia.compor_tudo()
        resultados.append({
            'texto': txt[:50] + ('...' if len(txt) > 50 else ''),
            'C': cadeia.C,
            'energia_composta': bloco.energia_valor,
            'energia_relativa': bloco.energia_relativa(),
            'n_tokens': len(proc.historico[-1]['tokens']),
            'vocab': len(set(proc.historico[-1]['tokens']))
        })

    for i, r in enumerate(resultados, 1):
        print(f"\n  Texto {i}: \"{r['texto']}\"")
        print(f"    Tokens: {r['n_tokens']}  |  Vocab único: {r['vocab']}")
        print(f"    Complexidade C:     {r['C']:.4f}")
        print(f"    Energia composta:   {r['energia_composta']:.6f}")
        print(f"    Energia relativa:   {r['energia_relativa']:.4f}")

    vencedor = max(resultados, key=lambda x: x['energia_relativa'])
    print(f"\n  → Maior densidade semântica: \"{vencedor['texto']}\"")
    print("=" * 56)


# ─────────────────────────────────────────────
# Demonstração
# ─────────────────────────────────────────────

if __name__ == "__main__":
    proc = ProcessadorGandarela()

    # Teste 1 — frase filosófica (seu território)
    proc.relatorio(
        "a granularidade da representação é determinada pelo contexto"
    )

    # Teste 2 — frase com palavras funcionais dominantes
    print()
    proc.relatorio(
        "e então ele foi até a casa e viu que ela estava lá"
    )

    # Teste 3 — comparação: conteúdo denso vs vazio
    comparar(
        "energia cinética quântica proporcional à massa e velocidade",
        "e o que é isso que está aqui e ali e lá"
    )
