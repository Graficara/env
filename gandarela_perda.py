"""
gandarela_perda.py — Função de perda da Lógica Gandarela
=========================================================
A função de perda mede o quanto o sistema errou e em que tipo de erro.

Três tipos de erro, tratados separadamente:
  1. Erro de energia    — distância entre E_previsto e E_esperado
  2. Erro de contexto   — ancoragem errada (C errado)
  3. Erro de ausência   — sistema retornou quase-zero quando não devia

Depende de: gandarela.py e gandarela_texto.py
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional
from gandarela import EPSILON, compor, energia as energia_fn


# ─────────────────────────────────────────────
# Estrutura de erro
# ─────────────────────────────────────────────

@dataclass
class ErroGandarela:
    """Resultado detalhado de uma avaliação de perda."""
    perda_total: float
    erro_energia: float       # distância entre E_prev e E_esp
    erro_contexto: float      # penalidade por C errado
    erro_ausencia: float      # penalidade por quase-zero indevido
    E_previsto: float
    E_esperado: float
    C_previsto: float
    C_esperado: float

    def resumo(self) -> str:
        linhas = [
            f"  Perda total:      {self.perda_total:.6f}",
            f"  Erro de energia:  {self.erro_energia:.6f}",
            f"  Erro de contexto: {self.erro_contexto:.6f}",
            f"  Erro de ausência: {self.erro_ausencia:.6f}",
            f"  E previsto:       {self.E_previsto:.6f}",
            f"  E esperado:       {self.E_esperado:.6f}",
            f"  C previsto:       {self.C_previsto:.4f}",
            f"  C esperado:       {self.C_esperado:.4f}",
        ]
        return "\n".join(linhas)


# ─────────────────────────────────────────────
# Função de perda principal
# ─────────────────────────────────────────────

def perda_gandarela(
    E_previsto: float,
    E_esperado: float,
    C_previsto: float,
    C_esperado: float,
    peso_energia: float = 1.0,
    peso_contexto: float = 0.5,
    peso_ausencia: float = 2.0,
    limiar_ausencia: float = 0.01,
) -> ErroGandarela:
    """
    Calcula a perda Gandarela entre previsão e esperado.

    Parâmetros:
        E_previsto     — energia que o sistema produziu
        E_esperado     — energia que deveria ter produzido
        C_previsto     — contexto usado pelo sistema
        C_esperado     — contexto correto
        peso_energia   — peso do erro de energia (padrão 1.0)
        peso_contexto  — peso do erro de contexto (padrão 0.5)
        peso_ausencia  — peso do erro de ausência (padrão 2.0, mais grave)
        limiar_ausencia — abaixo disso considera quase-zero indevido

    Retorna ErroGandarela com decomposição completa.
    """
    # Proteção — nunca opera com zero
    E_prev = max(E_previsto, EPSILON)
    E_esp  = max(E_esperado, EPSILON)
    C_prev = max(C_previsto, EPSILON)
    C_esp  = max(C_esperado, EPSILON)

    # ── Erro 1: energia ──────────────────────────────────────
    # Distância logarítmica — respeita a escala assintótica Gandarela.
    # log(E_prev/E_esp)² penaliza proporcionalmente, não absolutamente.
    # Exemplo: errar de 0.01 para 0.001 é tão grave quanto errar de 1.0 para 0.1
    erro_energia = (math.log(E_prev / E_esp)) ** 2

    # ── Erro 2: contexto ─────────────────────────────────────
    # Penaliza ancoragem errada.
    # Se C_prev != C_esp, a composição inteira foi feita na escala errada.
    erro_contexto = (math.log(C_prev / C_esp)) ** 2

    # ── Erro 3: ausência ─────────────────────────────────────
    # Penalidade extra quando o sistema retornou quase-zero
    # mas o esperado era alto — o pior erro Gandarela possível.
    # "Não saber" é diferente de "errar" — mas fingir ausência
    # quando há sinal é o erro mais grave.
    if E_prev < limiar_ausencia and E_esp >= limiar_ausencia:
        # Sistema colapsou para quase-zero quando não devia
        erro_ausencia = peso_ausencia * math.log(E_esp / E_prev)
    elif E_prev >= limiar_ausencia and E_esp < limiar_ausencia:
        # Sistema afirmou energia quando devia ser quase-zero
        erro_ausencia = peso_ausencia * math.log(E_prev / E_esp) * 0.5
    else:
        erro_ausencia = 0.0

    perda_total = (
        peso_energia  * erro_energia +
        peso_contexto * erro_contexto +
        peso_ausencia * erro_ausencia
    )

    return ErroGandarela(
        perda_total=perda_total,
        erro_energia=erro_energia,
        erro_contexto=erro_contexto,
        erro_ausencia=erro_ausencia,
        E_previsto=E_prev,
        E_esperado=E_esp,
        C_previsto=C_prev,
        C_esperado=C_esp,
    )


# ─────────────────────────────────────────────
# Perda sobre cadeia inteira
# ─────────────────────────────────────────────

def perda_cadeia(
    energias_previstas: list[float],
    energias_esperadas: list[float],
    C_previsto: float,
    C_esperado: float,
) -> dict:
    """
    Calcula perda média sobre uma cadeia de blocos.
    Cada bloco contribui com sua perda individual.
    A perda total é a média ponderada — blocos com maior E_esperado
    têm mais peso, porque errar em tokens importantes é pior.
    """
    if len(energias_previstas) != len(energias_esperadas):
        raise ValueError("Cadeias de tamanhos diferentes")

    perdas = []
    pesos = []
    for e_prev, e_esp in zip(energias_previstas, energias_esperadas):
        erro = perda_gandarela(e_prev, e_esp, C_previsto, C_esperado)
        perdas.append(erro.perda_total)
        pesos.append(max(e_esp, EPSILON))  # tokens importantes pesam mais

    soma_pesos = sum(pesos)
    perda_media = sum(p * w for p, w in zip(perdas, pesos)) / soma_pesos
    perda_max = max(perdas)
    token_critico = perdas.index(perda_max)

    return {
        "perda_media_ponderada": round(perda_media, 6),
        "perda_maxima": round(perda_max, 6),
        "token_critico_idx": token_critico,
        "n_blocos": len(perdas),
        "perdas_individuais": [round(p, 6) for p in perdas],
    }


# ─────────────────────────────────────────────
# Gradiente Gandarela (direção de correção)
# ─────────────────────────────────────────────

def gradiente_energia(
    E_previsto: float,
    E_esperado: float,
    C: float,
) -> float:
    """
    Direção e magnitude da correção necessária em E_previsto.

    Derivada da perda logarítmica em relação a E_previsto:
    dL/dE = 2 * log(E_prev/E_esp) / E_prev

    Positivo → E_prev está acima do esperado, deve diminuir.
    Negativo → E_prev está abaixo do esperado, deve aumentar.
    """
    E_prev = max(E_previsto, EPSILON)
    E_esp  = max(E_esperado, EPSILON)
    return 2.0 * math.log(E_prev / E_esp) / E_prev


def passo_correcao(
    E_previsto: float,
    E_esperado: float,
    C: float,
    taxa: float = 0.1,
) -> float:
    """
    Aplica um passo de correção em E_previsto na direção do esperado.
    Garante que o resultado nunca ultrapassa C e nunca chega a zero.
    """
    grad = gradiente_energia(E_previsto, E_esperado, C)
    E_novo = E_previsto - taxa * grad
    E_novo = max(E_novo, EPSILON)
    E_novo = min(E_novo, C - EPSILON)
    return E_novo


def convergir(
    E_inicial: float,
    E_alvo: float,
    C: float,
    taxa: float = 0.1,
    max_passos: int = 100,
    tolerancia: float = 1e-6,
) -> list[float]:
    """
    Simula o processo de aprendizado — passos sucessivos de correção
    até convergir para E_alvo (ou atingir max_passos).
    Retorna histórico de E a cada passo.
    """
    historico = [E_inicial]
    E = E_inicial
    for _ in range(max_passos):
        E_novo = passo_correcao(E, E_alvo, C, taxa)
        historico.append(E_novo)
        if abs(E_novo - E) < tolerancia:
            break
        E = E_novo
    return historico


# ─────────────────────────────────────────────
# Demonstração
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 56)
    print("  FUNÇÃO DE PERDA GANDARELA — demonstração")
    print("=" * 56)

    # ── Cenário 1: erro pequeno de energia ───────────────────
    print("\n[1] Erro pequeno de energia (C correto)")
    erro = perda_gandarela(
        E_previsto=0.12, E_esperado=0.14,
        C_previsto=3.0,  C_esperado=3.0
    )
    print(erro.resumo())

    # ── Cenário 2: erro grande de energia ────────────────────
    print("\n[2] Erro grande de energia (C correto)")
    erro = perda_gandarela(
        E_previsto=0.05, E_esperado=0.80,
        C_previsto=3.0,  C_esperado=3.0
    )
    print(erro.resumo())

    # ── Cenário 3: erro de contexto ──────────────────────────
    print("\n[3] Energia correta, contexto errado")
    erro = perda_gandarela(
        E_previsto=0.50, E_esperado=0.50,
        C_previsto=2.0,  C_esperado=5.0
    )
    print(erro.resumo())

    # ── Cenário 4: erro de ausência (o mais grave) ───────────
    print("\n[4] Erro de ausência — sistema colapsou para quase-zero")
    erro = perda_gandarela(
        E_previsto=0.001, E_esperado=0.80,
        C_previsto=3.0,   C_esperado=3.0
    )
    print(erro.resumo())

    # ── Cenário 5: convergência ──────────────────────────────
    print("\n[5] Convergência — aprendizado em passos")
    historico = convergir(
        E_inicial=0.05, E_alvo=0.50,
        C=3.0, taxa=0.15
    )
    print(f"  E inicial: {historico[0]:.6f}")
    print(f"  E alvo:    0.500000")
    print(f"  Passos até convergir: {len(historico)}")
    print(f"  E final:   {historico[-1]:.6f}")
    print(f"  Trajetória (primeiros 8 passos):")
    for i, e in enumerate(historico[:8]):
        barra = "█" * int(e / 0.5 * 30)
        print(f"    passo {i:02d}: {e:.6f}  {barra}")

    # ── Cenário 6: perda sobre cadeia ────────────────────────
    print("\n[6] Perda sobre cadeia inteira")
    # Simula previsão de energias para uma frase
    esperado  = [0.14, 0.14, 0.14, 0.02, 0.02, 0.14, 0.02]
    previsto  = [0.10, 0.13, 0.14, 0.02, 0.08, 0.01, 0.02]
    resultado = perda_cadeia(previsto, esperado, C_previsto=4.0, C_esperado=4.0)
    for k, v in resultado.items():
        print(f"  {k}: {v}")

    print(f"\n  ✓ Sistema nunca operou com zero em nenhum cálculo.")
    print("=" * 56)
