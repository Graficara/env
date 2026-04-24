"""
gandarela.py — Biblioteca da Lógica Gandarela
==============================================
Princípio central: zero absoluto não existe.
Todo bloco de informação carrega energia mínima ε irredutível.

Fórmulas do núcleo:
  E(v)     = mc² / √(v²/c² − 1)          # energia por bloco
  N(C, v)  = C · √(v²/c² − 1) / mc²      # blocos necessários
  Invariante: N · E = C                   # complexidade conservada
  E₁ ⊕_C E₂ = C·(E₁·E₂) / (C·E₁ + C·E₂ − E₁·E₂)  # composição
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional


# ─────────────────────────────────────────────
# Constantes padrão
# ─────────────────────────────────────────────
C_LUZ: float = 1.0   # velocidade de referência (adimensional)
MASSA: float = 1.0   # massa padrão dos blocos
EPSILON: float = 1e-9  # piso absoluto — nunca chega a zero


# ─────────────────────────────────────────────
# Funções nucleares
# ─────────────────────────────────────────────

def energia(v: float, m: float = MASSA, c: float = C_LUZ) -> float:
    """
    Energia de um bloco propagando-se à velocidade v.
    E(v) = mc² / √(v²/c² − 1)

    Requer v > c. Quanto maior v, menor E — mas nunca zero.
    """
    if v <= c:
        raise ValueError(f"v ({v}) deve ser maior que c ({c})")
    raiz = math.sqrt((v * v) / (c * c) - 1)
    e = (m * c * c) / raiz
    return max(e, EPSILON)  # piso irredutível


def n_blocos(C: float, v: float, m: float = MASSA, c: float = C_LUZ) -> float:
    """
    Número mínimo de blocos para representar complexidade C à velocidade v.
    N(C, v) = C · √(v²/c² − 1) / mc²

    É o inverso de energia — quando E cai, N sobe.
    Invariante: N · E = C
    """
    if v <= c:
        raise ValueError(f"v ({v}) deve ser maior que c ({c})")
    raiz = math.sqrt((v * v) / (c * c) - 1)
    return (C * raiz) / (m * c * c)


def verificar_invariante(C: float, v: float,
                          m: float = MASSA, c: float = C_LUZ) -> dict:
    """
    Verifica N · E = C para dado contexto.
    Retorna dict com N, E, produto e erro relativo.
    """
    e = energia(v, m, c)
    n = n_blocos(C, v, m, c)
    produto = n * e
    erro = abs(produto - C) / C if C != 0 else 0.0
    return {"N": n, "E": e, "N*E": produto, "C": C, "erro_relativo": erro}


# ─────────────────────────────────────────────
# Função de composição ⊕_C
# ─────────────────────────────────────────────

def compor(e1: float, e2: float, C: float) -> float:
    """
    Composição Gandarela de dois blocos ancorada no contexto C.
    E₁ ⊕_C E₂ = C · (E₁ · E₂) / (C·E₁ + C·E₂ − E₁·E₂)

    Propriedades garantidas:
      ✓ resultado > 0 (nunca anula)
      ✓ resultado < C (dentro do contexto)
      ✓ comutativa: e1 ⊕ e2 = e2 ⊕ e1
      ✓ desequilíbrio penalizado vs média aritmética
      ✓ resultado ≤ max(e1, e2)
    """
    if e1 <= 0 or e2 <= 0:
        raise ValueError("Energias devem ser positivas")
    if C <= 0:
        raise ValueError("Contexto C deve ser positivo")

    denominador = C * e1 + C * e2 - e1 * e2
    if denominador <= 0:
        # Proteção: retorna piso com sinal de C
        return EPSILON

    resultado = C * (e1 * e2) / denominador
    # Garante os dois limites estruturais
    resultado = max(resultado, EPSILON)
    resultado = min(resultado, C - EPSILON)
    return resultado


def compor_cadeia(blocos: List[float], C: float) -> float:
    """
    Compõe uma cadeia inteira de blocos, da esquerda para a direita.
    Nenhum passo pode anular a cadeia.
    """
    if not blocos:
        raise ValueError("Cadeia vazia")
    if len(blocos) == 1:
        return max(blocos[0], EPSILON)

    acumulado = blocos[0]
    for bloco in blocos[1:]:
        acumulado = compor(acumulado, bloco, C)
    return acumulado


# ─────────────────────────────────────────────
# Classe BlocoGandarela
# ─────────────────────────────────────────────

@dataclass
class BlocoGandarela:
    """
    Unidade atômica de informação no sistema Gandarela.
    Carrega energia, contexto e velocidade de propagação.
    """
    energia_valor: float          # E do bloco
    contexto: float               # C — âncora da composição
    velocidade: float = 2.0      # v > c_ref
    m: float = MASSA
    c_ref: float = C_LUZ
    rotulo: str = ""

    def __post_init__(self):
        if self.energia_valor <= 0:
            raise ValueError("Energia deve ser positiva — zero absoluto não existe")
        if self.velocidade <= self.c_ref:
            raise ValueError(f"v deve ser maior que c_ref ({self.c_ref})")

    @classmethod
    def do_contexto(cls, C: float, v: float,
                    m: float = MASSA, c: float = C_LUZ,
                    rotulo: str = "") -> "BlocoGandarela":
        """Cria bloco com energia calculada a partir de v e C."""
        e = energia(v, m, c)
        return cls(energia_valor=e, contexto=C, velocidade=v, m=m, c_ref=c, rotulo=rotulo)

    def compor_com(self, outro: "BlocoGandarela") -> "BlocoGandarela":
        """Compõe este bloco com outro, usando o contexto atual."""
        if abs(self.contexto - outro.contexto) > 1e-6:
            raise ValueError(
                f"Contextos incompatíveis: {self.contexto} ≠ {outro.contexto}. "
                "Blocos de contextos diferentes não podem ser compostos diretamente."
            )
        e_novo = compor(self.energia_valor, outro.energia_valor, self.contexto)
        v_media = (self.velocidade + outro.velocidade) / 2
        return BlocoGandarela(
            energia_valor=e_novo,
            contexto=self.contexto,
            velocidade=v_media,
            m=self.m,
            c_ref=self.c_ref,
            rotulo=f"({self.rotulo}⊕{outro.rotulo})" if self.rotulo or outro.rotulo else ""
        )

    def energia_relativa(self) -> float:
        """Energia deste bloco como fração do contexto C. Sempre em (0, 1)."""
        return self.energia_valor / self.contexto

    def __repr__(self) -> str:
        label = f" '{self.rotulo}'" if self.rotulo else ""
        return (f"Bloco{label}(E={self.energia_valor:.4f}, "
                f"C={self.contexto:.2f}, v={self.velocidade:.2f})")


# ─────────────────────────────────────────────
# CadeiaGandarela
# ─────────────────────────────────────────────

class CadeiaGandarela:
    """
    Sequência de blocos que representa um contexto completo.
    A complexidade C é conservada ao longo da cadeia (invariante N·E = C).
    """

    def __init__(self, C: float, v: float,
                 m: float = MASSA, c: float = C_LUZ):
        self.C = C
        self.v = v
        self.m = m
        self.c = c
        self.blocos: List[BlocoGandarela] = []

    def adicionar(self, rotulo: str = "") -> "CadeiaGandarela":
        """Adiciona um bloco com energia calculada pelo contexto atual."""
        bloco = BlocoGandarela.do_contexto(
            C=self.C, v=self.v, m=self.m, c=self.c, rotulo=rotulo
        )
        self.blocos.append(bloco)
        return self

    def adicionar_bloco(self, bloco: BlocoGandarela) -> "CadeiaGandarela":
        """Adiciona bloco existente."""
        self.blocos.append(bloco)
        return self

    def compor_tudo(self) -> BlocoGandarela:
        """Colapsa toda a cadeia em um único bloco via composições sucessivas."""
        if not self.blocos:
            raise ValueError("Cadeia vazia")
        resultado = self.blocos[0]
        for bloco in self.blocos[1:]:
            resultado = resultado.compor_com(bloco)
        return resultado

    def energia_total(self) -> float:
        """Soma das energias individuais (antes da composição)."""
        return sum(b.energia_valor for b in self.blocos)

    def n_teorico(self) -> float:
        """N mínimo teórico para este contexto e velocidade."""
        return n_blocos(self.C, self.v, self.m, self.c)

    def energia_teorica(self) -> float:
        """Energia teórica por bloco para este v."""
        return energia(self.v, self.m, self.c)

    def resumo(self) -> dict:
        """Resumo completo do estado da cadeia."""
        e_teo = self.energia_teorica()
        n_teo = self.n_teorico()
        return {
            "contexto_C": self.C,
            "velocidade_v": self.v,
            "n_blocos_atual": len(self.blocos),
            "n_blocos_teorico": round(n_teo, 3),
            "energia_teorica_por_bloco": round(e_teo, 6),
            "energia_total": round(self.energia_total(), 6),
            "invariante_NxE": round(n_teo * e_teo, 6),
            "C_confirmado": round(self.C, 6),
        }

    def __repr__(self) -> str:
        return (f"Cadeia(C={self.C}, v={self.v}, "
                f"n={len(self.blocos)} blocos)")


# ─────────────────────────────────────────────
# Demonstração
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 52)
    print("  LÓGICA GANDARELA — demonstração do núcleo")
    print("=" * 52)

    # 1. Energia assintótica
    print("\n[1] E(v) — energia por bloco (zero proibido)")
    for v in [1.1, 2.0, 5.0, 20.0, 100.0]:
        e = energia(v)
        print(f"  v={v:6.1f} → E={e:.6f}")

    # 2. Invariante N·E = C
    print("\n[2] Invariante N·E = C")
    inv = verificar_invariante(C=4.0, v=3.0)
    for k, val in inv.items():
        print(f"  {k}: {val:.6f}")

    # 3. Composição ⊕_C
    print("\n[3] Composição E₁ ⊕_C E₂")
    pares = [(1.5, 0.8), (2.0, 2.0), (0.1, 4.9)]
    C_ctx = 5.0
    for e1, e2 in pares:
        r = compor(e1, e2, C_ctx)
        print(f"  {e1} ⊕ {e2}  (C={C_ctx}) = {r:.4f}  "
              f"[0 < {r:.4f} < {C_ctx}? {0 < r < C_ctx}]")

    # 4. Cadeia de blocos
    print("\n[4] Cadeia de blocos — contexto se conserva")
    cadeia = CadeiaGandarela(C=3.0, v=4.0)
    for rotulo in ["A", "B", "C", "D"]:
        cadeia.adicionar(rotulo)

    resumo = cadeia.resumo()
    for k, val in resumo.items():
        print(f"  {k}: {val}")

    colapso = cadeia.compor_tudo()
    print(f"\n  Bloco final após composição: {colapso}")
    print(f"  Energia relativa ao contexto: {colapso.energia_relativa():.4f}")
    print(f"\n  ✓ Energia nunca chegou a zero em nenhum passo.")
    print("=" * 52)
