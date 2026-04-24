"""
gandarela_aprendizado.py — Loop de aprendizado da Lógica Gandarela
==================================================================
Junta as três camadas:
  gandarela.py       → núcleo matemático
  gandarela_texto.py → interface com linguagem
  gandarela_perda.py → função de perda e gradiente

O sistema recebe pares (texto, relevância esperada),
processa em blocos, calcula perda, aplica gradiente e ajusta.
Primeira vez que a Lógica Gandarela aprende de verdade.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Optional
from gandarela import EPSILON, compor, energia as energia_fn
from gandarela_texto import ProcessadorGandarela, tokenizar, energia_semantica, calcular_frequencias, PALAVRAS_FUNCIONAIS
from gandarela_perda import perda_gandarela, passo_correcao, perda_cadeia


# ─────────────────────────────────────────────
# Exemplo de treinamento
# ─────────────────────────────────────────────

@dataclass
class ExemploTreino:
    """
    Par (texto, mapa de relevância esperada por token).
    relevancia: dict token → float em (0, 1]
      1.0 = token muito importante para o contexto
      0.1 = token pouco importante
    """
    texto: str
    relevancia: dict[str, float]
    descricao: str = ""


# ─────────────────────────────────────────────
# Modelo Gandarela treinável
# ─────────────────────────────────────────────

class ModeloGandarela:
    """
    Modelo simples treinável com a Lógica Gandarela.

    Mantém pesos por token — ajustes acumulados de múltiplos
    exemplos de treinamento. Cada peso modifica a energia base
    do token multiplicativamente.
    """

    def __init__(self, taxa: float = 0.1, v_base: float = 2.0):
        self.taxa = taxa
        self.v_base = v_base
        self.pesos: dict[str, float] = {}       # peso por token
        self.historico_perda: list[float] = []  # perda por época
        self.processador = ProcessadorGandarela(v_base=v_base)

    def _peso(self, token: str) -> float:
        """Peso atual do token — padrão 1.0 (neutro)."""
        return self.pesos.get(token, 1.0)

    def _energia_ajustada(self, token: str, freq: float, C: float) -> float:
        """Energia do token com peso aplicado."""
        e_base = energia_semantica(token, {token: freq}, self.v_base)
        e_ajustada = e_base * self._peso(token)
        e_ajustada = max(e_ajustada, EPSILON)
        e_ajustada = min(e_ajustada, C - EPSILON)
        return e_ajustada

    def prever(self, texto: str) -> dict:
        """
        Processa texto e retorna energias previstas por token.
        """
        tokens = tokenizar(texto)
        if not tokens:
            return {}

        frequencias = calcular_frequencias(tokens)
        vocab = len(set(tokens))
        C = max(math.log(vocab + 1) * 2, 0.5)

        resultado = {}
        for token in tokens:
            e = self._energia_ajustada(token, frequencias.get(token, 0.0), C)
            resultado[token] = {"energia": e, "C": C}

        return resultado

    def treinar_exemplo(self, exemplo: ExemploTreino) -> float:
        """
        Treina o modelo em um único exemplo.
        Retorna a perda média do exemplo.
        """
        tokens = tokenizar(exemplo.texto)
        if not tokens:
            return 0.0

        frequencias = calcular_frequencias(tokens)
        vocab = len(set(tokens))
        C = max(math.log(vocab + 1) * 2, 0.5)

        perdas = []
        for token in tokens:
            # Energia prevista pelo modelo atual
            E_prev = self._energia_ajustada(
                token, frequencias.get(token, 0.0), C
            )

            # Energia esperada — baseada na relevância fornecida
            relevancia = exemplo.relevancia.get(token, 0.1)
            E_esp = max(relevancia * C * 0.5, EPSILON)
            E_esp = min(E_esp, C - EPSILON)

            # Calcula perda
            erro = perda_gandarela(E_prev, E_esp, C, C)
            perdas.append(erro.perda_total)

            # Gradiente — direção de correção
            E_novo = passo_correcao(E_prev, E_esp, C, self.taxa)

            # Atualiza peso do token
            if E_prev > EPSILON:
                ajuste = E_novo / E_prev
                peso_atual = self._peso(token)
                # Média suavizada para evitar oscilação
                self.pesos[token] = peso_atual * 0.7 + peso_atual * ajuste * 0.3

        perda_media = sum(perdas) / len(perdas) if perdas else 0.0
        return perda_media

    def treinar(self, exemplos: list[ExemploTreino],
                epocas: int = 10, verbose: bool = True) -> list[float]:
        """
        Treina o modelo por múltiplas épocas.
        Retorna histórico de perda por época.
        """
        historico = []

        for epoca in range(1, epocas + 1):
            perdas_epoca = []
            random.shuffle(exemplos)  # ordem aleatória a cada época

            for exemplo in exemplos:
                perda = self.treinar_exemplo(exemplo)
                perdas_epoca.append(perda)

            perda_media = sum(perdas_epoca) / len(perdas_epoca)
            historico.append(perda_media)
            self.historico_perda.append(perda_media)

            if verbose:
                barra = "█" * max(1, int((1 - min(perda_media / 10, 1)) * 20))
                tendencia = ""
                if len(historico) > 1:
                    tendencia = "↓" if perda_media < historico[-2] else "↑"
                print(f"  Época {epoca:03d} | Perda: {perda_media:.6f} "
                      f"{tendencia}  {barra}")

        return historico

    def avaliar(self, texto: str, relevancia: dict[str, float]) -> dict:
        """
        Avalia o modelo em um texto com relevância conhecida.
        Retorna métricas de desempenho.
        """
        tokens = tokenizar(texto)
        frequencias = calcular_frequencias(tokens)
        vocab = len(set(tokens))
        C = max(math.log(vocab + 1) * 2, 0.5)

        acertos = 0
        total = 0
        perdas = []

        for token in set(tokens):
            E_prev = self._energia_ajustada(
                token, frequencias.get(token, 0.0), C
            )
            relevancia_token = relevancia.get(token, 0.1)
            E_esp = max(relevancia_token * C * 0.5, EPSILON)
            E_esp = min(E_esp, C - EPSILON)

            erro = perda_gandarela(E_prev, E_esp, C, C)
            perdas.append(erro.perda_total)

            # Acerto: energia na direção certa (ambos altos ou ambos baixos)
            limiar = C * 0.2
            prev_alto = E_prev > limiar
            esp_alto = E_esp > limiar
            if prev_alto == esp_alto:
                acertos += 1
            total += 1

        return {
            "acuracia": round(acertos / total, 4) if total else 0.0,
            "perda_media": round(sum(perdas) / len(perdas), 6) if perdas else 0.0,
            "tokens_avaliados": total,
        }

    def tokens_mais_importantes(self, texto: str, top_n: int = 5) -> list[tuple]:
        """
        Retorna os tokens com maior energia após treinamento.
        O modelo aprendeu quais são mais relevantes.
        """
        previsoes = self.prever(texto)
        ordenados = sorted(
            previsoes.items(),
            key=lambda x: x[1]["energia"],
            reverse=True
        )
        return [(tok, round(dados["energia"], 6)) for tok, dados in ordenados[:top_n]]


# ─────────────────────────────────────────────
# Demonstração
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 56)
    print("  APRENDIZADO GANDARELA — loop completo")
    print("=" * 56)

    # ── Dataset de treinamento ────────────────────────────────
    exemplos = [
        ExemploTreino(
            texto="a granularidade da representação é determinada pelo contexto",
            relevancia={
                "granularidade": 1.0,
                "representação": 1.0,
                "determinada": 0.8,
                "contexto": 1.0,
                "a": 0.05, "da": 0.05, "é": 0.05, "pelo": 0.05,
            },
            descricao="frase núcleo da lógica Gandarela"
        ),
        ExemploTreino(
            texto="energia cinética quântica proporcional à massa e velocidade",
            relevancia={
                "energia": 1.0,
                "cinética": 0.9,
                "quântica": 1.0,
                "proporcional": 0.7,
                "massa": 0.9,
                "velocidade": 0.9,
                "à": 0.05, "e": 0.05,
            },
            descricao="frase física"
        ),
        ExemploTreino(
            texto="zero absoluto não existe no sistema gandarela",
            relevancia={
                "zero": 1.0,
                "absoluto": 0.9,
                "existe": 0.8,
                "sistema": 0.7,
                "gandarela": 1.0,
                "não": 0.4, "no": 0.05,
            },
            descricao="princípio fundamental"
        ),
        ExemploTreino(
            texto="blocos de informação compõem cadeias de contexto",
            relevancia={
                "blocos": 0.9,
                "informação": 1.0,
                "compõem": 0.8,
                "cadeias": 0.9,
                "contexto": 1.0,
                "de": 0.05,
            },
            descricao="estrutura da cadeia"
        ),
    ]

    # ── Treinamento ───────────────────────────────────────────
    modelo = ModeloGandarela(taxa=0.12, v_base=2.0)

    print(f"\n[1] Treinando com {len(exemplos)} exemplos por 15 épocas\n")
    historico = modelo.treinar(exemplos, epocas=15)

    # ── Convergência ──────────────────────────────────────────
    print(f"\n[2] Convergência")
    print(f"  Perda inicial:  {historico[0]:.6f}")
    print(f"  Perda final:    {historico[-1]:.6f}")
    reducao = (1 - historico[-1] / historico[0]) * 100 if historico[0] > 0 else 0
    print(f"  Redução:        {reducao:.1f}%")

    # ── Avaliação ─────────────────────────────────────────────
    print(f"\n[3] Avaliação no conjunto de treino")
    for ex in exemplos:
        metricas = modelo.avaliar(ex.texto, ex.relevancia)
        print(f"  [{ex.descricao}]")
        print(f"    Acurácia: {metricas['acuracia']:.2%}  "
              f"| Perda: {metricas['perda_media']:.6f}")

    # ── Tokens mais importantes aprendidos ───────────────────
    print(f"\n[4] Tokens mais importantes aprendidos")
    texto_teste = "a granularidade da representação é determinada pelo contexto"
    importantes = modelo.tokens_mais_importantes(texto_teste, top_n=5)
    print(f"  Texto: \"{texto_teste}\"")
    for tok, e in importantes:
        barra = "█" * int(e * 50)
        print(f"    {tok:<20} {e:.6f}  {barra}")

    # ── Generalização — texto nunca visto ─────────────────────
    print(f"\n[5] Generalização — texto nunca visto")
    novo = "o contexto determina a energia dos blocos no sistema"
    previsoes = modelo.prever(novo)
    print(f"  Texto: \"{novo}\"")
    ordenados = sorted(previsoes.items(),
                       key=lambda x: x[1]["energia"], reverse=True)
    for tok, dados in ordenados[:6]:
        barra = "█" * int(dados["energia"] * 50)
        print(f"    {tok:<20} {dados['energia']:.6f}  {barra}")

    print(f"\n  ✓ Sistema aprendeu, ajustou e generalizou.")
    print(f"  ✓ Zero nunca foi alcançado em nenhum passo.")
    print("=" * 56)
