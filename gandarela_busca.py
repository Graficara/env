"""
gandarela_busca.py
==================
Motor de busca semântica usando a física Gandarela.
Dado uma pergunta, encontra os paralipses com maior
ressonância energética com os tokens da pergunta.
"""

import json
import math
import re
import os

CORPUS_FILE = 'corpus.json'
EPSILON = 1e-10

STOPWORDS = {
    'a', 'o', 'e', 'é', 'de', 'do', 'da', 'dos', 'das', 'em', 'no', 'na',
    'nos', 'nas', 'um', 'uma', 'uns', 'umas', 'que', 'se', 'por', 'para',
    'com', 'como', 'mas', 'ou', 'ao', 'aos', 'às', 'seu', 'sua', 'seus',
    'suas', 'esse', 'essa', 'esses', 'essas', 'este', 'esta', 'estes',
    'estas', 'isso', 'isto', 'aqui', 'ali', 'já', 'não', 'mais', 'muito',
    'também', 'ainda', 'bem', 'só', 'foi', 'são', 'ser', 'ter', 'tem',
    'entre', 'sobre', 'pelo', 'pela', 'pelos', 'pelas', 'onde', 'quando',
    'quem', 'qual', 'porque', 'pois'
}

def tokenizar(texto: str) -> list:
    if not texto:
        return []
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', ' ', texto)
    texto = re.sub(r'\d+', ' ', texto)
    tokens = [t.strip() for t in texto.split() if len(t.strip()) > 2]
    return [t for t in tokens if t not in STOPWORDS]

def carregar_corpus() -> dict:
    """Carrega o corpus indexado."""
    if not os.path.exists(CORPUS_FILE):
        raise FileNotFoundError(
            f"Corpus não encontrado. Execute gandarela_corpus.py primeiro."
        )
    with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def energia_residual(corpus: dict, token: str) -> float:
    """
    Retorna energia residual para tokens não encontrados no corpus.
    Nunca zero — princípio Gandarela.
    """
    # Quanto menor o vocabulário coberto, maior a incerteza residual
    cobertura = len(corpus.get('indice_invertido', {}))
    return EPSILON * (1 + 1.0 / max(cobertura, 1))

def buscar(pergunta: str, top_n: int = 5) -> list:
    """
    Busca os paralipses mais relevantes para uma pergunta.
    
    Retorna lista de dicts com:
    - id, titulo, data, sintese, discurso
    - energia_ressonancia: quão forte é a conexão com a pergunta
    - tokens_encontrados: quais tokens da pergunta ressoaram
    - cobertura: % dos tokens da pergunta encontrados no corpus
    """
    corpus = carregar_corpus()
    indice = corpus['indice']
    indice_invertido = corpus['indice_invertido']

    tokens = tokenizar(pergunta)
    if not tokens:
        return []

    # Acumula energia de ressonância por paralipse
    ressonancia = {}  # pid → energia acumulada
    tokens_encontrados = {}  # pid → lista de tokens que ressoaram

    for token in tokens:
        if token in indice_invertido:
            for entrada in indice_invertido[token]:
                pid = entrada['id']
                e = entrada['energia']
                if pid not in ressonancia:
                    ressonancia[pid] = EPSILON  # nunca começa do zero
                    tokens_encontrados[pid] = []
                # Composição harmônica Gandarela — acumula sem anular
                ressonancia[pid] = ressonancia[pid] + e - (ressonancia[pid] * e)
                tokens_encontrados[pid].append(token)
        else:
            # Token não está no corpus — energia residual para todos
            for pid in indice:
                if pid not in ressonancia:
                    ressonancia[pid] = EPSILON
                ressonancia[pid] += energia_residual(corpus, token)

    if not ressonancia:
        return []

    # Ordena por energia de ressonância
    ordenados = sorted(ressonancia.items(), key=lambda x: x[1], reverse=True)

    resultados = []
    for pid, e_total in ordenados[:top_n]:
        dados = indice.get(pid, {})
        cobertura = len(set(tokens_encontrados.get(pid, []))) / max(len(tokens), 1)

        resultados.append({
            'id': dados.get('id'),
            'titulo': dados.get('titulo', ''),
            'data': dados.get('data', ''),
            'discurso': dados.get('discurso', ''),
            'sintese': dados.get('sintese', ''),
            'energia_ressonancia': round(e_total, 6),
            'tokens_encontrados': list(set(tokens_encontrados.get(pid, []))),
            'cobertura': round(cobertura, 2),
            'dentro_corpus': cobertura > 0
        })

    return resultados

def diagnostico(pergunta: str) -> dict:
    """
    Diagnóstico energético de uma pergunta.
    Mostra quais tokens foram encontrados, quais são novos.
    """
    corpus = carregar_corpus()
    indice_invertido = corpus['indice_invertido']
    tokens = tokenizar(pergunta)

    conhecidos = [t for t in tokens if t in indice_invertido]
    desconhecidos = [t for t in tokens if t not in indice_invertido]

    return {
        'tokens': tokens,
        'conhecidos': conhecidos,
        'desconhecidos': desconhecidos,
        'cobertura': round(len(conhecidos) / max(len(tokens), 1), 2),
        'dentro_corpus': len(conhecidos) > 0
    }

if __name__ == '__main__':
    print("=== TESTE DE BUSCA GANDARELA ===\n")

    perguntas = [
        "capitalismo e guerra",
        "algoritmos controlam a verdade",
        "família e violência doméstica",
        "tecnologia e liberdade",
        "o que é o paralipse"
    ]

    for pergunta in perguntas:
        print(f"Pergunta: '{pergunta}'")
        diag = diagnostico(pergunta)
        print(f"  Tokens conhecidos: {diag['conhecidos']}")
        print(f"  Cobertura: {diag['cobertura']*100:.0f}%")

        resultados = buscar(pergunta, top_n=3)
        for i, r in enumerate(resultados, 1):
            print(f"  [{i}] {r['titulo']}")
            print(f"      Ressonância: {r['energia_ressonancia']:.4f} | Tokens: {r['tokens_encontrados']}")
        print()
