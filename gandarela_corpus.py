"""
gandarela_corpus.py
===================
Lê todos os paralipses via API e constrói o índice semântico
usando a física Gandarela. Cada palavra vira um bloco com energia
proporcional à sua densidade no corpus inteiro.
"""

import json
import math
import re
import os
from collections import Counter
from urllib.request import urlopen, Request
from urllib.error import URLError

# ============================================
# CONFIGURAÇÃO
# ============================================
API_URL = os.environ.get(
    'PARALIPSE_API_URL',
    'https://paralipse2.graficagandarela.com.br/api/gandarela_api.php'
)
API_TOKEN = os.environ.get('PARALIPSE_API_TOKEN', 'gandarela_token_2026')
CORPUS_FILE = 'corpus.json'

# Stopwords — palavras funcionais com energia naturalmente baixa
STOPWORDS = {
    'a', 'o', 'e', 'é', 'de', 'do', 'da', 'dos', 'das', 'em', 'no', 'na',
    'nos', 'nas', 'um', 'uma', 'uns', 'umas', 'que', 'se', 'por', 'para',
    'com', 'como', 'mas', 'ou', 'ao', 'aos', 'às', 'seu', 'sua', 'seus',
    'suas', 'esse', 'essa', 'esses', 'essas', 'este', 'esta', 'estes',
    'estas', 'isso', 'isto', 'aqui', 'ali', 'já', 'não', 'mais', 'muito',
    'também', 'ainda', 'bem', 'só', 'foi', 'são', 'ser', 'ter', 'tem',
    'entre', 'sobre', 'pelo', 'pela', 'pelos', 'pelas', 'onde', 'quando',
    'quem', 'qual', 'porque', 'pois', 'then', 'the', 'of', 'in', 'to'
}

# ============================================
# FÍSICA GANDARELA
# ============================================
EPSILON = 1e-10  # piso — energia nunca é zero

def energia(frequencia_relativa: float) -> float:
    """
    Converte frequência relativa em energia Gandarela.
    Palavras raras têm alta energia. Palavras comuns têm baixa energia.
    Mas nenhuma chega a zero.
    """
    if frequencia_relativa <= 0:
        return EPSILON
    # Inversão logarítmica — raridade = energia
    e = 1.0 / (1.0 + math.log(1.0 + frequencia_relativa * 1000))
    return max(e, EPSILON)

def energia_campo(texto: str, freq_global: dict, total_tokens: int) -> dict:
    """
    Processa um campo de texto e retorna blocos de energia por token.
    """
    tokens = tokenizar(texto)
    blocos = {}
    for token in tokens:
        if token in STOPWORDS:
            freq_rel = freq_global.get(token, 1) / max(total_tokens, 1)
            blocos[token] = energia(freq_rel * 10)  # stopwords: energia baixa mas não zero
        else:
            freq_rel = freq_global.get(token, 1) / max(total_tokens, 1)
            blocos[token] = energia(freq_rel)
    return blocos

def tokenizar(texto: str) -> list:
    """Normaliza e tokeniza texto em português."""
    if not texto:
        return []
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', ' ', texto)
    texto = re.sub(r'\d+', ' ', texto)
    tokens = [t.strip() for t in texto.split() if len(t.strip()) > 2]
    return tokens

# ============================================
# ACESSO À API
# ============================================
def buscar_paralipses() -> list:
    """Busca todos os paralipses via API."""
    url = f"{API_URL}?action=listar&token={API_TOKEN}"
    try:
        req = Request(url, headers={'X-Gandarela-Token': API_TOKEN})
        with urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data.get('paralipses', [])
    except URLError as e:
        print(f"Erro ao acessar API: {e}")
        return []

# ============================================
# CONSTRUÇÃO DO CORPUS
# ============================================
def construir_corpus() -> dict:
    """
    Baixa todos os paralipses, processa em blocos Gandarela
    e salva o índice semântico em corpus.json.
    """
    print("Buscando paralipses da API...")
    paralipses = buscar_paralipses()

    if not paralipses:
        print("Nenhum paralipse encontrado.")
        return {}

    print(f"{len(paralipses)} paralipses recebidos. Construindo corpus...")

    # Campos que alimentam o aprendizado
    CAMPOS = ['titulo', 'discurso', 'interrogacoes', 'contextualizacao', 'analise', 'sintese']

    # Passo 1 — frequência global de todos os tokens no corpus inteiro
    freq_global = Counter()
    for p in paralipses:
        for campo in CAMPOS:
            tokens = tokenizar(p.get(campo, '') or '')
            freq_global.update(tokens)

    total_tokens = sum(freq_global.values())
    print(f"Vocabulário total: {len(freq_global)} tokens únicos, {total_tokens} ocorrências")

    # Passo 2 — construir índice semântico por paralipse
    indice = {}
    for p in paralipses:
        pid = str(p['id'])
        texto_completo = ' '.join([
            p.get(campo, '') or '' for campo in CAMPOS
        ])

        # Tokens deste paralipse
        tokens_local = tokenizar(texto_completo)
        freq_local = Counter(tokens_local)

        # Energia de cada token neste paralipse
        # Combina frequência local (relevância) com raridade global (especificidade)
        blocos = {}
        for token, freq in freq_local.items():
            if token in STOPWORDS:
                continue
            freq_rel_global = freq_global.get(token, 1) / total_tokens
            freq_rel_local = freq / max(len(tokens_local), 1)
            # TF-IDF Gandarela — combina presença local com raridade global
            e = energia(freq_rel_global) * (1 + math.log(1 + freq_rel_local * 100))
            blocos[token] = round(e, 6)

        # Ordena por energia decrescente
        blocos_ordenados = dict(
            sorted(blocos.items(), key=lambda x: x[1], reverse=True)
        )

        indice[pid] = {
            'id': p['id'],
            'titulo': p.get('titulo', ''),
            'data': p.get('data_publicacao', ''),
            'discurso': p.get('discurso', ''),
            'sintese': p.get('sintese', ''),
            'blocos': blocos_ordenados,
            'energia_total': round(sum(blocos_ordenados.values()), 4),
            'tokens_unicos': len(blocos_ordenados)
        }

    # Passo 3 — índice global de tokens → quais paralipses os contêm
    indice_invertido = {}
    for pid, dados in indice.items():
        for token, e in dados['blocos'].items():
            if token not in indice_invertido:
                indice_invertido[token] = []
            indice_invertido[token].append({'id': pid, 'energia': e})

    # Ordena cada entrada do índice invertido por energia
    for token in indice_invertido:
        indice_invertido[token].sort(key=lambda x: x['energia'], reverse=True)

    corpus = {
        'total_paralipses': len(paralipses),
        'total_tokens': total_tokens,
        'vocabulario': len(freq_global),
        'indice': indice,
        'indice_invertido': indice_invertido
    }

    # Salva corpus
    with open(CORPUS_FILE, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print(f"\nCorpus salvo em {CORPUS_FILE}")
    print(f"  Paralipses indexados: {len(indice)}")
    print(f"  Vocabulário único: {len(indice_invertido)} tokens")

    # Mostra os 10 tokens de maior energia no corpus
    energia_por_token = {
        token: max(e['energia'] for e in entradas)
        for token, entradas in indice_invertido.items()
    }
    top10 = sorted(energia_por_token.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 tokens mais energéticos no corpus:")
    for token, e in top10:
        print(f"  {token:<20} {e:.6f}")

    return corpus

if __name__ == '__main__':
    construir_corpus()
