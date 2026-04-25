"""
gandarela_chat.py
=================
Interface de chat que usa a Lógica Gandarela para responder.
O sistema responde apenas com o que aprendeu dos paralipses.
Energia nunca é zero — sempre há algo a dizer.

Expõe uma API HTTP simples para ser consumida pelo frontend.
"""

import json
import math
import os
import re
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from gandarela_busca import buscar, diagnostico, tokenizar, carregar_corpus

# ============================================
# CONFIGURAÇÃO
# ============================================
PORT = int(os.environ.get('PORT', 8000))
EPSILON = 1e-10

# ============================================
# MOTOR DE RESPOSTA GANDARELA
# ============================================

def compor_resposta(pergunta: str) -> dict:
    """
    Constrói uma resposta baseada nos paralipses mais ressonantes.
    
    Nunca retorna vazio — princípio Gandarela.
    A resposta reflete com honestidade o nível de energia do sistema
    em relação à pergunta.
    """
    diag = diagnostico(pergunta)
    resultados = buscar(pergunta, top_n=3)

    cobertura = diag['cobertura']
    conhecidos = diag['conhecidos']
    desconhecidos = diag['desconhecidos']

    # Caso 1: Alta ressonância — sistema conhece bem o tema
    if resultados and cobertura >= 0.5:
        principal = resultados[0]
        secundarios = resultados[1:]

        resposta = _resposta_alta_energia(
            pergunta, principal, secundarios, conhecidos
        )
        nivel = 'alto'

    # Caso 2: Ressonância parcial — sistema conhece parte do tema
    elif resultados and cobertura > 0:
        principal = resultados[0]
        resposta = _resposta_media_energia(
            pergunta, principal, conhecidos, desconhecidos
        )
        nivel = 'medio'

    # Caso 3: Baixa ressonância — tema fora do corpus
    # Mas energia residual nunca é zero
    else:
        resposta = _resposta_energia_residual(pergunta, desconhecidos)
        nivel = 'residual'

    return {
        'resposta': resposta,
        'nivel_energia': nivel,
        'cobertura': cobertura,
        'tokens_ativos': conhecidos,
        'tokens_novos': desconhecidos,
        'fontes': [
            {
                'id': r['id'],
                'titulo': r['titulo'],
                'data': r['data'],
                'energia': r['energia_ressonancia']
            }
            for r in resultados
        ]
    }

def _resposta_alta_energia(pergunta, principal, secundarios, tokens_ativos):
    titulo = principal['titulo']
    sintese = principal['sintese']
    discurso = principal['discurso']

    partes = []
    partes.append(f"O que aprendi sobre isso vem principalmente de **{titulo}**.")
    partes.append(f"\n\nO discurso dominante que esse paralipse interroga: *\"{discurso}\"*")

    if sintese:
        partes.append(f"\n\nA síntese: {sintese}")

    if secundarios:
        partes.append(f"\n\nEsse tema também ressoa em:")
        for s in secundarios:
            partes.append(f"\n— **{s['titulo']}** ({s['data']})")

    partes.append(f"\n\nTokens que ativaram essa resposta: {', '.join(tokens_ativos)}.")

    return ''.join(partes)

def _resposta_media_energia(pergunta, principal, conhecidos, desconhecidos):
    titulo = principal['titulo']
    sintese = principal['sintese']

    partes = []
    partes.append(
        f"Encontrei ressonância parcial com **{titulo}**."
    )

    if sintese:
        partes.append(f"\n\n{sintese}")

    if desconhecidos:
        partes.append(
            f"\n\nOs conceitos *{', '.join(desconhecidos)}* ainda não estão no meu corpus. "
            f"Quanto mais o Paralipse publicar sobre isso, mais preciso fico."
        )

    return ''.join(partes)

def _resposta_energia_residual(pergunta, desconhecidos):
    partes = []
    partes.append(
        "Esse tema ainda não está no meu corpus. "
        "Minha energia sobre ele é residual — presente, mas fraca."
    )

    if desconhecidos:
        partes.append(
            f"\n\nOs conceitos *{', '.join(desconhecidos[:5])}* não aparecem "
            f"nos paralipses que estudei até agora."
        )

    partes.append(
        "\n\nQuando o Paralipse publicar sobre isso, aprenderei. "
        "Até lá, posso falar sobre o que já conheço: capitalismo, "
        "algoritmos, poder, desigualdade, tecnologia, linguagem, violência."
    )

    return ''.join(partes)

# ============================================
# SERVIDOR HTTP
# ============================================

class GandarelaHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass  # silencia logs padrão

    def do_OPTIONS(self):
        self._cors()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        self._cors()

        if path == '/status':
            self._json({'status': 'online', 'motor': 'Gandarela'})

        elif path == '/corpus':
            try:
                corpus = carregar_corpus()
                self._json({
                    'total_paralipses': corpus['total_paralipses'],
                    'vocabulario': corpus['vocabulario'],
                    'total_tokens': corpus['total_tokens']
                })
            except FileNotFoundError:
                self._json({'erro': 'Corpus não construído. Rode gandarela_corpus.py'}, 503)

        elif path == '/chat':
            pergunta = params.get('q', [''])[0].strip()
            if not pergunta:
                self._json({'erro': 'Parâmetro q ausente. Ex: /chat?q=capitalismo'}, 400)
                return
            try:
                resultado = compor_resposta(pergunta)
                self._json(resultado)
            except FileNotFoundError:
                self._json({'erro': 'Corpus não construído. Rode gandarela_corpus.py'}, 503)

        elif path == '/buscar':
            pergunta = params.get('q', [''])[0].strip()
            top = int(params.get('top', ['5'])[0])
            if not pergunta:
                self._json({'erro': 'Parâmetro q ausente'}, 400)
                return
            try:
                resultados = buscar(pergunta, top_n=top)
                self._json({'resultados': resultados})
            except FileNotFoundError:
                self._json({'erro': 'Corpus não construído'}, 503)

        else:
            self._json({
                'motor': 'Gandarela',
                'rotas': {
                    'GET /status': 'Status do sistema',
                    'GET /corpus': 'Informações do corpus',
                    'GET /chat?q=pergunta': 'Conversa com o Gandarela',
                    'GET /buscar?q=pergunta&top=5': 'Busca paralipses relevantes'
                }
            })

    def do_POST(self):
        parsed = urlparse(self.path)
        self._cors()

        if parsed.path == '/chat':
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
                pergunta = data.get('pergunta', '').strip()
                if not pergunta:
                    self._json({'erro': 'Campo pergunta ausente'}, 400)
                    return
                resultado = compor_resposta(pergunta)
                self._json(resultado)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                self._json({'erro': str(e)}, 500)
        else:
            self._json({'erro': 'Rota não encontrada'}, 404)

    def _cors(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def _json(self, data, code=200):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8'))

# ============================================
# ENTRADA
# ============================================

if __name__ == '__main__':
    print(f"Gandarela Chat rodando na porta {PORT}")
    print(f"  GET /status")
    print(f"  GET /corpus")
    print(f"  GET /chat?q=sua+pergunta")
    print(f"  GET /buscar?q=sua+pergunta")
    print()

    # Verifica corpus
    try:
        corpus = carregar_corpus()
        print(f"Corpus carregado: {corpus['total_paralipses']} paralipses, "
              f"{corpus['vocabulario']} tokens únicos")
    except FileNotFoundError:
        print("AVISO: Corpus não encontrado. Execute gandarela_corpus.py antes de iniciar o chat.")

    server = HTTPServer(('0.0.0.0', PORT), GandarelaHandler)
    print(f"\nServidor iniciado. Acesse http://localhost:{PORT}")
    server.serve_forever()
