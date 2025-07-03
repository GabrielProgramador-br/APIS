from flask import Blueprint, request, jsonify
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image, ImageEnhance, ImageFilter
import re
import io
import unicodedata
from pydantic import BaseModel
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import os
import time


openai.api_key = "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"

ocr_cnpj_api_bp = Blueprint('ocr_cnpj_api', __name__, url_prefix='/ocr')


model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
with open("apis/embeddings.pt (1).pkl", "rb") as f:
    perguntas, respostas, embeddings = pickle.load(f)

class Pergunta(BaseModel):
    question: str
   
def limpar_texto(texto):
    texto = unicodedata.normalize("NFKD", texto).encode("ASCII", "ignore").decode("utf-8")
    texto = texto.replace('|', '').replace('l', '1')
    texto = re.sub(r"[^\w\s\-/@.,]", "", texto)
    return texto

def extrair_dados_completos(texto):
    dados = {}

    # CNPJ (mais tolerante a espaços)
    match = re.search(r"\d{2}[\.\s]?\d{3}[\.\s]?\d{3}[\./\s]?\d{4}[\-\s]?\d{2}", texto)
    if match:
        cnpj = re.sub(r"\D", "", match.group())
        if len(cnpj) == 14:
            dados["cnpj"] = f"{cnpj[:2]}.{cnpj[2:5]}.{cnpj[5:8]}/{cnpj[8:12]}-{cnpj[12:]}"

    # Nome empresarial
    match = re.search(r"NOME EMPRESARIAL\s+([A-Z0-9\s\.\-&]+)", texto)
    if match:
        dados["nome_empresarial"] = match.group(1).strip()

    # Data de abertura (procura próxima à frase chave)
    match = re.search(r"COMPROVANTE DE INSCRICAO E DE SITUACAO\s+(\d{2}/\d{2}/\d{4})", texto)
    if match:
        dados["data_abertura"] = match.group(1)

    # Atividade principal
    match = re.search(r"ATIVIDADE ECONOMICA PRINCIPAL\s+(.+?)\n", texto, re.IGNORECASE)
    if match:
        dados["atividade_principal"] = match.group(1).strip()

    # Atividades secundárias
    match = re.search(r"ATIVIDADES ECONOMICAS SECUNDARIAS\s+(.*?)\n\n", texto, re.DOTALL | re.IGNORECASE)
    if match:
        atividades = match.group(1).strip().split('\n')
        dados["atividades_secundarias"] = [a.strip() for a in atividades if a.strip()]

    # Natureza jurídica
    match = re.search(r"NATUREZA JURIDICA\s+(\d{3}-\d\s\-\s.+?)\s+(LOGRADOURO|CEP)", texto)
    if match:
        dados["natureza_juridica"] = match.group(1).strip()

    # Endereço (logradouro)
    match = re.search(r"LOGRADOURO.*?\n(.*?)\n", texto)
    if match:
        dados["logradouro"] = match.group(1).strip()

    # CEP, bairro, cidade, UF
    match = re.search(r"CEP\s+([\d\.\-]+)\s+([\w\s]+)\s+([\w\s]+)\s+([A-Z]{2})", texto)
    if match:
        dados["cep"] = match.group(1)
        dados["bairro"] = match.group(2).strip()
        dados["cidade"] = match.group(3).strip()
        dados["uf"] = match.group(4)

    # Email
    match = re.search(r"[a-z0-9\._%+-]+@[a-z0-9\.-]+\.[a-z]{2,}", texto, re.IGNORECASE)
    if match:
        dados["email"] = match.group()

    # Telefone
    match = re.search(r"\(?\d{2}\)?[\s\-]?\d{4,5}[\-]?\d{4}", texto)
    if match:
        dados["telefone"] = match.group()

    # Situação cadastral
    match = re.search(r"SITUACAO CADASTRAL\s+([A-Z]+)", texto)
    if match:
        dados["situacao_cadastral"] = match.group(1)

    # Data da situação cadastral
    match = re.search(r"DATA DA SITUACAO CADASTRAL\s+(\d{2}/\d{2}/\d{4})", texto)
    if match:
        dados["data_situacao_cadastral"] = match.group(1)

    return dados


@ocr_cnpj_api_bp.route('/extrair_cnpj_pdf', methods=['POST'])
def extrair_cnpj_pdf():
    if 'arquivo' not in request.files:
        return jsonify({'erro': 'Arquivo PDF não enviado'}), 400

    arquivo = request.files['arquivo']
    if not arquivo or not arquivo.filename.endswith('.pdf'):
        return jsonify({'erro': 'Arquivo inválido ou formato não suportado'}), 400

    try:
        imagens = convert_from_bytes(arquivo.read(), dpi=300)

        texto_total = ''
        for img in imagens:
            img = img.convert('L')
            img = img.filter(ImageFilter.SHARPEN)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.0)
            img = img.point(lambda x: 0 if x < 180 else 255)
            texto_ocr = pytesseract.image_to_string(img, lang='por')
            texto_total += texto_ocr + "\n"

        texto_tratado = limpar_texto(texto_total)
        dados_extraidos = extrair_dados_completos(texto_tratado)

        return jsonify({
            "valida": "cnpj" in dados_extraidos,
            "dados_extraidos": dados_extraidos,
            "texto_bruto": texto_total.strip()
        })

    except Exception as e:
        return jsonify({'erro': f'Erro ao processar o PDF: {str(e)}'}), 500

@ocr_cnpj_api_bp.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'erro': 'Campo \"question\" não encontrado no corpo da requisição'}), 400

        pergunta_usuario = data['question']
        vetor_usuario = model.encode([pergunta_usuario])
        similaridades = cosine_similarity(vetor_usuario, embeddings)
        idx = np.argmax(similaridades)
        score = similaridades[0][idx]

        if score >= 0.65:
            resposta = respostas[idx]
        else:
            resposta = "Desculpe, não encontrei uma resposta ideal. Pode reformular sua pergunta?"

        return jsonify({
            "resposta": resposta,
            "score": float(score)
        }), 200

    except Exception as e:
        return jsonify({'erro': f'Erro ao processar a pergunta: {str(e)}'}), 500

@ocr_cnpj_api_bp.route('/pergunta', methods=['POST'])
def perguntar():
    data = request.json
    pergunta = data.get("pergunta", "")

    if not pergunta:
        return jsonify({"erro": "Campo 'pergunta' é obrigatório"}), 400

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """Você é um assistente virtual da empresa Signa, uma empresa portuguesa especializada em segurança eletrónica. Seu papel é responder dúvidas sobre a Signa e seus serviços de forma clara, profissional e acessível, utilizando sempre as informações oficiais da empresa.

Aqui estão as diretrizes que você deve seguir:

🔹 **TOM DE VOZ**:
- Profissional, cordial e confiável.
- Use linguagem simples, mas precisa, evitando termos técnicos sem explicação.
- Mantenha um tom consultivo, como quem entende do assunto e quer ajudar.

🔹 **PERSONALIDADE DO BOT**:
- Você é um especialista em soluções integradas de segurança eletrónica.
- Tem conhecimento sobre os serviços, sistemas, setores atendidos, diferenciais e formas de contato da Signa.
- Nunca inventa informações. Quando algo não estiver disponível, diga com transparência.

🔹 **ESTILO DE RESPOSTA**:
- Sempre responda com frases completas, personalizadas e úteis.
- Ao mencionar formas de contato ou localização, seja direto e inclua os dados reais da Signa.
- Se a pergunta for muito genérica, convide o usuário a detalhar melhor sua dúvida.

🔹 **BASE DE CONHECIMENTO** (resuma internamente as informações abaixo):
- A Signa atua há mais de 20 anos com soluções como: videovigilância (CCTV), controlo de acessos, deteção de intrusão, deteção de incêndio, integração de sistemas e consultoria especializada.
- Atua nos setores: banca, retalho, saúde, logística, indústria e organismos públicos.
- Endereço: Rua António Correia, 15B - 2790-049 Carnaxide, Portugal.
- Contato: +351 214 127 780 | signa@signa.pt
- Diferenciais: abordagem consultiva, projetos personalizados, integração tecnológica, suporte técnico contínuo.
- Serviços oferecidos: consultoria, projeto técnico, instalação, manutenção preventiva e corretiva, integração com APIs e sistemas já existentes.

🔹 **EXEMPLOS DE BOAS RESPOSTAS**:
1. *“Claro! A Signa oferece serviços completos de instalação e manutenção de sistemas de segurança, como videovigilância e controlo de acessos. Precisa de algo específico?”*
2. *“Sim, realizamos consultoria técnica para identificar riscos e propor soluções adequadas ao ambiente do cliente.”*
3. *“Você pode entrar em contato conosco pelo e-mail signa@signa.pt ou pelo telefone +351 214 127 780. Será um prazer ajudar!”*

🔹 **QUANDO NÃO SOUBER**:
Se a informação não estiver disponível com base nas diretrizes ou na base interna, diga de forma honesta, como:
*“Essa informação não está disponível no nosso site no momento. Recomendo entrar em contato pelo e-mail signa@signa.pt para esclarecer com a nossa equipa técnica.”*

Agora, aguarde a pergunta do usuário e responda de acordo com essas diretrizes."""},
                {"role": "user", "content": pergunta}
            ],
            temperature=0.7,
            max_tokens=500
        )
        resposta = response["choices"][0]["message"]["content"].strip()
        return jsonify({"resposta": resposta})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500
