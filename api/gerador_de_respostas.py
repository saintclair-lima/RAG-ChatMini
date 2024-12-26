import json
import asyncio
import torch

from concurrent.futures import ThreadPoolExecutor
from time import time
from transformers import BertTokenizer, BertForQuestionAnswering, pipeline
from typing import Callable, Generator

from api.environment.environment import environment
from api.utils.utils import InterfaceChroma, InterfaceOllama, DadosChat
    

class GeradorDeRespostas:
    '''
    Classe cuja função é realizar consulta em um banco de vetores existente e, por meio de uma API de um LLM
    gera uma texto de resposta que condensa as informações resultantes da consulta.
    '''
    def __init__(self,
                url_banco_vetores:str=environment.URL_BANCO_VETORES,
                colecao_de_documentos:str=environment.NOME_COLECAO_DE_DOCUMENTOS,
                funcao_de_embeddings:Callable=None,
                fazer_log:bool=True,
                device: str=None):

        self.device = device
        self.executor = ThreadPoolExecutor(max_workers=environment.THREADPOOL_MAX_WORKERS)
        
        if fazer_log: print(f'-- Gerador de respostas em inicialização (device={self.device})...')

        self.interface_chromadb = InterfaceChroma(url_banco_vetores, colecao_de_documentos, funcao_de_embeddings, fazer_log)

        # Carregando modelo e tokenizador pre-treinados
        # optou-se por não usar pipeline, por ser mais lento que usar o modelo diretamente
        if fazer_log: print(f'--- preparando modelo e tokenizador do Bert (usando {environment.EMBEDDING_SQUAD_PORTUGUESE})...')
        self.modelo_bert_qa = BertForQuestionAnswering.from_pretrained(environment.EMBEDDING_SQUAD_PORTUGUESE).to(self.device)
        self.tokenizador_bert = BertTokenizer.from_pretrained(environment.EMBEDDING_SQUAD_PORTUGUESE, device=self.device)
        
        self.modelo_bert_qa_pipeline = pipeline("question-answering", environment.EMBEDDING_SQUAD_PORTUGUESE, device=self.device)

        if fazer_log: print(f'--- preparando o Llama (usando {environment.MODELO_LLAMA})...')
        self.interface_ollama = InterfaceOllama(url_llama=environment.URL_LLAMA, nome_modelo=environment.MODELO_LLAMA)

    async def consultar_documentos_banco_vetores(self, pergunta: str, num_resultados:int=environment.NUM_DOCUMENTOS_RETORNADOS):
        return self.interface_chromadb.consultar_documentos(pergunta, num_resultados)
    
    def formatar_lista_documentos(self, documentos: dict):
        return [
            {
                'id': documentos['ids'][0][idx],
                 'score_distancia': 1 - documentos['distances'][0][idx], # Distância do cosseno vaia entre 1 e 0
                 'metadados': documentos['metadatas'][0][idx],
                 'conteudo': f"{documentos['documents'][0][idx]}"
            }
            for idx in range(len(documentos['ids'][0]))]

    # AFAZER: Considerar remover essa função.
    # Era utilizada com gerar_resposta_llama, mas ficou obsoleta usando o for assíncrono
    async def async_stream_wrapper(self, sync_generator: Generator):
        loop = asyncio.get_running_loop()
        for item in sync_generator:
            yield await loop.run_in_executor(self.executor, lambda x=item: x)

    async def estimar_resposta(self, pergunta, texto_documento: str):
        # Optou-se por não utilizar a abordagem com pipeline por ser mais lenta
        res = self.modelo_bert_qa_pipeline(question=pergunta, context=texto_documento)
        inputs = self.tokenizador_bert.encode_plus(
            pergunta,
            texto_documento,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = self.modelo_bert_qa(**inputs)

        # AFAZER: Avaliar se score ponderado faz sentido
        # Extraindo os logits como tensores
        logits_inicio = outputs.start_logits
        logits_fim = outputs.end_logits

        # Obtenção dos logits positivos
        logits_inicio_positivos = logits_inicio[0][logits_inicio[0] > 0]
        logits_fim_positivos = logits_fim[0][logits_fim[0] > 0]

        # Média dos logits positivos
        media_logits_inicio_positivos = logits_inicio_positivos.mean().item() if logits_inicio_positivos.numel() > 0 else 0
        media_logits_fim_positivos = logits_fim_positivos.mean().item() if logits_fim_positivos.numel() > 0 else 0

        # Obtendo os índices e valores dos melhores logits
        indice_melhor_logit_inicio = logits_inicio[0].argmax().item()
        melhor_logit_inicio = logits_inicio[0][indice_melhor_logit_inicio].item()

        indice_melhor_logit_fim = logits_fim[0].argmax().item()
        melhor_logit_fim = logits_fim[0][indice_melhor_logit_fim].item()

        # Calcula media_logits_positivos
        media_logits_positivos = (media_logits_inicio_positivos + media_logits_fim_positivos) / 2

        # scores em formato float para serialização com JSON
        score = float(melhor_logit_inicio + melhor_logit_fim)
        score_ponderado = float(score * media_logits_positivos)

        # calculando score estimado
        start_logits_softmax = torch.softmax(logits_inicio, dim=-1)
        end_logits_softmax = torch.softmax(logits_fim, dim=-1)
        score_estimado = float(torch.max(start_logits_softmax).item() * torch.max(end_logits_softmax).item())

        # score: soma do melhor Logit inicial com o melhor logit final
        # score_estimado: multiplicação do softmax dos logits de inicio pelo dos logits de fim
        # -- (manter somente se a performance do score do Bert pelo pipeline ficar lenta)
        # score_ponderado: score ponderado pela média dos logits de inicio e fim, só quando positivos 
        # -- (quanto mais logits positivos, mais o documento tem melhor avaliação)

        tokens_resposta = inputs['input_ids'][0][indice_melhor_logit_inicio:indice_melhor_logit_fim + 1]
        resposta = self.tokenizador_bert.decode(tokens_resposta, skip_special_tokens=True)
        
        return {
            'resposta': (resposta, res['answer']),
            'score': (score, res['score'], score_estimado),
            'score_ponderado': score_ponderado
        }

    async def consultar(self, dados_chat: DadosChat, fazer_log:bool=True):
        contexto = dados_chat.contexto
        pergunta = dados_chat.pergunta
        
        if len(pergunta.split(' ')) > 300:
            #AFAZER: decidir se mantém essa limitação. Colocada a princípio para evitar
            #        problema de truncation com o Bert. Ajuda com Prompt Injection?
            yield json.dumps({
                    "erro": "Pergunta com mais de 300 palavras",
                    "mensagem": "Por motivos de segurança, a pergunta deve ter no máximo 300 palavras. Por favor, reformule o que você deseja perguntar, para ficar dentro desse limite.",
                    "severidade": "leve"
                    })
            return

        if fazer_log: print(f'Gerador de respostas: realizando consulta para "{pergunta}"...')

        # Recuperando documentos usando o ChromaDB
        marcador_tempo_inicio = time()
        documentos = await self.consultar_documentos_banco_vetores(pergunta)
        lista_documentos = self.formatar_lista_documentos(documentos)
        marcador_tempo_fim = time()
        tempo_consulta = marcador_tempo_fim - marcador_tempo_inicio
        if fazer_log: print(f'--- consulta no banco concluída ({tempo_consulta} segundos)')

        # # Atribuindo scores usando Bert
        # if fazer_log: print(f'--- aplicando scores do Bert aos documentos recuperados...')
        # marcador_tempo_inicio = time()
        # for documento in lista_documentos:
        #     try:
        #         resposta_estimada = await self.estimar_resposta(pergunta, documento['conteudo'])
        #         documento['score_bert'] = resposta_estimada['score']
        #         documento['score_ponderado'] = resposta_estimada['score_ponderado']
        #         documento['resposta_bert'] = resposta_estimada['resposta']
        #     except Exception as excecao:
        #         print ('Vixe, deu erro. Tentando continuar...')
        #         documento['score_bert'] = None
        #         documento['score_ponderado'] = None
        #         documento['resposta_bert'] = None
        #         yield json.dumps({
        #             "info": str(excecao),
        #             "mensagem": "Houve um erro ao avaliar os documentos encontrados. Documentos retornados sem valor de avaliação",
        #             "severidade": "leve"
        #             })
        # marcador_tempo_fim = time()
        # tempo_bert = marcador_tempo_fim - marcador_tempo_inicio
        # if fazer_log: print(f'--- scores atribuídos ({tempo_bert} segundos)')
        
        docs = [doc['conteudo'].split(' ') for doc in lista_documentos]
        ctxt = list(contexto)
        for doc in docs: ctxt += doc
        mock_llama_data = {
            'context': ctxt * 3,
            'response': 'Esta é uam resposta padrão pra ser usada somente em teztes'
        }
        marcador_tempo_fim = time()
        tempo_llama = marcador_tempo_fim - marcador_tempo_inicio
        if fazer_log: print(f'--- resposta do Llama concluída ({tempo_llama} segundos)')

        yield "CHEGOU_AO_FIM_DO_TEXTO_DA_RESPOSTA"

        # Retornando dados compilados
        yield json.dumps(
            {
                "pergunta": pergunta,
                "documentos": lista_documentos,
                "resposta_llama": mock_llama_data,
                "resposta": mock_llama_data['response'].replace('\n\n', '\n'),
                "tempo_consulta": tempo_consulta,
                #"tempo_bert": tempo_bert,
                "tempo_inicio_resposta": 0.0,
                "tempo_llama_total": tempo_llama
            },
            ensure_ascii=False
        )
        print('Concluído')

    