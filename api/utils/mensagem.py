from typing import Any
import json

class Mensagem:
    def __init__(
        self,
        tipo: str,
        descricao: str = None,
        mensagem: str = None,
        dados: Any = None):
            self.tipo=tipo
            self.descricao=descricao
            
    def json(self):
        return json.dumps(
            {
                'tipo': self.tipo,
                'descricao': self.descricao
            }
        )

class MensagemInfo(Mensagem):
    def __init__(self,
        descricao: str,
        mensagem: str = None):
            Mensagem.__init__(self, 'info', descricao)
            self.mensagem=mensagem

    def json(self):
        return json.dumps(
            {
                'tipo': self.tipo,
                'descricao': self.descricao,
                'mensagem': self.mensagem
            }
        )

class MensagemErro(Mensagem):
    def __init__(self,
        descricao: str,
        mensagem: str = None):
            Mensagem.__init__(self, 'erro', descricao)
            self.mensagem=mensagem

    def json(self):
        return json.dumps(
            {
                'tipo': self.tipo,
                'descricao': self.descricao,
                'mensagem': self.mensagem
            }
        )

class MensagemControle(Mensagem):
    def __init__(self,
        descricao: str,
        dados: Any = None,
        mensagem: str = None):
            Mensagem.__init__(self, 'controle', descricao)
            self.mensagem=mensagem
            self.dados=dados

    def json(self):
        return json.dumps(
            {
                'tipo': self.tipo,
                'descricao': self.descricao,
                'mensagem': self.mensagem,
                'dados': self.dados
            }
        )

class MensagemDados(Mensagem):
    def __init__(self,
        descricao: str,
        dados: Any,
        mensagem: str = None):
            Mensagem.__init__(self, 'dados', descricao)
            self.mensagem=mensagem
            self.dados=dados

    def json(self):
        return json.dumps(
            {
                'tipo': self.tipo,
                'descricao': self.descricao,
                'mensagem': self.mensagem,
                'dados': self.dados
            }
        )
