unzip api/testes/resultados/testes_automatizados.json.zip -d api/testes/resultados/
rm api/testes/resultados/testes_automatizados.json.zip

unzip api/conteudo/conteudo.zip -d api/conteudo/
rm api/conteudo/conteudo.zip

cp api/.env.TEMPLATE api/.env
