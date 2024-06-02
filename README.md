# IA que ler seus PDFs

IA capaz de ler e subir os dados do PDF para o banco de dados pessoal, com isso o Ollama busca dentre os melhores resultados compativeis no banco de dados para conseguir responder sua pergunta,
exemplo nesse teste perguntei qual era o nível de testoterona e teve o resultado apresentado. Os PDFs podem ser aadicionados depois, conforme você tiver mais, não vai ser preciso preciso carregar
sempre o banco de dados, pois no codigo é divididp chunks com 1200 caracteres e para cada chunk é dado um ID próprio que sempre vai identificar se já tem aquela informação quando for atualizado o banco 
de dados para que não tenha consumo de processamento.

## 💻 Projeto

- Fazer perguntas sobre o PDF
- Expansão de dados
- Banco de dados próprio

## 👨‍💻 Tecnologias Utilizadas

Utilizando apenas **PYTHON** e as bibliotecas:
> - IA ollama3 8b
> - Chroma DB
> - Langchain
