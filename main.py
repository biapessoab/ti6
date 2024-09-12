import requests

# Ler o resultado da classificação
with open('predicted_classes.txt', 'r') as f:
    predicted_class = f.read().strip()

# Configurar a URL e parâmetros da API
url = "https://api.edamam.com/api/recipes/v2"
params = {
    'type': 'public',
    'q': predicted_class,  # Usar a classe prevista como ingrediente
    'app_id': '03ef7e2d',
    'app_key': 'dc7be51f1bdd4c32ca45288155a1d928',
    'to': 10
}

# Fazer a requisição para a API
response = requests.get(url, params=params)
dados = response.json()

# Exibir os resultados
if 'hits' in dados and dados['hits']:
    print("Receitas encontradas:")
    for item in dados['hits']:
        receita = item['recipe']
        nome = receita['label']
        print(f"- {nome}")
else:
    print("Nenhuma receita encontrada.")
