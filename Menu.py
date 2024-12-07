import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Carregar o modelo treinado
model = load_model('modelo_sono.h5')

# Função para mostrar 3 resultados do dataset
def mostrar_resultados(dataset):
    print("Mostrando 3 resultados do dataset:")
    print(dataset.sample(3))

# Função para fazer a medição de um usuário
def medir_usuario():
    print("Digite os dados do usuário:")
    cafeina = float(input("Quantidade de cafeína ingerida (0 a 7 xícaras por dia): (Pode ser float)"))
    sono = float(input("Quantidade de horas de sono (4 a 10 horas): (Pode ser float)"))
    exercicio = int(input("Quantidade de minutos de exercício por semana (0 a 300 minutos): (Número inteiro)"))
    estresse = int(input("Nível de estresse (1 a 10): (Número inteiro)"))

    # Preparar os dados e fazer a predição
    user_data = np.array([[cafeina, sono, exercicio, estresse]])
    # Normalização com os dados máximos
    user_data = user_data / np.array([7, 10, 300, 10])  

    prediction = model.predict(user_data)
    qualidade = ['Ruim', 'Médio', 'Bom', 'Ótimo']
    print(f"A qualidade do sono do usuário é: {qualidade[np.argmax(prediction)]}")

# Função para comparar dois usuários
def comparar_usuarios():
    print("Digite os dados do primeiro usuário:")
    cafeina_1 = float(input("Quantidade de cafeína ingerida (0 a 7 xícaras por dia): (Pode ser float)"))
    sono_1 = float(input("Quantidade de horas de sono (4 a 10 horas): (Pode ser float)"))
    exercicio_1 = int(input("Quantidade de minutos de exercício por semana (0 a 300 minutos): (Número inteiro)"))
    estresse_1 = int(input("Nível de estresse (1 a 10): (Número inteiro)"))

    print("Digite os dados do segundo usuário:")
    cafeina_2 = float(input("Quantidade de cafeína ingerida (0 a 7 xícaras por dia): (Pode ser float)"))
    sono_2 = float(input("Quantidade de horas de sono (4 a 10 horas):(Pode ser float) "))
    exercicio_2 = int(input("Quantidade de minutos de exercício por semana (0 a 300 minutos): (Número inteiro)"))
    estresse_2 = int(input("Nível de estresse (1 a 10): (Número inteiro)"))

    # Preparar os dados e fazer as previsões
    user_data_1 = np.array([[cafeina_1, sono_1, exercicio_1, estresse_1]])
    user_data_2 = np.array([[cafeina_2, sono_2, exercicio_2, estresse_2]])

    # Normalização caso tenha feito no treino
    user_data_1 = user_data_1 / np.array([7, 10, 300, 10])
    user_data_2 = user_data_2 / np.array([7, 10, 300, 10])

    prediction_1 = model.predict(user_data_1)
    prediction_2 = model.predict(user_data_2)

    qualidade = ['Ruim', 'Médio', 'Bom', 'Ótimo']
    print(f"A qualidade do sono do primeiro usuário é: {qualidade[np.argmax(prediction_1)]}")
    print(f"A qualidade do sono do segundo usuário é: {qualidade[np.argmax(prediction_2)]}")

# Função principal para exibir o menu
def menu():
    dataset = pd.read_csv("qualidade_sono_dataset.csv")  # Carregar o dataset
    while True:
        print("\n===== Menu =====")
        print("1. Mostrar 3 resultados do dataset")
        print("2. Medir a qualidade do sono de um usuário")
        print("3. Comparar dois usuários")
        print("4. Sair")
        escolha = input("Escolha uma opção (1-4): ")

        if escolha == '1':
            mostrar_resultados(dataset)
        elif escolha == '2':
            medir_usuario()
        elif escolha == '3':
            comparar_usuarios()
        elif escolha == '4':
            print("Saindo...")
            break
        else:
            print("Opção inválida. Tente novamente.")

# Executar o menu
if __name__ == "__main__":
    menu()