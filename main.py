import pandas as pd #biblioteca para criar e modificar dataframes
import numpy as np #biblioteca para trabalhar com vetores e matrizes
from sklearn import tree #biblioteca para criar e treinar árvores de decisão
from sklearn.metrics import accuracy_score #biblioteca para calcular a precisão da árvore de decisão
from sklearn.model_selection import train_test_split #biblioteca para dividir os dados em conjuntos de treinamento e teste
from sklearn.tree import DecisionTreeClassifier #biblioteca para criar um modelo de árvore de decisão
import os #biblioteca de comandos do sistema
import matplotlib.pyplot as plt #biblioteca para criar gráficos
import time #biblioteca para trabalhar com tempo
df = pd.read_csv("hepatitis.csv", na_values='?') #lê o arquivo csv e preenche os valores ausentes com '?'

# trata os valores ausentes
df.fillna(df.mean(axis=0), inplace=True)  # Preencher valores ausentes com a média das colunas numéricas
df.fillna(df.mode(axis=0).iloc[0], inplace=True)  # Preencher valores ausentes com a moda das colunas categóricas
df.rename(columns={
    'Class': 'Vivo/Morto',
    'Age': 'Idade',
    'Sex': 'Sexo',
    'Steroids': 'Analgésicos',
    'Antivirals': 'Antivirais',
    'Fatigue': 'Fadiga',
    'Malaise': 'Mal-estar',
    'Anorexia': 'Anorexia',
    'Liver Big': 'Fígado grande',
    'Liver Firm': 'Fígado firme',
    'Spleen Palpable': 'Baço palpável',
    'Spiders': 'Aranhas', #círculo de veia pulsante
    'Ascites': 'Ascite', #acumulo de liquido dentro do abdômen
    'Varices': 'Varizes',
    'Bilirubin': 'Bilirrubina', #substancia amarelada encontrada na bile
    'Alk phosphate': 'Fosfatase alcalina', #enzimas hidrolases na membrana celular
    'Sgot': 'TGO', #transaminase glutamico-oxalacetica, enzima que converte nitrogenio pra aminoacido
    'Albumin': 'Albumina', #proteinas globulares soluveis em agua, encontradas no plasma
    'Histology': 'Histologia', #estuda funções da celula.
}, inplace=True) #renomeia as colunas do dataframe

def mostrarDataframe(): # função para mostrar o dataframe
    print(df)
    voltarMenu()

def calcularPrecisao(): # função que calcula a precisão da árvore de decisão
    X = df.drop('Vivo/Morto', axis=1) #prepara os dados X e y
    y = df['Vivo/Morto']

    # divide os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  #80% para treinamento e 20% para teste

    model = DecisionTreeClassifier(random_state=42)  # cria um modelo de árvore de decisão de classificação
    model.fit(X_train, y_train)  # treina o modelo com os dados de treinamento
    y_pred = model.predict(X_test)  # faz previsões com os dados de teste
    accuracy = accuracy_score(y_test, y_pred) #calcula a precisão da ávore de decisão

    print(f'Precisão da árvore de decisão: {accuracy:.2f}')

    voltarMenu() #pergunta ao usuário se ele quer voltar para o menu principal do programa

def desenharArvore(): # função para desenhar a árvore de decisão
    X = df.drop('Vivo/Morto', axis=1) #prepara os dados X e y
    y = df['Vivo/Morto']

    model = DecisionTreeClassifier(random_state=42) # cria um modelo de árvore de decisão de classificação
    model.fit(X, y) # treina o modelo com os dados

    plt.figure(figsize=(20,10)) #cria um desenho da árvore de decisão
    tree.plot_tree(model, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
    plt.show() #imprime o desenho da árvore
    
    time.sleep(1) #pausa o programa por 1 segundo
    voltarMenu() #pergunta ao usuário se ele quer voltar para o menu principal do programa

def adicionarDado(): # função para adicionar um novo dado ao dataframe
    novo_dado = {}
    for coluna in df.columns: #para cada coluna dentro do dataframe:
        if coluna == 'Vivo/Morto':
            valor = input(f"Digite o valor para a coluna '{coluna}' (0 para 'Não', 1 para 'Sim'): ")
        else:
            valor = input(f"Digite o valor para a coluna '{coluna}': ")
        novo_dado[coluna] = valor

    df.loc[len(df)] = novo_dado # adiciona o novo dado ao dataframe
    print("Novo dado adicionado com sucesso!")
    voltarMenu() #pergunta ao usuário se ele quer voltar para o menu principal do programa

def voltarMenu(): # função que pergunta ao usuário se ele quer voltar para o menu inicial
    continuar = input("\nDeseja voltar para o menu inicial?\nDigite 'sim' para continuar.\n").lower()
    if continuar == 'sim':
        os.system('cls')
        main() #volta para o menu principal
    else:
        print("Programa finalizado.")
        exit() #fecha o programa

def main(): # função principal
    print("--- Projeto de Banco de dados de Pacientes com Hepatite ---")
    print("1: Mostrar o Dataframe completo")
    print("2: Calcular a Precisão da árvore de decisão do dataframe")
    print("3: Desenho da árvore")
    print("4: Adicionar um dado novo")
    print("5: Sair do programa")
    escolha = int(input("Escolha uma opção: "))
    match escolha: #estrutura de condição match case
        case 1:
            mostrarDataframe()
        case 2:
            calcularPrecisao()
        case 3:
            desenharArvore()
        case 4:
            adicionarDado()
        case 5:
            print("Saindo do programa...")
            exit() #fecha o programa
    if(escolha <=0 or escolha > 6):
        print("Opção inválida")
        main() #volta pro menu

main() # Início do programa