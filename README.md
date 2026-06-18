# Previsão da Copa do Mundo da FIFA de 2026

O objetivo do exercício é construir um sistema probabilístico para simular a Copa do Mundo de 2026, estimando não apenas o resultado de partidas individuais, mas também as probabilidades de cada seleção avançar de fase, chegar às oitavas, quartas, semifinal, final e conquistar o título. Para isso, foram considerados dados históricos de partidas internacionais, ranking FIFA, medidas dinâmicas de força das seleções e, em uma das abordagens, probabilidades derivadas do mercado (odds de casas de apostas).

## Metodologia

A modelagem foi organizada em três versões principais: 

- A primeira versão é o modelo Poisson puro. Nesse caso, o modelo não prevê diretamente vitória, empate ou derrota. Ele prevê a quantidade esperada de gols de cada equipe em uma partida. Assim, para um jogo entre duas seleções, o modelo estima dois parâmetros: o número esperado de gols do primeiro time e o número esperado de gols do segundo time. A partir desses valores, os gols simulados são sorteados por distribuições de Poisson. Por exemplo, se o modelo estima que uma equipe tem média esperada de 1,6 gol e a outra de 0,9 gol, cada simulação gera um placar possível, como 1 a 0, 2 a 1, 0 a 0 etc. 

- A segunda versão é uma extensão probabilística do modelo Poisson. Nela, os gols esperados continuam sendo estimados pelo modelo, mas, em vez de sortear diretamente um placar, calcula-se a probabilidade de cada placar possível dentro de um intervalo, por exemplo de 0 a 10 gols para cada seleção. Com isso, é possível montar uma matriz de placares possíveis. Cada célula dessa matriz representa a probabilidade de um placar específico, como 0 a 0, 1 a 0, 1 a 1, 2 a 1 e assim por diante. A partir dessa matriz, somam-se as probabilidades dos placares em que o primeiro time vence, dos placares em que há empate e dos placares em que o segundo time vence. Dessa forma, o modelo Poisson passa a gerar também probabilidades de vitória, empate e derrota para cada partida.

- A terceira versão combina as probabilidades geradas pelo modelo Poisson com probabilidades derivadas de casa de apostas (odds). Essa adaptação foi necessária porque as odds disponíveis são apenas para os jogos da Copa do Mundo de 2026, e não para todo o histórico utilizado no treinamento. Por isso, não seria adequado usar as odds como variáveis comuns no treinamento do modelo, pois o modelo praticamente não aprenderia seu efeito histórico. Em vez disso, as odds são usadas como uma camada externa de calibração. Primeiro, o modelo Poisson estima as probabilidades estatísticas da partida com base no histórico, no ranking FIFA, na força ofensiva, na força defensiva, na forma recente e no rating dinâmico das seleções. Depois, essas probabilidades são combinadas com as odds. Por exemplo, nossa composição final tem que 60% da probabilidade final vem do modelo estatístico e 40% vem das probabilidades de mercado.

Essa terceira abordagem mantém a estrutura estatística do modelo Poisson, mas incorpora uma informação externa relevante. As odds funcionam como uma síntese das expectativas de mercado antes dos jogos. Assim, o modelo final não depende exclusivamente do histórico, mas também considera uma avaliação agregada disponível antes da partida.

## Monte Carlo

Com esse procedimento, é possível gerar probabilidades em diferentes níveis. No nível da partida, é possível estimar a probabilidade de vitória, empate e derrota de cada seleção. No nível da competição, é possível simular milhares de Copas do Mundo e calcular a frequência com que cada seleção avança de fase, chega ao mata-mata, às oitavas, quartas, semifinal, final e vence o torneio. Assim, se o Brasil for campeão em 1.200 de 10.000 simulações de monte carlo, sua probabilidade estimada de título será de 12%.
