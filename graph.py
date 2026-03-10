from algs4.bag import Bag
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from collections import Counter

class Graph:

    def __init__(self, v):
        self.V = v
        self.E = 0
        self.map = {}
        self.adj = []
        
    def __str__(self):
        s = "%d vertices, %d edges\n" % (self.V, self.E)
        s += "\n".join("%d: %s" % (v, " ".join(str(w)
                                               for w in self.adj[v])) for v in range(self.V))
        return s

    def add_mapping(self, node_value: int) -> None:
        self.map[node_value] = len(self.adj)
        self.adj.append(Bag())

    def get_position(self, node_value:int) -> int:
        return self.map[node_value]

    def get_value(self, node_position:int) -> int:
        for key, value in self.map.items():
            if value == node_position:
                return key

    def add_edge(self, v, w, directed: bool = False):
        if v not in self.map:
            self.add_mapping(v)
        if w not in self.map:
            self.add_mapping(w)
        if directed:
            if w in self.adj[self.get_position(v)]:
                return
        v_position = self.get_position(v)
        w_position = self.get_position(w)
        self.adj[v_position].add(w)
        self.adj[w_position].add(v)
        self.E += 1

    def degree(self, v, from_value: bool = True) -> int:
        v_position = self.get_position(v) if from_value else v 
        return self.adj[v_position].size()
    
    def density(self):
        print(2* self.E, self.V * (self.V - 1))
        return (2 * self.E) / (self.V * (self.V - 1))

    def add_node_from_file(self, file_path: str = "data.txt") -> None:
        with open(file_path, 'r') as file:
            linhas = file.readlines()
        for linha in linhas:
            linha_tratada = linha.replace("\t"," ").replace("\n","").split(" ")
            no_origem, no_destino = map(int, linha_tratada)
            self.add_edge(no_origem, no_destino)

    def print_mapping(self, len:int = 10) -> None:
        for i, (key, value) in enumerate(self.map.items()):
            if i >= len:
                break
            print(f"{key} -> {value}")

    def print_adjacency_list(self, len:int = 10) -> None:
        print(f"Vértices(Ordem): {self.V}; Arestas(Tamanho): {self.E}; Densidade: {self.density():.4f} ({'Denso' if self.density() >= 0.8 else 'Esparso'}); Grau mínimo: {self.min_degree()}; Grau máximo: {self.max_degree()}")
        for i in range(min(len, self.V)):
            print(f"{self.get_value(i)}: {self.adj[i]} - Grau: {self.degree(self.get_value(i))}")
    
    def save_graph(self, len:int = 10, save_path: str = "graph.png") -> None:
        G = nx.Graph()
        for v, neighbors in enumerate(self.adj):
            if v >= len:
                break
            for w in neighbors:
                G.add_edge(self.get_value(v), w)
        plt.figure(figsize=(10, 8))  # Ajusta o tamanho da figura
        nx.draw(G, with_labels=True, node_color='lightblue', font_weight='bold')
        plt.savefig(save_path)  # Salva a imagem no caminho especificado

    def print_degree_for_value(self, node_value: int) -> None:
        degree = self.degree(node_value)
        print(f"Grau do vértice {node_value}: {degree}")

    def save_histogram_degree(self, save_path: str = "histogram.png") -> None:
        degrees = [self.degree(v) for v in self.map.keys()]
        plt.figure(figsize=(10, 6))
        plt.hist(degrees, bins=range(max(degrees) + 1), color='lightblue', edgecolor='black')
        plt.title('Histograma de Graus dos Vértices')
        plt.xlabel('Grau')
        plt.ylabel('Frequência')
        plt.savefig(save_path)

    def min_degree(self) -> int:
        return min(self.degree(v) for v in self.map.keys())

    def max_degree(self) -> int:
        return max(self.degree(v) for v in self.map.keys())

    def get_degree_distribution(self):
        """Retorna a distribuição de graus: dicionário {grau: frequência}"""
        degrees = [self.degree(v) for v in self.map.keys()]
        return dict(Counter(degrees))
    
    def fit_power_law(self):
        """
        Ajusta uma lei de potência P(k) ~ k^(-gamma) à distribuição de graus.
        Retorna: (gamma, k_min, r_squared, degree_distribution)
        """
        # Obter distribuição de graus
        degree_dist = self.get_degree_distribution()
        
        # Ordenar por grau
        degrees = np.array(sorted(degree_dist.keys()))
        frequencies = np.array([degree_dist[k] for k in degrees])
        
        # Normalizar para obter probabilidade
        probabilities = frequencies / frequencies.sum()
        
        # Filtrar zeros e valores muito baixos para evitar problemas no log
        mask = (probabilities > 0) & (degrees > 0)
        degrees_filtered = degrees[mask]
        probabilities_filtered = probabilities[mask]
        
        # Definir função de lei de potência: P(k) = C * k^(-gamma)
        def power_law(k, C, gamma):
            return C * k**(-gamma)
        
        # Ajustar usando mínimos quadrados no espaço logarítmico
        # log(P(k)) = log(C) - gamma * log(k)
        log_k = np.log(degrees_filtered)
        log_p = np.log(probabilities_filtered)
        
        # Regressão linear no espaço log-log
        coeffs = np.polyfit(log_k, log_p, 1)
        gamma = -coeffs[0]  # O coeficiente angular negativo é gamma
        log_C = coeffs[1]
        C = np.exp(log_C)
        
        # Calcular R² (coeficiente de determinação)
        log_p_pred = log_C - gamma * log_k
        ss_res = np.sum((log_p - log_p_pred)**2)
        ss_tot = np.sum((log_p - log_p.mean())**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Grau mínimo considerado (normalmente k_min é onde a lei de potência começa)
        k_min = degrees_filtered.min()
        
        return {
            'gamma': gamma,
            'C': C,
            'k_min': k_min,
            'r_squared': r_squared,
            'degrees': degrees_filtered,
            'probabilities': probabilities_filtered,
            'degrees_all': degrees,
            'probabilities_all': probabilities
        }
    
    def save_power_law_plot(self, save_path: str = "power_law_fit.png") -> dict:
        """
        Gera e salva o gráfico log-log da distribuição de graus com o ajuste de lei de potência.
        Retorna os parâmetros do ajuste.
        """
        # Ajustar lei de potência
        fit_results = self.fit_power_law()
        gamma = fit_results['gamma']
        C = fit_results['C']
        r_squared = fit_results['r_squared']
        degrees = fit_results['degrees']
        probabilities = fit_results['probabilities']
        
        # Criar gráfico log-log
        plt.figure(figsize=(10, 7))
        
        # Plot dos dados observados
        plt.loglog(degrees, probabilities, 'bo', alpha=0.6, markersize=8, label='Dados observados')
        
        # Plot da lei de potência ajustada
        k_fit = np.linspace(degrees.min(), degrees.max(), 100)
        p_fit = C * k_fit**(-gamma)
        plt.loglog(k_fit, p_fit, 'r-', linewidth=2, 
                   label=f'Lei de potência: $P(k) \\sim k^{{-\\gamma}}$\n$\\gamma = {gamma:.3f}$\n$R^2 = {r_squared:.4f}$')
        
        plt.xlabel('Grau (k)', fontsize=12)
        plt.ylabel('P(k)', fontsize=12)
        plt.title('Distribuição de Graus - Ajuste de Lei de Potência', fontsize=14)
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3, which='both')
        
        # Adicionar anotação com interpretação
        interpretation = self.interpret_gamma(gamma)
        plt.text(0.02, 0.02, interpretation, transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return fit_results
    
    def interpret_gamma(self, gamma: float) -> str:
        """
        Interpreta o valor do expoente gamma da lei de potência.
        """
        if gamma < 2:
            return f" γ > 2 Rede extremamente concentrada em hubs."
        elif 2 <= gamma < 3:
            return f" 2<=γ<3 A maioria dos nós tem poucas conexões"
        elif 3 <= gamma < 4:
            return f"3 ≤ γ < 4: Rede com hubs menos dominantes, mas ainda presentes."
        else:
            return f"γ ≥ 4: Rede sem hubs bem definidos, com comportamento semelhante a uma rede aleatória."
    #https://www.pnas.org/doi/10.1073/pnas.202301299#:~:text=7%2C%20where%20vertices%20represent%20substrates,introduced%20by%20Sol%C3%A9%20et%20al.
    def print_power_law_analysis(self):
        """
        Imprime análise detalhada do ajuste de lei de potência.
        """
        fit_results = self.fit_power_law()
        gamma = fit_results['gamma']
        r_squared = fit_results['r_squared']
        k_min = fit_results['k_min']
        
        print("\n" + "="*60)
        print("ANÁLISE DE LEI DE POTÊNCIA")
        print("="*60)

        print(f"\nExpoente (γ): {gamma:.4f}")
        print(f"Coeficiente de determinação (R²): {r_squared:.4f}")
        print(f"Grau mínimo considerado (k_min): {k_min}")

        print(f"\nInterpretação do γ: {self.interpret_gamma(gamma)}")

        print("\nIndício de comportamento de lei de potência:")

        if r_squared >= 0.9:
            print("  Forte indício de lei de potência")

        elif r_squared >= 0.8:
            print("  Indício razoável de lei de potência")

        elif r_squared >= 0.7:
            print("  Possível comportamento de lei de potência")

        else:
            print("  Os dados não sugerem claramente uma lei de potência")

        print("\nObs: O R² varia entre 0 e 1.")
        print("0 indica que o modelo não explica a variação dos dados.")
        print("1 indica que o modelo explica totalmente a variação.")
        print("Quanto mais próximo de 1, melhor o ajuste do modelo aos dados.")
# https://www.datacamp.com/tutorial/coefficient-of-determination abaixo do titulo "What Is the Coefficient of Determination?""