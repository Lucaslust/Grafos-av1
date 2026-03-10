from graph import Graph

NUMERO_VERTICES = 5242
NUMERO_ARESTAS = 28980

if __name__ == "__main__":
    grafo = Graph(NUMERO_VERTICES)
    grafo.add_node_from_file("data.txt")
    grafo.print_adjacency_list()
    grafo.save_graph(save_path="graph_10.png")
    grafo.save_histogram_degree()
    
    # Análise de lei de potência
    print("\n" + "="*60)
    print("GERANDO ANÁLISE DE LEI DE POTÊNCIA...")
    print("="*60)
    
    # Gerar gráfico log-log com ajuste
    fit_results = grafo.save_power_law_plot(save_path="power_law_fit.png")
    
    # Imprimir análise detalhada
    grafo.print_power_law_analysis()
    
    # Salvar resultados em arquivo de texto
    with open("power_law_results.txt", "w", encoding="utf-8") as f:
        f.write("RESULTADOS DO AJUSTE DE LEI DE POTÊNCIA\n")
        f.write("="*60 + "\n\n")
        f.write(f"Expoente (γ): {fit_results['gamma']:.4f}\n")
        f.write(f"Constante de normalização (C): {fit_results['C']:.6e}\n")
        f.write(f"Coeficiente de determinação (R²): {fit_results['r_squared']:.4f}\n")
        f.write(f"Grau mínimo considerado (k_min): {fit_results['k_min']}\n")
        f.write(f"\nInterpretação do expoente γ = {fit_results['gamma']:.4f}:\n")
        f.write(grafo.interpret_gamma(fit_results['gamma']) + "\n")
    
    print("\n✓ Gráfico salvo em: power_law_fit.png")
    print("✓ Resultados salvos em: power_law_results.txt")


""" 
TODO:
1. Checar por que só aparece 1 aresta na visualização mesmo tendo duas conexões no grafo 
""" 
