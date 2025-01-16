#5. Abrir mascaras manualess sacar medidas de circularidad, area, textura etc

#6. hacer lo mismo para las automaticas


#7. sacar las medidas de correlacion esntre las carateristicas


#8 guardar resultados e imprimirlos en el paper

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# Simulación de características de glándulas
# Conjunto manual (menos glándulas)
manual_glands = {
    "area": np.random.normal(loc=500, scale=50, size=50),
    "circularity": np.random.uniform(0.7, 1.0, size=50),
    "texture": np.random.normal(loc=100, scale=10, size=50),
}

# Conjunto automático (más glándulas)
automatic_glands = {
    "area": np.random.normal(loc=510, scale=60, size=150),
    "circularity": np.random.uniform(0.65, 1.0, size=150),
    "texture": np.random.normal(loc=105, scale=15, size=150),
}

# Convertir a DataFrame
manual_df = pd.DataFrame(manual_glands)
automatic_df = pd.DataFrame(automatic_glands)


# Función para calcular correlaciones y KS Test
def evaluate_correlation(manual_df, automatic_df):
    results = []

    for feature in manual_df.columns:
        # Correlación de Pearson y Spearman
        pearson_corr, p_pearson = stats.pearsonr(manual_df[feature], automatic_df[feature][:len(manual_df)])
        spearman_corr, p_spearman = stats.spearmanr(manual_df[feature], automatic_df[feature][:len(manual_df)])

        # Prueba KS para distribuciones
        ks_stat, p_ks = stats.ks_2samp(manual_df[feature], automatic_df[feature])

        # Guardar resultados
        results.append({
            "Feature": feature,
            "Pearson Correlation": pearson_corr,
            "Pearson p-value": p_pearson,
            "Spearman Correlation": spearman_corr,
            "Spearman p-value": p_spearman,
            "KS Statistic": ks_stat,
            "KS p-value": p_ks,
        })

    return pd.DataFrame(results)


# Evaluar correlación
correlation_results = evaluate_correlation(manual_df, automatic_df)


# Mostrar resultados
def display_results(df):
    print("\nResultados de Correlación y Pruebas KS:\n")
    print(df.round(3))

    # Visualización rápida
    for feature in df['Feature']:
        plt.figure()
        plt.hist(manual_df[feature], bins=15, alpha=0.5, label='Manual')
        plt.hist(automatic_df[feature], bins=15, alpha=0.5, label='Automático')
        plt.title(f"Distribución de {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frecuencia")
        plt.legend()
        plt.show()


# Mostrar los resultados y gráficas
display_results(correlation_results)
