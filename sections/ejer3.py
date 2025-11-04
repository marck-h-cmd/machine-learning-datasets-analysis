import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

def main():
    st.header("Ejercicio 3: Dataset 'Iris'")
    
    st.markdown("""
    ### Objetivo
    Implementar un flujo completo de preprocesamiento y visualizar resultados.
    """)
    
    # Cargar dataset desde sklearn
    st.subheader("1️⃣ Carga del Dataset desde scikit-learn")
    
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['target_name'] = df['target'].apply(lambda x: iris.target_names[x])
    
    st.success("✅ Dataset Iris cargado desde sklearn.datasets")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Dimensiones:** {df.shape[0]} filas × {df.shape[1]} columnas")
        st.write(f"**Características:** {len(iris.feature_names)}")
        st.write(f"**Clases:** {len(iris.target_names)}")
    with col2:
        st.write("**Nombres de características:**")
        for i, name in enumerate(iris.feature_names):
            st.write(f"{i+1}. {name}")
        st.write("**Clases:**")
        for i, name in enumerate(iris.target_names):
            st.write(f"- {name} (código: {i})")
    
    st.write("**Primeras filas del dataset:**")
    st.dataframe(df.head(10))
    
    # Conversión a DataFrame y agregar nombres de columnas
    st.subheader("2️⃣ Conversión a DataFrame con Nombres de Columnas")
    
    st.write("✅ El dataset ya está convertido a DataFrame con nombres de columnas")
    st.write("**Estructura del DataFrame:**")
    buffer = StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)
    
    # Estadísticas descriptivas
    st.write("**Estadísticas descriptivas:**")
    st.dataframe(df.describe())
    
    st.write("**Distribución de clases:**")
    class_dist = df['target_name'].value_counts()
    st.dataframe(class_dist.to_frame('Cantidad'))
    
    # Estandarización con StandardScaler
    st.subheader("3️⃣ Estandarización con StandardScaler")
    
    # Separar características y target
    X = df[iris.feature_names].copy()
    y = df['target'].copy()
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=iris.feature_names,
        index=X.index
    )
    
    st.success("✅ Estandarización completada")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Antes de estandarización:**")
        st.dataframe(X.describe())
    
    with col2:
        st.write("**Después de estandarización:**")
        st.dataframe(X_scaled.describe())
    
    # División del dataset
    st.subheader("4️⃣ División del Dataset (70% entrenamiento, 30% prueba)")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    st.success("✅ Datos divididos en 70% entrenamiento y 30% prueba")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Conjunto de Entrenamiento:**")
        st.write(f"- Filas: {X_train.shape[0]}")
        st.write(f"- Columnas: {X_train.shape[1]}")
        st.write("**Distribución de clases:**")
        train_dist = pd.Series(y_train).map({i: name for i, name in enumerate(iris.target_names)}).value_counts()
        st.dataframe(train_dist.to_frame('Cantidad'))
    
    with col2:
        st.write("**Conjunto de Prueba:**")
        st.write(f"- Filas: {X_test.shape[0]}")
        st.write(f"- Columnas: {X_test.shape[1]}")
        st.write("**Distribución de clases:**")
        test_dist = pd.Series(y_test).map({i: name for i, name in enumerate(iris.target_names)}).value_counts()
        st.dataframe(test_dist.to_frame('Cantidad'))
    
    # Gráfico de dispersión
    st.subheader("5️⃣ Visualización: Gráfico de Dispersión")
    
    st.markdown("""
    **Gráfico de dispersión: Sepal Length vs Petal Length diferenciado por clase**
    """)
    
    # Crear DataFrame para el gráfico
    plot_df = X_scaled.copy()
    plot_df['target'] = df['target'].values
    plot_df['target_name'] = df['target_name'].values
    
    # Crear el gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['red', 'blue', 'green']
    target_names = iris.target_names
    
    for i, (target, name) in enumerate(zip(range(len(target_names)), target_names)):
        mask = plot_df['target'] == target
        ax.scatter(
            plot_df.loc[mask, 'sepal length (cm)'],
            plot_df.loc[mask, 'petal length (cm)'],
            c=colors[i],
            label=name,
            alpha=0.7,
            s=50
        )
    
    ax.set_xlabel('Sepal Length (cm) [Estandarizado]', fontsize=12)
    ax.set_ylabel('Petal Length (cm) [Estandarizado]', fontsize=12)
    ax.set_title('Distribución de Sepal Length vs Petal Length por Clase', fontsize=14, fontweight='bold')
    ax.legend(title='Clase', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Gráfico adicional: todas las combinaciones de características
    if st.checkbox("Mostrar gráficos adicionales de todas las características"):
        st.write("**Matriz de dispersión de todas las características:**")
        
        plot_df_vis = X_scaled.copy()
        plot_df_vis['target_name'] = df['target_name'].values
        
        # sns.pairplot crea su propia figura, así que no necesitamos crear una antes
        pair_grid = sns.pairplot(plot_df_vis, hue='target_name', diag_kind='hist', palette=['red', 'blue', 'green'], height=2.5)
        pair_grid.fig.suptitle('Matriz de Dispersión - Todas las Características', y=1.02, fontsize=16)
        plt.tight_layout()
        st.pyplot(pair_grid.fig)
    
    # Estadísticas descriptivas del dataset estandarizado
    st.subheader("📊 Salida Esperada")
    
    st.write("**Estadísticas descriptivas del dataset estandarizado:**")
    st.dataframe(X_scaled.describe())
    
    st.write("**Resumen de las operaciones realizadas:**")
    st.code(f"""
Dataset Iris:
- Total de muestras: {df.shape[0]}
- Características: {len(iris.feature_names)}
- Clases: {len(iris.target_names)}

Después de estandarización:
- Media de cada característica: ~0
- Desviación estándar de cada característica: ~1

División de datos:
- X_train: {X_train.shape} ({X_train.shape[0]/len(df)*100:.1f}%)
- X_test: {X_test.shape} ({X_test.shape[0]/len(df)*100:.1f}%)
- y_train: {y_train.shape}
- y_test: {y_test.shape}

Distribución balanceada por clase en ambos conjuntos.
    """)
    
    # Información adicional
    st.markdown("---")
    st.info("""
    💡 **Nota:** El dataset Iris es ideal para aprendizaje porque:
    - Es pequeño y manejable
    - Tiene características bien definidas
    - Las clases están balanceadas
    - No contiene valores nulos
    - Es perfecto para visualización
    """)

