import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.header("📚 Ejercicio 2: Procesamiento del Dataset 'Student Performance'")
    
    st.markdown("""
    ### Objetivo
    Procesar los datos para un modelo que prediga la nota final (G3) de los estudiantes.
    """)
    
    # Cargar dataset
    st.subheader("1️⃣ Carga del Dataset")
    
    uploaded_file = st.file_uploader("Cargar archivo student-mat.csv", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, sep=';')  # El dataset de student performance usa ; como separador
        st.success("✅ Dataset cargado exitosamente")
    else:
        st.warning("⚠️ Por favor, carga el archivo student-mat.csv")
        st.info("💡 El dataset 'Student Alcohol Consumption' está disponible en Kaggle")
        return
    
    # Mostrar información inicial
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Dimensiones:** {df.shape[0]} filas × {df.shape[1]} columnas")
    with col2:
        st.write(f"**Columnas:** {len(df.columns)}")
    
    st.write("**Primeras filas:**")
    st.dataframe(df.head())
    
    # Exploración inicial
    st.subheader("2️⃣ Exploración Inicial")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Tipos de datos:**")
        types_df = pd.DataFrame({
            'Columna': df.columns,
            'Tipo': [str(dtype) for dtype in df.dtypes]
        })
        st.dataframe(types_df)
        
        st.write("**Valores nulos:**")
        nulls = df.isnull().sum()
        nulls_df = nulls[nulls > 0].to_frame('Valores Nulos')
        if len(nulls_df) > 0:
            st.dataframe(nulls_df)
        else:
            st.success("✅ No hay valores nulos")
    
    with col2:
        st.write("**Variables categóricas:**")
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        st.write(f"Total: {len(categorical_cols)}")
        st.write(", ".join(categorical_cols[:10]))
        if len(categorical_cols) > 10:
            st.write(f"... y {len(categorical_cols) - 10} más")
        
        st.write("**Variables numéricas:**")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        st.write(f"Total: {len(numeric_cols)}")
        st.write(", ".join(numeric_cols))
    
    st.write("**Estadísticas descriptivas:**")
    st.dataframe(df.describe())
    
    # Análisis de variables categóricas
    if st.checkbox("Mostrar análisis detallado de variables categóricas"):
        st.write("**Distribución de variables categóricas:**")
        for col in categorical_cols[:5]:  # Mostrar solo las primeras 5
            st.write(f"**{col}:**")
            st.dataframe(df[col].value_counts())
    
    # Limpieza de datos
    st.subheader("3️⃣ Limpieza de Datos")
    
    df_clean = df.copy()
    
    # Eliminar duplicados
    st.write("**Eliminando duplicados...**")
    duplicates = df_clean.duplicated().sum()
    if duplicates > 0:
        df_clean = df_clean.drop_duplicates()
        st.success(f"✅ {duplicates} duplicados eliminados")
    else:
        st.info("ℹ️ No se encontraron duplicados")
    
    # Verificar valores inconsistentes
    st.write("**Verificando valores inconsistentes...**")
    
    # Verificar que G1, G2, G3 están en el rango esperado (0-20 típicamente)
    grade_cols = ['G1', 'G2', 'G3']
    inconsistent_rows = []
    for col in grade_cols:
        if col in df_clean.columns:
            # Buscar valores fuera del rango 0-20 (ajustar según el dataset)
            invalid = (df_clean[col] < 0) | (df_clean[col] > 20)
            if invalid.any():
                inconsistent_rows.extend(df_clean[invalid].index.tolist())
    
    if inconsistent_rows:
        df_clean = df_clean.drop(index=list(set(inconsistent_rows)))
        st.success(f"✅ {len(set(inconsistent_rows))} filas con valores inconsistentes eliminadas")
    else:
        st.info("ℹ️ No se encontraron valores inconsistentes")
    
    st.write(f"**Dimensiones después de la limpieza:** {df_clean.shape[0]} filas × {df_clean.shape[1]} columnas")
    
    # Análisis de correlación entre G1, G2, G3 (Reto adicional)
    st.subheader("🔍 Reto Adicional: Análisis de Correlación")
    
    grade_cols = ['G1', 'G2', 'G3']
    existing_grade_cols = [col for col in grade_cols if col in df_clean.columns]
    
    if len(existing_grade_cols) >= 2:
        corr_matrix = df_clean[existing_grade_cols].corr()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Matriz de correlación:**")
            st.dataframe(corr_matrix)
        
        with col2:
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=1, ax=ax)
            plt.title('Correlación entre G1, G2 y G3')
            st.pyplot(fig)
    
    # Codificación de variables categóricas (One Hot Encoding)
    st.subheader("4️⃣ Codificación de Variables Categóricas (One Hot Encoding)")
    
    df_encoded = df_clean.copy()
    
    # Aplicar One Hot Encoding a todas las variables categóricas
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        st.write(f"**Aplicando One Hot Encoding a {len(categorical_cols)} variables categóricas...**")
        df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, prefix=categorical_cols, drop_first=True)
        st.success(f"✅ Codificación completada. Nuevas dimensiones: {df_encoded.shape}")
        st.write(f"**Columnas después de One Hot Encoding:** {df_encoded.shape[1]}")
    else:
        st.info("ℹ️ No se encontraron variables categóricas")
    
    st.write("**Primeras columnas después de la codificación:**")
    st.dataframe(df_encoded.head())
    
    # Normalización de variables numéricas
    st.subheader("5️⃣ Normalización de Variables Numéricas")
    
    # Identificar columnas numéricas a normalizar (excluyendo G3 que es el target)
    numeric_cols_to_normalize = []
    target_cols = ['G3']  # Variable objetivo
    
    for col in df_encoded.select_dtypes(include=[np.number]).columns:
        if col not in target_cols:
            numeric_cols_to_normalize.append(col)
    
    if numeric_cols_to_normalize:
        # Usar MinMaxScaler para normalización (valores entre 0 y 1)
        scaler = MinMaxScaler()
        df_normalized = df_encoded.copy()
        
        original_values = df_normalized[numeric_cols_to_normalize].copy()
        df_normalized[numeric_cols_to_normalize] = scaler.fit_transform(
            df_normalized[numeric_cols_to_normalize]
        )
        
        st.write(f"**Variables normalizadas:** {', '.join(numeric_cols_to_normalize[:5])}...")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Antes de normalización (muestra):**")
            st.dataframe(original_values[['age', 'absences', 'G1', 'G2']].describe() 
                        if all(col in original_values.columns for col in ['age', 'absences', 'G1', 'G2']) 
                        else original_values.describe())
        with col2:
            st.write("**Después de normalización (muestra):**")
            st.dataframe(df_normalized[numeric_cols_to_normalize].describe())
        
        df_final = df_normalized
    else:
        df_final = df_encoded
        st.info("ℹ️ No se encontraron variables numéricas para normalizar")
    
    # Separar X y y
    st.subheader("6️⃣ Separación de Características (X) y Variable Objetivo (y)")
    
    if 'G3' in df_final.columns:
        target_col = 'G3'
    else:
        target_col = st.selectbox("Selecciona la variable objetivo:", 
                                 df_final.select_dtypes(include=[np.number]).columns)
    
    X = df_final.drop(columns=[target_col])
    y = df_final[target_col]
    
    st.success(f"✅ Separación completada. Features: {X.shape[1]}, Target: {target_col}")
    st.write(f"**X (características):** {X.shape}")
    st.write(f"**y (objetivo):** {y.shape}")
    
    # División en entrenamiento y prueba
    st.subheader("7️⃣ División en Conjuntos de Entrenamiento y Prueba")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    st.success("✅ Datos divididos en 80% entrenamiento y 20% prueba")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Conjunto de Entrenamiento:**")
        st.write(f"- X_train: {X_train.shape}")
        st.write(f"- y_train: {y_train.shape}")
        st.write(f"- Media de G3: {y_train.mean():.2f}")
    
    with col2:
        st.write("**Conjunto de Prueba:**")
        st.write(f"- X_test: {X_test.shape}")
        st.write(f"- y_test: {y_test.shape}")
        st.write(f"- Media de G3: {y_test.mean():.2f}")
    
    # Resumen final
    st.subheader("📊 Resumen Final")
    
    st.write("**Dataset procesado (primeras filas):**")
    summary_df = X_train.head().copy()
    summary_df['G3'] = y_train.head().values
    st.dataframe(summary_df)
    
    st.code(f"""
Resumen de dimensiones:
- Dataset original: {df.shape}
- Dataset procesado: {df_final.shape}
- X_train: {X_train.shape}
- X_test: {X_test.shape}
- y_train: {y_train.shape}
- y_test: {y_test.shape}
    """)

