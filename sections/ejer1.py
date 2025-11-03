import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
from io import StringIO

def main():
    st.header("🚢 Ejercicio 1: Análisis del Dataset 'Titanic'")
    
    st.markdown("""
    ### Objetivo
    Preparar los datos para un modelo que prediga la supervivencia de los pasajeros.
    """)
    
    # Cargar dataset
    st.subheader("1️⃣ Carga del Dataset")
    
    # Intentar cargar desde diferentes fuentes
    try:
        # Primero intentar desde seaborn
        df = sns.load_dataset('titanic')
        st.success("✅ Dataset cargado desde Seaborn")
    except:
        # Si no funciona, intentar cargar desde archivo
        uploaded_file = st.file_uploader("Cargar archivo titanic.csv", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Dataset cargado desde archivo")
        else:
            st.warning("⚠️ Por favor, carga el archivo titanic.csv o asegúrate de tener seaborn instalado")
            st.info("💡 El dataset de Titanic está disponible en Kaggle o puede cargarse usando `sns.load_dataset('titanic')`")
            return
    
    # Mostrar información inicial
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Dimensiones:** {df.shape[0]} filas × {df.shape[1]} columnas")
    with col2:
        st.write(f"**Columnas:** {', '.join(df.columns.tolist())}")
    
    st.write("**Primeras filas:**")
    st.dataframe(df.head())
    
    # Exploración inicial
    st.subheader("2️⃣ Exploración Inicial")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Información del dataset:**")
        buffer = StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)
        
        st.write("**Tipos de datos:**")
        types_df = pd.DataFrame({
            'Columna': df.columns,
            'Tipo': [str(dtype) for dtype in df.dtypes]
        })
        st.dataframe(types_df)
    
    with col2:
        st.write("**Valores nulos:**")
        nulls = df.isnull().sum()
        st.dataframe(nulls[nulls > 0].to_frame('Valores Nulos'))
        
        st.write("**Estadísticas descriptivas:**")
        st.dataframe(df.describe())
    
    # Limpieza de datos
    st.subheader("3️⃣ Limpieza de Datos")
    
    st.write("**Eliminando columnas irrelevantes (Name, Ticket, Cabin)...**")
    columns_to_drop = ['Name', 'Ticket', 'Cabin']
    existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    if existing_cols_to_drop:
        df_clean = df.drop(columns=existing_cols_to_drop)
        st.success(f"✅ Columnas eliminadas: {', '.join(existing_cols_to_drop)}")
    else:
        df_clean = df.copy()
        st.info("ℹ️ Algunas columnas no se encontraron en el dataset")
    
    st.write("**Reemplazando valores nulos...**")
    
    # Reemplazar nulos en columnas numéricas con la media
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            mean_val = df_clean[col].mean()
            df_clean[col].fillna(mean_val, inplace=True)
            st.write(f"  - {col}: reemplazado con media ({mean_val:.2f})")
    
    # Reemplazar nulos en columnas categóricas con la moda
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
            df_clean[col].fillna(mode_val, inplace=True)
            st.write(f"  - {col}: reemplazado con moda ({mode_val})")
    
    st.success("✅ Valores nulos reemplazados")
    
    st.write("**Verificando duplicados:**")
    duplicates = df_clean.duplicated().sum()
    if duplicates > 0:
        df_clean = df_clean.drop_duplicates()
        st.success(f"✅ {duplicates} duplicados eliminados")
    else:
        st.info("ℹ️ No se encontraron duplicados")
    
    st.write("**Dataset después de la limpieza:**")
    st.dataframe(df_clean.head())
    st.write(f"**Dimensiones finales:** {df_clean.shape[0]} filas × {df_clean.shape[1]} columnas")
    
    # Codificación de variables categóricas
    st.subheader("4️⃣ Codificación de Variables Categóricas")
    
    df_encoded = df_clean.copy()
    
    # Codificar Sex
    if 'Sex' in df_encoded.columns:
        le_sex = LabelEncoder()
        df_encoded['Sex'] = le_sex.fit_transform(df_encoded['Sex'])
        st.write(f"✅ Sex codificado: {dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))}")
    
    # Codificar Embarked
    if 'Embarked' in df_encoded.columns:
        le_embarked = LabelEncoder()
        df_encoded['Embarked'] = le_embarked.fit_transform(df_encoded['Embarked'].astype(str))
        st.write(f"✅ Embarked codificado: {dict(zip(le_embarked.classes_, le_embarked.transform(le_embarked.classes_)))}")
    
    # Codificar otras variables categóricas restantes
    other_categorical = df_encoded.select_dtypes(include=['object']).columns
    for col in other_categorical:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        st.write(f"✅ {col} codificado")
    
    st.write("**Dataset después de la codificación:**")
    st.dataframe(df_encoded.head())
    
    # Estandarización
    st.subheader("5️⃣ Estandarización de Variables Numéricas")
    
    # Identificar columnas numéricas (excluyendo la variable objetivo si existe)
    numeric_cols_to_scale = []
    for col in numeric_cols:
        if col in df_encoded.columns and col not in ['Survived']:  # Excluir target si existe
            numeric_cols_to_scale.append(col)
    
    if numeric_cols_to_scale:
        scaler = StandardScaler()
        df_scaled = df_encoded.copy()
        
        # Guardar valores originales para mostrar comparación
        original_values = df_scaled[numeric_cols_to_scale].copy()
        
        df_scaled[numeric_cols_to_scale] = scaler.fit_transform(df_scaled[numeric_cols_to_scale])
        
        st.write("**Variables estandarizadas:** " + ", ".join(numeric_cols_to_scale))
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Antes de estandarización:**")
            st.dataframe(original_values.describe())
        with col2:
            st.write("**Después de estandarización:**")
            st.dataframe(df_scaled[numeric_cols_to_scale].describe())
        
        df_final = df_scaled
    else:
        df_final = df_encoded
        st.info("ℹ️ No se encontraron variables numéricas para estandarizar")
    
    # División en entrenamiento y prueba
    st.subheader("6️⃣ División en Conjuntos de Entrenamiento y Prueba")
    
    # Identificar variable objetivo
    if 'Survived' in df_final.columns:
        target_col = 'Survived'
    else:
        target_col = st.selectbox("Selecciona la variable objetivo:", df_final.columns)
    
    X = df_final.drop(columns=[target_col])
    y = df_final[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y if y.dtype != 'float64' else None
    )
    
    st.success("✅ Datos divididos en 70% entrenamiento y 30% prueba")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Conjunto de Entrenamiento:**")
        st.write(f"- Filas: {X_train.shape[0]}")
        st.write(f"- Columnas: {X_train.shape[1]}")
        st.dataframe(X_train.head())
    
    with col2:
        st.write("**Conjunto de Prueba:**")
        st.write(f"- Filas: {X_test.shape[0]}")
        st.write(f"- Columnas: {X_test.shape[1]}")
        st.dataframe(X_test.head())
    
    # Salida esperada
    st.subheader("📊 Salida Esperada")
    
    st.write("**Primeros 5 registros procesados:**")
    processed_df = X_train.copy()
    processed_df['Target'] = y_train.values
    st.dataframe(processed_df.head())
    
    st.write("**Shape de los conjuntos:**")
    st.code(f"""
X_train.shape: {X_train.shape}
X_test.shape: {X_test.shape}
y_train.shape: {y_train.shape}
y_test.shape: {y_test.shape}
    """)

