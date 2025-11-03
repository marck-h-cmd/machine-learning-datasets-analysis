import streamlit as st
from sections import ejer1, ejer2, ejer3

# Configuración de la página
st.set_page_config(
    page_title="Procesamiento de Datasets en Machine Learning",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo personalizado
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">📊 Procesamiento de Datasets en Machine Learning</h1>', unsafe_allow_html=True)

# Sidebar con navegación
st.sidebar.title("📚 Secciones")
st.sidebar.markdown("---")

# Botones de navegación en el sidebar
if st.sidebar.button("🚢 Ejercicio 1: Titanic", use_container_width=True, type="primary"):
    st.session_state.section = "ejer1"

if st.sidebar.button("📚 Ejercicio 2: Student Performance", use_container_width=True):
    st.session_state.section = "ejer2"

if st.sidebar.button("🌸 Ejercicio 3: Iris", use_container_width=True):
    st.session_state.section = "ejer3"

# Inicializar sección por defecto
if "section" not in st.session_state:
    st.session_state.section = "ejer1"

# Mostrar la sección seleccionada
st.markdown("---")

if st.session_state.section == "ejer1":
    ejer1.main()
elif st.session_state.section == "ejer2":
    ejer2.main()
elif st.session_state.section == "ejer3":
    ejer3.main()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Información")
st.sidebar.info("Selecciona una sección para comenzar a explorar el procesamiento de datasets en Machine Learning.")

