#Alejandro Aristizabal y Paula Silva

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

print("Cargando librerías necesarias...")
# Cargar los datos
data3 = pd.read_csv("base_3_transaccional_b2b.txt", sep="\t")

# ------------------------------
# 3. Datos B2B (data3)
# ------------------------------
data_apriori3 = (
    data3.groupby(['id_b2b', 'fecha_factura'])['producto']
    .apply(list)
    .reset_index(name='productos')
)

# ------------------------------
#Reglas B2B
# ------------------------------

# Convert transactions into correct format (list of lists)
transactions_b2b = data_apriori3['productos'].tolist()

# Apply TransactionEncoder
te_b2b = TransactionEncoder()
data_encoded_b2b = te_b2b.fit_transform(transactions_b2b)
df_encoded_b2b = pd.DataFrame(data_encoded_b2b, columns=te_b2b.columns_)

# Run Apriori algorithm specifically for B2B dataset
frequent_itemsets_b2b = apriori(df_encoded_b2b, min_support=0.01, use_colnames=True)
rules_b2b = association_rules(frequent_itemsets_b2b, metric="confidence", min_threshold=0.8)

# Display rules specifically for B2B
print("\nAssociation Rules for B2B dataset:")


# ------------------------------
#Reglas de Transaccion y Cotizaciones 
# ------------------------------

# Expandir los productos según cantidad (equivalente a lo que hiciste en R)
data1_expanded = data1.loc[data1.index.repeat(data1.cantidad)].reset_index(drop=True)
data2_expanded = data2.loc[data2.index.repeat(data2.cantidad)].reset_index(drop=True)

# Agrupar por pedido y cotización, obteniendo listas únicas de productos (sin repetir dentro de cada pedido/cotización)
transactions1 = data1_expanded.groupby('pedido')['producto'].apply(set).tolist()
transactions2 = data2_expanded.groupby('cotizacion')['producto'].apply(set).tolist()

# Combinar transacciones
transactions_combined = transactions1 + transactions2

# Codificar transacciones usando matrices dispersas (sparse=True es clave aquí para reducir memoria)
te = TransactionEncoder()
te_matrix_sparse = te.fit_transform(transactions_combined, sparse=True)

# Crear DataFrame disperso
df_sparse = pd.DataFrame.sparse.from_spmatrix(te_matrix_sparse, columns=te.columns_)

# Ejecutar apriori con matrices dispersas y low_memory=True
frequent_itemsets = apriori(df_sparse, min_support=0.0001, use_colnames=True, low_memory=True)

# Generar reglas de asociación
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Mostrar reglas resultantes
print("Reglas de asociación (Soporte=0.001, Matriz Dispersa):")

# ------------------------------
#Funcion para recomendar Productos Relacionados
# ------------------------------



def obtener_productos_relacionados(rules_df, producto):
    # Filtrar reglas donde 'producto' esté en el antecedente (antecedents)
    reglas_filtradas = rules_df[rules_df['antecedents'].apply(lambda x: producto in x)]
    
    if not reglas_filtradas.empty:
        # Extraer productos del consecuente (consequents)
        productos_recomendados = set()
        for productos in reglas_filtradas['consequents']:
            productos_recomendados.update(productos)
        
        # Remover el producto original si aparece en recomendaciones
        productos_recomendados.discard(producto)

        if productos_recomendados:
            print("Productos recomendados:")
            for prod in productos_recomendados:
                print("-", prod)
        else:
            print("No se encontraron productos diferentes relacionados al producto ingresado.")
    else:
        print("No se encontraron productos relacionados con el producto ingresado.")

# Ejemplo de uso (asumiendo que ya generaste las reglas anteriormente)
producto = "producto_540"
obtener_productos_relacionados(rules, producto)

# ------------------------------
#Funcion para recomendar Productos B2B
# ------------------------------
def obtener_productos_relacionados_b2b(rules_b2b, producto, data):
    # Filtrar reglas donde 'producto' está en los antecedentes (antecedents)
    reglas_filtradas = rules_b2b[rules_b2b['antecedents'].apply(lambda x: producto in x)]
    
    if not reglas_filtradas.empty:
        # Obtener productos únicos del consecuente (consequents)
        productos_recomendados = set()
        for productos in reglas_filtradas['consequents']:
            productos_recomendados.update(productos)
        
        # Remover el producto original de las recomendaciones, si existe
        productos_recomendados.discard(producto)
        
        if productos_recomendados:
            print("Productos recomendados (B2B):")
            for prod in productos_recomendados:
                # Obtener el valor de la columna "alineación con portafolio estratégico b2b"
                alineacion = data[data['producto'] == prod]['alineación con portafolio estratégico b2b'].values
                alineacion_str = alineacion[0] if len(alineacion) > 0 else "No especificado"
                print(f"- {prod} (Alineación: {alineacion_str})")
        else:
            print("No se encontraron productos diferentes relacionados al producto ingresado (B2B).")
    else:
        print("No se encontraron productos relacionados con el producto ingresado (B2B).")

# Ejemplo de uso con reglas B2B generadas previamente
producto_b2b = "Producto_55"
obtener_productos_relacionados_b2b(rules_b2b, producto_b2b, data3)

# ------------------------------
#Algoritmo KNN
# ------------------------------

# Importar librerías necesarias
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

data1 = pd.read_csv("base_2_cotizaciones.txt", sep="\t")
# Asegurarse de que no haya valores faltantes en la columna 'id'
data1 = data1.dropna(subset=['id'])


# Convertir 'id' a cadenas si es necesario
data1['id'] = data1['id'].astype(str)


# Generar la matriz binaria de forma eficiente

# Crear índices únicos para clientes y productos
clientes_index = {cliente: idx for idx, cliente in enumerate(data1['id'].unique())}
productos_index = {producto: idx for idx, producto in enumerate(data1['producto'].unique())}


# Mapear los valores de 'id' y 'producto' a índices numéricos
rows = data1['id'].map(clientes_index).tolist()
cols = data1['producto'].map(productos_index).tolist()

# Crear una lista de valores (todas las celdas serán 1, indicando compra)
values = [1] * len(data1)

# Crear la matriz dispersa
matriz_binaria = csr_matrix((values, (rows, cols)))

# Verificar la matriz dispersa
print("Forma de la matriz dispersa:", matriz_binaria.shape)
print("Elementos no nulos en la matriz dispersa:")
print(matriz_binaria.nonzero())

# ------------------------------
# Entrenar el modelo KNN
# ------------------------------

# Crear el modelo KNN
knn = NearestNeighbors(n_neighbors=5, metric='cosine')  # Usamos similitud coseno
knn.fit(matriz_binaria)

# ------------------------------
# Función para generar recomendaciones
# ------------------------------
def recomendar_producto_a_clientes(producto_id, data, matriz_binaria, modelo_knn):
    """
    Encuentra clientes a los que se les puede recomendar un producto específico.

    :param producto_id: ID del producto a recomendar.
    :param data: Dataset original con columnas 'id' y 'producto'.
    :param matriz_binaria: Matriz dispersa de clientes x productos.
    :param modelo_knn: Modelo KNN entrenado.
    :return: Lista de IDs de clientes a los que se puede recomendar el producto.
    """
    # Verificar si el producto existe en el dataset
    if producto_id not in productos_index:
        print(f"El producto {producto_id} no está en el dataset.")
        return []

    # Obtener el índice del producto
    producto_idx = productos_index[producto_id]

    # Identificar clientes que ya compraron el producto
    clientes_que_compraron = data[data['producto'] == producto_id]['id'].astype(str).unique()

    # Identificar todos los clientes únicos y convertirlos a cadenas
    todos_los_clientes = set(data['id'].astype(str).unique())

    # Identificar clientes que no han comprado el producto
    clientes_sin_comprar = todos_los_clientes - set(clientes_que_compraron)

    # Inicializar el conjunto de clientes recomendados
    clientes_recomendados = set()

    # Procesar clientes que ya compraron
    for cliente_id in clientes_que_compraron:
        cliente_id = str(cliente_id)  # Asegurarse de que sea una cadena
        if cliente_id not in clientes_index:
            print(f"Cliente {cliente_id} no encontrado en clientes_index. Omitiendo...")
            continue

        # Obtener el índice del cliente
        cliente_idx = clientes_index[cliente_id]
        cliente_vector = matriz_binaria[cliente_idx]

        # Verificar si el cliente tiene vecinos usando KNN
        try:
            distancias, indices = modelo_knn.kneighbors(cliente_vector)
        except ValueError as e:
            print(f"Error con el cliente {cliente_id}: {e}")
            continue

        # Agregar vecinos a la lista de clientes recomendados si no han comprado el producto
        for vecino_idx in indices.flatten():
            vecino_id = list(clientes_index.keys())[vecino_idx]
            if vecino_id in clientes_sin_comprar:
                clientes_recomendados.add(vecino_id)

    return list(clientes_recomendados)

# Ejemplo de uso

# Reducir el dataset al 25% de los datos (por clientes o por filas)
data_muestra = data1.sample(frac=0.05, random_state=42)  # Cambia según la opción

# Usar el dataset reducido en la función
producto_ejemplo = "producto_27"
clientes_para_recomendar = recomendar_producto_a_clientes(producto_ejemplo, data_muestra, matriz_binaria, knn)

# Imprimir los resultados
print(f"Clientes a los que se puede recomendar el producto {producto_ejemplo} (con 1/4 del dataset):")
print(clientes_para_recomendar)

# ------------------------------
#KNN con diferente ejemplo de Uso
# ------------------------------

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder


# Cargar dataset de clientes (puedes usar data1 o data3, dependiendo del objetivo)
# Aquí asumimos que data1 es una tabla transaccional con columnas: 'cliente', 'producto'
clientes_data = data3[['id_b2b', 'producto']]  # Ignoramos la columna "cantidad"

# Crear una matriz binaria de clientes x productos (presencia/ausencia)
clientes_productos_binario = clientes_data.pivot_table(
    index='id_b2b',
    columns='producto',
    aggfunc=lambda x: 1,  # Indica que el producto fue comprado
    fill_value=0  # Llena las celdas vacías con 0
)

# Entrenar el modelo KNN

# Crear el modelo KNN
knn = NearestNeighbors(n_neighbors=5, metric='cosine')  # Usamos similitud coseno
knn.fit(clientes_productos_binario)

# ------------------------------
# Función para recomendar productos con KNN
# ------------------------------

def recomendar_con_knn(cliente_id, clientes_productos_binario, modelo_knn):
    """
    Encuentra clientes similares y recomienda productos basados en ellos.

    :param cliente_id: ID del cliente para el cual se generarán recomendaciones.
    :param clientes_productos_binario: Matriz binaria de clientes x productos.
    :param modelo_knn: Modelo KNN entrenado.
    :return: Lista de productos recomendados.
    """
    if cliente_id not in clientes_productos_binario.index:
        print(f"El cliente {cliente_id} no está en el dataset.")
        return []

    # Encontrar clientes similares
    cliente_vector = clientes_productos_binario.loc[cliente_id].values.reshape(1, -1)
    distancias, indices = modelo_knn.kneighbors(cliente_vector)

    # Extraer IDs de clientes más cercanos
    clientes_similares = clientes_productos_binario.index[indices.flatten()].tolist()

    # Obtener productos comprados por los clientes similares
    productos_similares = clientes_productos_binario.loc[clientes_similares].sum(axis=0)

    # Filtrar productos ya comprados por el cliente original
    productos_comprados = clientes_productos_binario.loc[cliente_id]
    productos_recomendados = productos_similares[productos_similares > 0].drop(productos_comprados[productos_comprados > 0].index)

    # Ordenar productos recomendados por frecuencia
    productos_recomendados = productos_recomendados.sort_values(ascending=False)

    # Retornar los productos recomendados
    return productos_recomendados.index.tolist()

# Ejemplo de uso

# ID del cliente para el cual queremos generar recomendaciones
cliente_ejemplo = "B2B_01"  # Cambiar por un cliente existente en el dataset

# Obtener recomendaciones
recomendaciones = recomendar_con_knn(cliente_ejemplo, clientes_productos_binario, knn)

if recomendaciones:
    print(f"Recomendaciones para el cliente {cliente_ejemplo}:")
    for producto in recomendaciones[:10]:  # Mostrar las 10 primeras recomendaciones
        print("-", producto)
else:
    print(f"No hay recomendaciones disponibles para el cliente {cliente_ejemplo}.")


# ------------------------------
# Búsqueda de hiperparámetros para reglas B2B
# ------------------------------
import random

# ------------------------------
# Muestreo de datos para reducir tamaño
# ------------------------------

# Muestreo para B2B (por ejemplo, 10,000 transacciones aleatorias)
sample_size_b2b = 10000
transactions_b2b_sample = random.sample(transactions_b2b, min(sample_size_b2b, len(transactions_b2b)))

# Muestreo para transacciones generales (por ejemplo, 10,000 transacciones aleatorias)
sample_size_general = 10000
transactions_combined_sample = random.sample(transactions_combined, min(sample_size_general, len(transactions_combined)))

# Codificar datos muestreados en matrices dispersas
te_b2b = TransactionEncoder()
te_matrix_sparse_b2b = te_b2b.fit_transform(transactions_b2b_sample, sparse=True)
df_sparse_b2b = pd.DataFrame.sparse.from_spmatrix(te_matrix_sparse_b2b, columns=te_b2b.columns_)

te = TransactionEncoder()
te_matrix_sparse_general = te.fit_transform(transactions_combined_sample, sparse=True)
df_sparse_general = pd.DataFrame.sparse.from_spmatrix(te_matrix_sparse_general, columns=te.columns_)

# ------------------------------
# Búsqueda de hiperparámetros para B2B
# ------------------------------

print("\n*** Búsqueda de hiperparámetros para reglas B2B ***")

support_values = [0.01, 0.001,0.0001]  # Reducido aún más
confidence_values = [0.5, 0.7,0.8,]

mejores_reglas_b2b = None
mejor_soporte_b2b = None
mejor_confianza_b2b = None
max_reglas_generadas_b2b = 0

for support in support_values:
    # Limitar el tamaño máximo de los conjuntos frecuentes (max_len=3)
    frequent_itemsets_b2b = apriori(df_sparse_b2b, min_support=support, use_colnames=True, max_len=3, low_memory=True)

    for confidence in confidence_values:
        rules_b2b = association_rules(frequent_itemsets_b2b, metric="confidence", min_threshold=confidence)
        num_reglas = len(rules_b2b)
        print(f"B2B - Soporte: {support}, Confianza: {confidence}, Reglas generadas: {num_reglas}")
        
        if num_reglas > max_reglas_generadas_b2b:
            max_reglas_generadas_b2b = num_reglas
            mejores_reglas_b2b = rules_b2b
            mejor_soporte_b2b = support
            mejor_confianza_b2b = confidence

print("\nMejores hiperparámetros para B2B:")
print(f"Soporte: {mejor_soporte_b2b}, Confianza: {mejor_confianza_b2b}, Reglas generadas: {max_reglas_generadas_b2b}")

# ------------------------------
# Búsqueda de hiperparámetros para reglas generales
# ------------------------------

print("\n*** Búsqueda de hiperparámetros para reglas generales ***")

mejores_reglas = None
mejor_soporte = None
mejor_confianza = None
max_reglas_generadas = 0

for support in support_values:
    frequent_itemsets = apriori(df_sparse_general, min_support=support, use_colnames=True, max_len=3, low_memory=True)

    for confidence in confidence_values:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence)
        num_reglas = len(rules)
        print(f"Generales - Soporte: {support}, Confianza: {confidence}, Reglas generadas: {num_reglas}")
        
        if num_reglas > max_reglas_generadas:
            max_reglas_generadas = num_reglas
            mejores_reglas = rules
            mejor_soporte = support
            mejor_confianza = confidence

print("\nMejores hiperparámetros para reglas generales:")
print(f"Soporte: {mejor_soporte}, Confianza: {mejor_confianza}, Reglas generadas: {max_reglas_generadas}")