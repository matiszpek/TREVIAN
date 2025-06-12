import trimesh
import numpy as np
from sklearn.decomposition import PCA

# === CONFIGURACIÓN ===
ruta_stl = "pie_escaneado.stl"
umbral_normal = 0.7  # coseno con el eje Z negativo

# === CARGAR MALLA ===
print("Cargando el modelo...")
mesh = trimesh.load(ruta_stl)
if not isinstance(mesh, trimesh.Trimesh):
    mesh = mesh.dump(concatenate=True)

print(f"Número de caras en la malla original: {len(mesh.faces)}")

# === PCA PARA REORIENTAR EL PIE ===
print("Aplicando PCA para reorientar el modelo...")

vertices = mesh.vertices
centroide = np.mean(vertices, axis=0)
vertices_centrados = vertices - centroide

pca = PCA(n_components=3)
pca.fit(vertices_centrados)
R = pca.components_.T  # columnas = nuevos ejes

# Detectar eje más alineado con Z
proyecciones = np.abs(R.T @ np.array([0, 0, 1]))
indice_z = np.argmax(proyecciones)

# Invertir si el eje Z estimado apunta hacia arriba (empeine abajo)
if R[:, indice_z][2] > 0:
    R[:, indice_z] *= -1

# Asegurar base ortonormal positiva
if np.linalg.det(R) < 0:
    R[:, -1] *= -1

# Crear y aplicar transformación
transformacion = np.eye(4)
transformacion[:3, :3] = R.T
transformacion[:3, 3] = -R.T @ centroide
mesh.apply_transform(transformacion)

# === EXPORTAR PIE ORIENTADO SIN FILTRO ===
mesh.export("pie_orientado.stl")
print("📦 Pie orientado exportado como 'pie_orientado.stl'")

# === CALCULAR NORMALES ===
face_normals = mesh.face_normals
z_down = np.array([0, 0, -1])
dot_products = face_normals @ z_down

# === FILTRO POR DIRECCIÓN DE NORMAL ===
indices_por_normal = np.where(dot_products > umbral_normal)[0]
print(f"Caras que apuntan hacia abajo: {len(indices_por_normal)}")

# === CREAR NUEVA MALLA CON FILTRO ===
if len(indices_por_normal) == 0:
    print("⚠️ No se encontraron caras que cumplan los criterios.")
else:
    nueva_malla = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces[indices_por_normal],
        process=False
    )
    nueva_malla.export("planta_pie.stl")
    print("✅ Planta del pie exportada como 'planta_pie.stl'")
