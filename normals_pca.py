import trimesh
import numpy as np
from sklearn.decomposition import PCA

# === CONFIGURACIÓN ===
ruta_stl = "pie_escaneado.stl"
umbral_normal = 0.7

# === CARGAR MALLA ===
print("Cargando el modelo...")
mesh = trimesh.load(ruta_stl)
if not isinstance(mesh, trimesh.Trimesh):
    mesh = mesh.dump(concatenate=True)

print(f"Número de caras en la malla original: {len(mesh.faces)}")

# === PCA PARA REORIENTAR ===
print("Aplicando PCA para reorientar el modelo...")

# Obtener los vértices únicos
vertices = mesh.vertices
centroide = np.mean(vertices, axis=0)
vertices_centrados = vertices - centroide

# Aplicar PCA
pca = PCA(n_components=3)
pca.fit(vertices_centrados)

# Base ortonormal: los componentes principales
R = pca.components_.T  # columnas son las nuevas bases (autovectores)

# Asegurar una orientación correcta (determinante positivo)
if np.linalg.det(R) < 0:
    R[:, -1] *= -1

# Crear matriz de transformación homogénea 4x4 para aplicar a la malla
transformacion = np.eye(4)
transformacion[:3, :3] = R.T         # rotación inversa para alinear con XYZ
transformacion[:3, 3] = -R.T @ centroide  # translación al origen después de rotar

# Aplicar transformación a la malla
mesh.apply_transform(transformacion)

# === CALCULAR NORMALES ===
face_normals = mesh.face_normals
z_down = np.array([0, 0, -1])
dot_products = face_normals @ z_down

# === FILTRAR CARAS QUE APUNTAN HACIA ABAJO ===
indices_por_normal = np.where(dot_products > umbral_normal)[0]
print(f"Caras que apuntan hacia abajo: {len(indices_por_normal)}")

indices_finales = indices_por_normal

# === CREAR NUEVA MALLA ===
if len(indices_finales) == 0:
    print("⚠️ No se encontraron caras que cumplan los criterios. Ajusta los umbrales.")
else:
    nueva_malla = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces[indices_finales],
        process=False
    )

    # === EXPORTAR ===
    nueva_malla.export("planta_pie_orientada.stl")
    print("✅ Planta del pie orientada exportada como 'planta_pie_orientada.stl'")
#    nueva_malla.show()
