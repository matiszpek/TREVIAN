import trimesh
import numpy as np

# === CONFIGURACIÓN ===
ruta_stl = "foot.stl"  # Cambia esto por el nombre de tu archivo STL
umbral_normal = 0.9          # Coseno del ángulo con el eje Z negativo
porcentaje_z = 15            # Parte inferior del modelo (en %)

# === CARGAR MALLA ===
print("Cargando el modelo...")
mesh = trimesh.load(ruta_stl)
if not isinstance(mesh, trimesh.Trimesh):
    mesh = mesh.dump(concatenate=True)

print(f"Número de caras en la malla original: {len(mesh.faces)}")

# === CALCULAR NORMALES ===
face_normals = mesh.face_normals
z_down = np.array([0, 0, -1])
dot_products = face_normals @ z_down  # Producto punto entre normal y -Z

# === FILTRO POR DIRECCIÓN DE NORMAL ===
indices_por_normal = np.where(dot_products > umbral_normal)[0]
print(f"Caras que apuntan hacia abajo: {len(indices_por_normal)}")

""" # === FILTRO POR POSICIÓN Z ===
centros_caras = mesh.triangles_center
z_coords = centros_caras[:, 2]
umbral_z = np.percentile(z_coords, porcentaje_z)
indices_por_z = np.where(z_coords < umbral_z)[0]
print(f"Caras en el {porcentaje_z}% inferior del modelo: {len(indices_por_z)}")

# === COMBINAR AMBOS FILTROS ===
indices_finales = np.intersect1d(indices_por_normal, indices_por_z)
print(f"Caras que cumplen ambos criterios: {len(indices_finales)}")
 """

indices_finales=indices_por_normal

# === CREAR NUEVA MALLA ===
if len(indices_finales) == 0:
    print("⚠️ No se encontraron caras que cumplan los criterios. Ajusta los umbrales.")
else:
    nueva_malla = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces[indices_finales],
        process=False
    )

    # === GUARDAR O VISUALIZAR ===
    nueva_malla.export("planta_pie.stl")
    print("✅ Planta del pie exportada como 'planta_pie.stl'")
#    nueva_malla.show()
