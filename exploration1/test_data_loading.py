import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Charger les données
data_path = Path('input/Wafer_Map_Datasets.npz')
print(f"Chargement des données depuis {data_path}...")

try:
    data = np.load(data_path)
    print("\nClés disponibles dans le fichier:")
    print(data.files)
    
    # Afficher les informations sur les données
    wafer_maps = data['arr_0']  # Premier tableau pour les wafer maps
    failure_types = data['arr_1']  # Deuxième tableau pour les types de défauts
    
    print(f"\nNombre total d'échantillons: {len(wafer_maps)}")
    print(f"Forme des wafer maps: {wafer_maps.shape}")
    print(f"Forme des types de défauts: {failure_types.shape}")
    print("\nTypes de défauts uniques:")
    unique_types, counts = np.unique(failure_types, return_counts=True)
    for type_, count in zip(unique_types, counts):
        print(f"{type_}: {count} échantillons")
    
    # Visualiser quelques exemples
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.ravel()
    
    for i in range(8):
        idx = np.random.randint(0, len(wafer_maps))
        axes[i].imshow(wafer_maps[idx], cmap='gray')
        axes[i].set_title(f"Type: {failure_types[idx]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"❌ Erreur lors du chargement des données: {str(e)}")
    
    # Afficher plus d'informations sur l'erreur si disponible
    import traceback
    print("\nDétails de l'erreur:")
    print(traceback.format_exc()) 