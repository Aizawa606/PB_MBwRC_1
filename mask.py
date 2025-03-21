import cv2
import os

input_folder = "./DB1_B"
output_folder = "./DB1_B_MASKS"

# Upewnij się, że folder na maski istnieje
os.makedirs(output_folder, exist_ok=True)

# Przetwarzanie obrazów
for filename in os.listdir(input_folder):
    img_path = os.path.join(input_folder, filename)
    mask_path = os.path.join(output_folder, filename)  # Maska ma tę samą nazwę co oryginał

    # Wczytaj obraz w odcieniach szarości
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"❌ Błąd: Nie udało się wczytać {img_path}")
        continue

    # Binaryzacja metodą Otsu
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Zapisz maskę
    cv2.imwrite(mask_path, mask)

print("✅ Maski zostały wygenerowane i zapisane w:", output_folder)
