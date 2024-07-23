import cv2
import mediapipe as mp

# Mediapipe çizim yardımcı fonksiyonları ve stillerini içe aktarma
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Mediapipe Hands çözümünü içe aktarma
mpHands = mp.solutions.hands

# OpenCV kullanarak varsayılan kameradan video akışını başlatma
cap = cv2.VideoCapture(0)

# Mediapipe Hands çözümünü başlatma
hands = mpHands.Hands()

while True:
    # Kameradan bir kare okuma
    data, image = cap.read()

    # Görüntüyü yatay olarak çevirme ve BGR'den RGB renk uzayına dönüştürme
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # Mediapipe Hands çözümünü kullanarak elleri tespit etme
    results = hands.process(image)

    # Görüntüyü tekrar BGR formatına dönüştürme
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Eğer eller tespit edilmişse, her el için landmarkları işle ve dikdörtgen çiz
    if results.multi_hand_landmarks:
        all_landmarks = []  # Tüm landmarkları saklamak için bir liste
        num_hands = len(results.multi_hand_landmarks)  # Tespit edilen el sayısı

        for hand_landmarks in results.multi_hand_landmarks:
            # Görüntü boyutlarını al: h (yükseklik), w (genişlik), c (kanal sayısı)
            h, w, c = image.shape

            # Her bir landmarkın koordinatlarını piksel cinsinden hesapla
            landmarks = [(int(point.x * w), int(point.y * h)) for point in hand_landmarks.landmark]
            all_landmarks.extend(landmarks)  # Tüm landmarkları listeye ekle

        if all_landmarks:  # Eğer herhangi bir landmark varsa
            # Landmarkların minimum ve maksimum x ve y koordinatlarını bul
            x_min = min([coord[0] for coord in all_landmarks])
            y_min = min([coord[1] for coord in all_landmarks])
            x_max = max([coord[0] for coord in all_landmarks])
            y_max = max([coord[1] for coord in all_landmarks])

            # Tespit edilen ellerin etrafına bir dikdörtgen çiz
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Tespit edilen el sayısını belirten bir metin ekle
            text = f'El: {num_hands}'
            cv2.putText(image, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # İşlenmiş görüntüyü gösterme
    cv2.imshow('HandTracker', image)

    # 'q' tuşuna basıldığında döngüyü sonlandırma
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera kaynağını serbest bırakma
cap.release()

# Açık olan tüm OpenCV pencerelerini kapatma
cv2.destroyAllWindows()
