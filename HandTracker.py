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

    # Eğer eller tespit edilmişse, her el için landmarkları çizme
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,  # Çizim yapılacak görüntü
                hand_landmarks,  # El landmarkları
                mpHands.HAND_CONNECTIONS  # El landmarkları arasındaki bağlantılar
            )

    # İşlenmiş görüntüyü gösterme
    cv2.imshow('HandTracker', image)

    # 'q' tuşuna basıldığında döngüyü sonlandırma
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera kaynağını serbest bırakma
cap.release()

# Açık olan tüm OpenCV pencerelerini kapatma
cv2.destroyAllWindows()
