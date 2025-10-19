import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
import pyttsx3
from datetime import datetime
import os

class SmartLeafDiseaseDetector:
    def __init__(self, model_name="linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"):
        print("Loading AI model... Please wait...")
        self.classifier = pipeline(
            "image-classification",
            model=model_name
        )
        print("Model loaded successfully!")

        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 150)

        # Only gating on leaf region
        self.min_leaf_coverage = 0.20   # require ≥20% green coverage

        self.report_data = []

    def has_leaf(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        coverage = cv2.countNonZero(green_mask) / (frame.shape[0] * frame.shape[1])
        return coverage >= self.min_leaf_coverage

    def predict_disease(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        return self.classifier(pil_image, top_k=3)

    def draw_results(self, frame, predictions):
        y = 30
        top = predictions[0]
        disease = top['label']
        conf = top['score'] * 100

        healthy = 'healthy' in disease.lower()
        color = (0, 255, 0) if healthy else (0, 0, 255)

        cv2.putText(frame, f"Disease: {disease}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Confidence: {conf:.1f}%", (10, y+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        y += 70
        cv2.putText(frame, "Top 3 Predictions:", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        for i, p in enumerate(predictions):
            y += 25
            cv2.putText(frame, f"{i+1}. {p['label']}: {p['score']*100:.1f}%",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        return frame, disease, conf, healthy

    def speak(self, msg):
        self.tts.say(msg)
        self.tts.runAndWait()

    def save_result(self, frame):
        os.makedirs('detections', exist_ok=True)
        fn = f"detections/{datetime.now():%Y%m%d_%H%M%S}.png"
        cv2.imwrite(fn, frame)
        print(f"Saved result: {fn}")

    def run_webcam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return

        print("Press SPACE to analyze, q to quit")

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (640, 480))

            cv2.putText(frame, "Press SPACE to analyze leaf", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow("Live", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                if not self.has_leaf(frame):
                    msg = "No leaf detected. Please position a leaf in view."
                    print(msg); self.speak(msg)
                    continue

                preds = self.predict_disease(frame)
                result_frame, disease, conf, healthy = self.draw_results(frame.copy(), preds)

                msg = (f"Leaf healthy, {conf:.1f}% confidence" 
                       if healthy else 
                       f"Disease: {disease}, {conf:.1f}% confidence")
                print(msg); self.speak(msg)

                self.save_result(result_frame)
                cv2.imshow("Result", result_frame)
                cv2.waitKey(3000)

            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def run_image(self, path):
        img = cv2.imread(path)
        if img is None:
            print("Error: Image not found"); return
        frame = cv2.resize(img, (640, 480))

        if not self.has_leaf(frame):
            msg = "No leaf detected in image."
            print(msg); self.speak(msg)
            return

        preds = self.predict_disease(frame)
        result_frame, disease, conf, healthy = self.draw_results(frame.copy(), preds)

        msg = (f"Leaf healthy, {conf:.1f}% confidence" 
               if healthy else 
               f"Disease: {disease}, {conf:.1f}% confidence")
        print(msg); self.speak(msg)

        self.save_result(result_frame)
        cv2.imshow("Result", result_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    detector = SmartLeafDiseaseDetector()
    choice = input("Mode: 1-Webcam, 2-Image file: ")
    if choice == '1':
        detector.run_webcam()
    elif choice == '2':
        path = input("Enter image path: ")
        detector.run_image(path)
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
