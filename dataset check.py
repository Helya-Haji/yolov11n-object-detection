import os
import cv2

input_dir = r""
output_dir = r""

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith('.bmp'):
        image_path = os.path.join(input_dir, filename)
        label_path = os.path.join(input_dir, filename.replace('.bmp', '.txt'))

        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id, cx, cy, w, h = map(float, parts[:5])

                        x = int((cx - w / 2) * width)
                        y = int((cy - h / 2) * height)
                        box_w = int(w * width)
                        box_h = int(h * height)
                        top_left = (x, y)
                        bottom_right = (x + box_w, y + box_h)

                        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
                        cv2.putText(image, str(int(class_id)), (x, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, image)

print("Annotation done and images saved.")
