import cv2
import concurrent.futures
import os

def load_image(image_path, format='RGB'):
    try:
        img = cv2.imread(image_path)
        if img is not None:
            if format == 'RGB':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        else:
            print(f"Error loading image {image_path}: Image is None")
            return None
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
    
def load_images(image_paths, max_workers=8):
    images = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_image = {executor.submit(load_image, path): path for path in image_paths}
        for future in concurrent.futures.as_completed(future_to_image):
            image = future.result()
            if image is not None:
                images.append(image)
    return images

def load_images_in_parallel(image_dir, max_workers=8):
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
    image_paths.sort()
    images = load_images(image_paths, max_workers=max_workers)
    return images