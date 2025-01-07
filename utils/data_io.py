import cv2
import concurrent.futures
import os

def load_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is not None:
            return img
        else:
            print(f"Error loading image {image_path}: Image is None")
            return None
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
    
def load_images_(image_paths, max_workers=8):
    images = [None] * len(image_paths)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(load_image, path): index for index, path in enumerate(image_paths)}
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            image = future.result()
            if image is not None:
                images[index] = image
    return images

def load_images(image_paths):
    images = [None] * len(image_paths)
    for index, path in enumerate(image_paths):
        image = load_image(path)
        if image is not None:
            images[index] = image
    return images

def load_images_in_parallel(image_dir, max_workers=8):
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
    image_paths.sort()
    images = load_images(image_paths, max_workers=max_workers)
    return images

# def load_image_test(image_path):
#     try:
#         img = cv2.imread(image_path)
#         if img is not None:
#             return img, image_path
#         else:
#             print(f"Error loading image {image_path}: Image is None")
#             return None
#     except Exception as e:
#         print(f"Error loading image {image_path}: {e}")
#         return None

# def load_images_test(image_paths, max_workers=8):
#     images = [None] * len(image_paths)
#     path_new = [None] * len(image_paths)
#     with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#         future_to_index = {executor.submit(load_image_test, path): index for index, path in enumerate(image_paths)}
#         for future in concurrent.futures.as_completed(future_to_index):
#             index = future_to_index[future]
#             image, path_ = future.result()
#             if image is not None:
#                 images[index] = image
#                 path_new[index] = os.path.basename(path_)
#     return images, path_new

# def load_images_in_parallel_test(image_dir, max_workers=8):
#     image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
#     image_paths.sort()
#     images, new_image_paths = load_images_test(image_paths, max_workers=max_workers)
#     return images, new_image_paths

# if __name__ == "__main__":
#     image_dir = '/cpfs01/shared/sport/donglinfeng/UnifiedBall/data/tabletennis/all/match1/frame/000'
#     images, new_image_paths = load_images_in_parallel_test(image_dir)
#     print(new_image_paths)
