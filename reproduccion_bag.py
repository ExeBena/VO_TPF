import pyrealsense2 as rs
import cv2
import numpy as np

# Ruta al archivo .bag
# bag_file = "/home/exebena/Documentos/Doctorado/dockerRobotic/rosbag/rosbag2_2025_07_19-15_03_24_0.db3"
bag_file = "rosbag/20250721_123505.bag"

# Crear un pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configurar para reproducir desde archivo
rs.config.enable_device_from_file(config, bag_file)

# Habilitar los streams necesarios (pueden ser depth, color, etc.)
config.enable_stream(rs.stream.depth)
config.enable_stream(rs.stream.color)

# Iniciar pipeline
profile = pipeline.start(config)

# Procesar frames
try:
    while True:
        # Esperar un nuevo set de frames
        frames = pipeline.wait_for_frames()
        
        # Obtener frames de profundidad y color
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        # Convertir a matrices de numpy
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Mostrar imágenes
        cv2.imshow("Color", color_image)
        cv2.imshow("Depth", depth_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Liberar recursos
    pipeline.stop()
    cv2.destroyAllWindows()
