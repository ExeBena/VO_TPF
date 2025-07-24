import pyrealsense2 as rs
import numpy as np
import cv2


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# config.enable_stream(rs.stream.depth)
# config.enable_stream(rs.stream.color)
profile = pipeline.start(config)

intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        frame = pipeline.wait_for_frames()

        aligned_frames = align.process(frame)

        frameColor_alineado = aligned_frames.get_color_frame()
        frameDepth_alineado = aligned_frames.get_depth_frame()

        # depth_frame = frames.get_depth_frame()
        # color_frame = frames.get_color_frame()
        if not frameDepth_alineado or not frameColor_alineado:
            continue

        depth_image = np.asanyarray(frameDepth_alineado.get_data())
        color_image = np.asanyarray(frameColor_alineado.get_data())

        # Process the depth and color images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=.15), cv2.COLORMAP_JET)
#        images = np.hstack((color_image, depth_image))
        
        # gray = cv2.convertScaleAbs(depth_colormap,alpha=255.0,beta=0.0)
        # ret,thresh1 = cv2.threshold(depth_colormap,127,255,cv2.THRESH_BINARY)

        # masked = cv2.bitwise_and(depth_image, depth_image, mask=thresh1)

        # Convertirlo a una matriz NumPy
        #depth_image = np.asanyarray(depth_frame.get_data())

        # Leer la distancia en milímetros del píxel (200, 300)
        px,py = 200, 300
        depth_mm = depth_image[py, px]
        print(f"Distancia en mm: {depth_mm}")

        # Convertir a metros si querés
        depth_m = depth_mm / 1000.0
        print(f"Distancia en metros: {depth_m:.2f}")

        # depth = frameDepth_alineado.get_distance(200, 300)
        depth = depth_image[py,px]/1000.
        X,Y,Z = rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], depth)

        print(f"X= {X:.2f} m, Y={Y:.2f} m, Z={Z:.2f} m")

        depth_image = cv2.circle(depth_image, (px, py), 5, (2**16-1, 2**16-1, 2**16-1),-1)



        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', depth_image)
        cv2.imshow('RealSense_color', color_image)


        #cv2.imshow('RealSense', [color_image, depth_colormap])
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
