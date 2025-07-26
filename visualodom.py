import pyrealsense2 as rs
import cv2
import numpy as np

bag_file = "../rosbag/20250723_120139.bag"

# Crear un pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configurar para reproducir desde archivo
rs.config.enable_device_from_file(config, bag_file,repeat_playback=False)

# Configurar para reproducir dsde camara
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Habilitar los streams necesarios (pueden ser depth, color, etc.)
config.enable_stream(rs.stream.depth)
config.enable_stream(rs.stream.color)


# Start the pipeline
profile = pipeline.start(config)    

align_to = rs.stream.color
align = rs.align(align_to)

#Setup parameters
shiTomasiCorrnerParams = dict( maxCorners = 150, qualityLevel = 0.3, minDistance = 2, blockSize = 9 )
lucasKanadeParams = dict( winSize = (15, 15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03) )

# Find features to track
frameFirst = pipeline.wait_for_frames()
frameGrayPrev = cv2.cvtColor(np.asanyarray(frameFirst.get_color_frame().get_data()), cv2.COLOR_BGR2GRAY)
cornersPrev = cv2.goodFeaturesToTrack(frameGrayPrev, mask = None, **shiTomasiCorrnerParams)
mask = np.zeros_like(np.asanyarray(frameFirst.get_color_frame().get_data()))

intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

K = np.array([[intrinsics.fx, 0, intrinsics.ppx],
              [0, intrinsics.fy, intrinsics.ppy],
              [0, 0, 1]], dtype=np.float64)
distCoeffs = np.array(intrinsics.coeffs, dtype=np.float64)

T= np.eye(4) # Translation vector
T_old= np.eye(4)
j = 0
traslacion = np.zeros((3, 1))  # Initialize translation vector
while True:
    j+=1
    # NUEVO FRAME
    try:
        frame = pipeline.wait_for_frames()
    except RuntimeError:
        print("Fin del stream")
        break

    # ALIENACION DE FRAME
    aligned_frames = align.process(frame)

    frameColor_alineado = aligned_frames.get_color_frame()
    frameDepth_alineado = aligned_frames.get_depth_frame()

    if not frameColor_alineado or not frameDepth_alineado:
        continue

    frameColor = np.asanyarray(frameColor_alineado.get_data())
    frameDepth = np.asanyarray(frameDepth_alineado.get_data())
    height, width = frameDepth.shape
    
    frameGrayCur = cv2.cvtColor(frameColor, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    cornersCur, status, error = cv2.calcOpticalFlowPyrLK(frameGrayPrev, frameGrayCur, cornersPrev, None, **lucasKanadeParams)


    # Si hay menos de 20 puntos, recalcular cornersPrev
    if cornersCur.shape[0]<20:        
        mascara = np.full(frameGrayPrev.shape, 255, dtype=np.uint8)
        cornersAux = cornersPrev.copy()
        for pt in cornersPrev.reshape(-1, 2):
            cv2.circle(mascara, (int(pt[0]),int(pt[1])), 5, 0, -1)

        # img_to_features = cv2.equalizeHist(frameGrayPrev)
        img_to_features = frameGrayPrev.copy()
        
        params = shiTomasiCorrnerParams.copy()
        cornersPrev = cv2.goodFeaturesToTrack(img_to_features, mask=mascara, **params)

        # si aún no se encuentran corners, reduzco qualityLevel
        while cornersPrev is None:
            params['qualityLevel'] *= 0.9
            cornersPrev = cv2.goodFeaturesToTrack(img_to_features, mask=mascara, **params)

        cornersPrev = np.append(cornersAux,cornersPrev,axis=0)
        cornersCur, status, error = cv2.calcOpticalFlowPyrLK(frameGrayPrev, frameGrayCur, cornersPrev, None, **lucasKanadeParams)
 
        frameGrayPrev = frameGrayCur.copy()
    # Select good points
    if cornersCur is not None:
        goodNew = cornersCur[status.flatten() == 1]
        goodOld = cornersPrev[status.flatten() == 1]

    vectors = []#np.array([])
    points3D = []
    points2D = []
 
    # Dibujo los puntos y armo los vectores 3D y 2D
    for i, (new, old) in enumerate(zip(goodNew, goodOld)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        frameColor = cv2.circle(frameColor, (int(a), int(b)), 5, (0, 0, 255), -1)

        # vectors.append([float(a), float(b), float(frameDepth[int(b),int(a)]/1000.)])
        u_int, v_int = int(c), int(d)
        if not (0 <= u_int < width and 0 <= v_int < height):
            continue
        depth = frameDepth_alineado.get_distance(u_int,v_int)
        # depth = frameDepth[v_int-2:v_int+2, u_int-2:u_int+2] / 1000.0  # Convert to meters


        X,Y,Z = rs.rs2_deproject_pixel_to_point(intrinsics, [u_int, v_int], depth)
        points3D.append([X, Y, Z])
        points2D.append([a, b])


    points3D = np.array(points3D, dtype=np.float32)
    points2D = np.array(points2D, dtype=np.float32)

    # Me aseguro de quedarme con los puntos positivos       
    validos = (points2D[:, 0] >= 0) & (points2D[:, 1] >= 0)
  
    # Muestro las imagenes
    img = cv2.add(cv2.cvtColor(frameColor, cv2.COLOR_RGB2BGR), mask)
    cv2.imshow('Optical Flow', img)
    cv2.imshow('Depth Matrix', frameDepth)

    # Update previous frame and corners
    frameGrayPrev = frameGrayCur.copy()
    cornersPrev = goodNew.reshape(-1, 1, 2)

    # Solve PnP para obtener pose, compruebo que haya mas de 4 puntos necesarios para
    # Ransac
    if len(points3D) >= 4 and len(points2D) >= 4:
        points3D = np.array(points3D, dtype=np.float32).reshape(-1, 3)
        points2D = np.array(points2D, dtype=np.float32).reshape(-1, 2)
        _, rvec, tvec, inliers = cv2.solvePnPRansac(points3D[validos], points2D[validos], K, distCoeffs)
        if rvec is not None and tvec is not None:
            # Convert rotation vector to rotation matrix
            R, _ = cv2.Rodrigues(rvec)

            #Transformacion homogenea
            T[:3,:3] = R.T
            T[:3,3] = tvec.ravel()
            traslacion += tvec.reshape(3, 1)

            T_old = T@T_old

            # print("Transformacion homogenea")
            print(T_old)
            
        else:
            print("No valid pose found.")
    else:
        continue

    if cv2.waitKey(30) & 0xFF == ord('q'):
        pipeline.stop()
        break

print("Matriz de transformacion homogenea total")
print(T_old)

cv2.destroyAllWindows()