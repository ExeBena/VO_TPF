import pyrealsense2 as rs
import cv2
import numpy as np

def match_knn(des1, des2, k=2, ratio=0.75, base='flann'):
    # Validaciones
    if des1 is None or des2 is None:
        return []
    if len(des1) == 0 or len(des2) < k:
        return []

    # Config según el tipo de descriptor
    # if binary:
    #     # ORB/BRISK/BRIEF → uint8 + LSH
    #     FLANN_INDEX_LSH = 6
    #     index_params = dict(algorithm=6,table_number=12, key_size=20, multi_probe_level=2)
    #     search_params = dict(checks=50)
    #     # ¡No conviertas a float32!

    if base == 'flann':
        base = cv2.FlannBasedMatcher(dict(algorithm=6,table_number=12, key_size=20, multi_probe_level=2),
                                      dict(checks=50))
    elif base == 'bf':
        base = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=cross)
    # matches = bf.knnMatch(des1,des2,k=2)
    

    matches = base.knnMatch(des1, des2, k=k)

    good = []
    for m_n in matches:
        if len(m_n) < 2:   # <- importantísimo para evitar (DMatch,) 
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            good.append(m)
    return good




bag_file = "../rosbag/20250723_123323.bag"

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
orb = cv2.ORB_create(nfeatures=2000)

# Find features to track
frameFirst = pipeline.wait_for_frames()
frameGrayPrev = cv2.cvtColor(np.asanyarray(frameFirst.get_color_frame().get_data()), cv2.COLOR_BGR2GRAY)

kp1, des1 = orb.detectAndCompute(frameGrayPrev, None)

# cornersPrev = cv2.goodFeaturesToTrack(frameGrayPrev, mask = None, **shiTomasiCorrnerParams)
mask = np.zeros_like(np.asanyarray(frameFirst.get_color_frame().get_data()))

intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

K = np.array([[intrinsics.fx, 0, intrinsics.ppx],
              [0, intrinsics.fy, intrinsics.ppy],
              [0, 0, 1]], dtype=np.float64)
distCoeffs = np.array(intrinsics.coeffs, dtype=np.float64)

T= np.eye(4) # Translation vector
T_old= np.eye(4)
R_prev=np.eye(3) # Rotation matrix   
t_prev=np.zeros((3, 1)) # Translation vector
j = 0
traslacion = np.zeros((3, 1))  # Initialize translation vector

cross = False
while True:
    j+=1
    # Wait for a new frame
    try:
        frame = pipeline.wait_for_frames()
    except RuntimeError:
        print("Fin del stream")
        break

    aligned_frames = align.process(frame)

    frameColor_alineado = aligned_frames.get_color_frame()
    frameDepth_alineado = aligned_frames.get_depth_frame()

    if not frameColor_alineado or not frameDepth_alineado:
        continue

    frameColor = np.asanyarray(frameColor_alineado.get_data())
    frameDepth = np.asanyarray(frameDepth_alineado.get_data())
    height, width = frameDepth.shape
    
    frameGrayCur = cv2.cvtColor(frameColor, cv2.COLOR_BGR2GRAY)

    kp2, des2 = orb.detectAndCompute(frameGrayCur, None)

    
    vectors = []#np.array([])
    points3D = []
    points2D = []

    good = match_knn(des1, des2, k=2, ratio=0.75, base='bf')

    
    points3D, points2D = [], []
    for m in good:
        u_prev,v_prev = kp1[m.queryIdx].pt
        u, v = kp2[m.trainIdx].pt

        depth = frameDepth[int(v_prev),int(u_prev)] /1000.

        X,Y,Z = rs.rs2_deproject_pixel_to_point(intrinsics, [int(u_prev), int(v_prev)], depth)

        points3D.append([X, Y, Z])
        points2D.append([u,v])

        mask = cv2.line(mask, (int(u_prev),int(v_prev)), (int(u),int(v)), (0, 255, 0), 2)
        frameColor = cv2.circle(frameColor, (int(u),int(v)), 5, (0, 0, 255), -1)

    points3D = np.array(points3D, dtype=np.float32)
    points2D = np.array(points2D, dtype=np.float32)

    if points2D.shape[0]>1 and points3D.shape[0]>1:        
        validos = (points2D[:, 0] >= 0) & (points2D[:, 1] >= 0)
    else:
        continue
    

    img = cv2.add(cv2.cvtColor(frameColor, cv2.COLOR_RGB2BGR), mask)

    # Update previous frame and corners
    kp1, des1 = kp2, des2
    
    # Solve PnP para obtener pose
    if len(points3D) >= 4 and len(points2D) >= 4:

        points3D = np.array(points3D, dtype=np.float32).reshape(-1, 3)
        points2D = np.array(points2D, dtype=np.float32).reshape(-1, 2)
        _, rvec, tvec, inliers = cv2.solvePnPRansac(points3D[validos], points2D[validos], K, distCoeffs)
        if rvec is not None and tvec is not None:
            # Convert rotation vector to rotation matrix
            R, _ = cv2.Rodrigues(rvec)

            if(np.abs(np.trace(R_prev)- np.trace(R)) < 0.5 and np.linalg.norm(t_prev-tvec) < 3/30):
                

                #Transformacion homogenea
                T[:3,:3] = R.T
                T[:3,3] = tvec.ravel()
                traslacion += tvec.reshape(3, 1)

                T_old = T_old@T

                R_prev = R
                t_prev = tvec.reshape(3, 1)
                
                # print("Transformacion homogenea")
                # print(T_old)
            
            # else:
            #     print("Pose no válida, rotación o traslación demasiado grandes.")
            
        else:
            print("No valid pose found.")
    else:
        continue

    # Show the image
    cv2.imshow('Optical Flow', img)
    cv2.imshow('Depth Matrix', frameDepth)


    if cv2.waitKey(30) & 0xFF == ord('q'):
        pipeline.stop()
        break

print("Matriz de transformacion homogenea total")
print(T_old)
print(f"Rotacion sobre el eje Y: {(np.arcsin(T_old[0, 2])*180/np.pi):.2f}°")
print(f"Desplazamiento en X:{T_old[0, 3]:.2f}, en Y:{T_old[1, 3]:.2f}, en Z:{T_old[2, 3]:.2f}")
print("Tener en cuenta que el eje Y tiene dirección hacia abajo y la transforamción es del mundo respecto a la cámara.")

cv2.destroyAllWindows()