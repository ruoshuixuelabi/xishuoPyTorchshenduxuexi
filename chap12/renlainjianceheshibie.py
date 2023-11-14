
# # 基于dlib的人脸检测。
# import cv2
import dlib
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 读取图片
# img_path = r'./data/f170_1.png'
# img = cv2.imread(img_path)
# origin_img = img.copy()
# # 定义人脸检测器
# detector = dlib.get_frontal_face_detector()
# # 定义人脸关键点检测器
# predictor = dlib.shape_predictor(".\\shape_predictor_68_face_landmarks.dat")
# # 检测得到的人脸
# faces = detector(img, 0)
# # 如果存在人脸
# if len(faces):
#     print("Found %d faces in this image." % len(faces))
#     for i in range(len(faces)):
#         landmarks = np.matrix([[p.x, p.y] for p in predictor(img, faces[i]).parts()])
#         for point in landmarks:
#             pos = (point[0, 0], point[0, 1])
#             cv2.circle(img, pos, 1, color=(0, 255, 255), thickness=3)
# else:
#     print('Face not found!')
#
# cv2.namedWindow("Origin Face", cv2.WINDOW_FREERATIO)
# cv2.namedWindow("Detected Face", cv2.WINDOW_FREERATIO)
# cv2.imshow("Origin Face", origin_img)
# cv2.waitKey(0)
# cv2.imshow("Detected Face", img)
# cv2.waitKey(0)


# 基于facenet_pytorch实现人脸识别。
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1


# 获得人脸特征向量
def load_known_faces(dstImgPath, mtcnn, resnet):
    aligned = []
    knownImg = cv2.imread(dstImgPath)  # 读取图片
    face = mtcnn(knownImg)

    if face is not None:
        aligned.append(face[0])
    aligned = torch.stack(aligned).to(device)
    with torch.no_grad():
        known_faces_emb = resnet(aligned).detach().cpu()  # 使用resnet模型获取人脸对应的特征向量
    print("\n人脸对应的特征向量为：\n", known_faces_emb)
    return known_faces_emb, knownImg


# 计算人脸特征向量间的欧氏距离，设置阈值，判断是否为同一个人脸
def match_faces(faces_emb, known_faces_emb, threshold):
    isExistDst = False
    distance = (known_faces_emb[0] - faces_emb[0]).norm().item()
    print("\n两张人脸的欧氏距离为：%.2f" % distance)
    if (distance < threshold):
        isExistDst = True
    return isExistDst


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    mtcnn = MTCNN(min_face_size=12, thresholds=[0.2, 0.2, 0.3], keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    MatchThreshold = 0.8

    # known_faces_emb, _ = load_known_faces('./data/f170_2.png', mtcnn, resnet)
    # faces_emb, img = load_known_faces('./data/f170_1.png', mtcnn, resnet)
    known_faces_emb, _ = load_known_faces('./data/F_H27_2_A_00268_00.jpg', mtcnn, resnet)
    faces_emb, img = load_known_faces('./data/F_H98_5_A_00026_00.jpg', mtcnn, resnet)
    isExistDst = match_faces(faces_emb, known_faces_emb, MatchThreshold)
    print("设置的人脸特征向量匹配阈值为：", MatchThreshold)
    if isExistDst:
        boxes, prob, landmarks = mtcnn.detect(img, landmarks=True)
        print('由于欧氏距离小于匹配阈值，故匹配')
    else:
        print('由于欧氏距离大于匹配阈值，故不匹配')